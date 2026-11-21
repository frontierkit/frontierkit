"""Check that a recorded choice still holds.

A choice between pipelines is made on one day. The model behind it can
change, slow down or disappear on another, and nothing tells you.
:func:`verify_decision` runs the chosen pipeline again on the same
examples and compares the fresh run with what the decision record holds::

    decision = load_decision("decision.json")
    check = verify_decision(decision, pipelines, executor, eval_set)
    check.status      # "valid", "quality_dropped", "slower", ...
    check.message     # one sentence saying what was found

The answer is one of:

``valid``
    No drop in quality was found, and it is not slower.
``quality_dropped``
    The same examples now score lower, by more than the tolerance and
    more than chance could explain.
``slower``
    Quality held, but answers now take much longer.
``unavailable``
    The pipeline could not be run: the service refused, timed out or
    no longer knows the model.
``not_comparable``
    What you asked to check is not what was recorded: other examples,
    a missing pipeline, or a step that now runs on something else.

How quality is compared. Each example's fresh score is set against its
recorded score, and a range is built for the average change that stays
valid however often it is looked at. Examples are asked one at a time,
in a shuffled order, and the check stops early once the range settles
the matter either way. ``valid`` therefore has two strengths, told apart
by ``confirmed``: either the change was shown to be within the
tolerance, or no drop was found with the examples there are. The second
is weaker, and ``detectable`` says how big a drop would have gone
unnoticed. Few examples only catch large drops.

How speed is compared. A run counts as slow when it takes more than
``slower_factor`` times the recorded median, and at least
``min_slowdown_ms`` longer, so that a step taking 2 ms instead of 1 is
not news. The status is ``slower`` only when clearly more than half the
runs are slow, so one stalled call does not raise an alarm.

Run this on a schedule and each run is a separate test. At the default
confidence of 0.99 expect about one false alarm in a hundred runs when
nothing has changed and answers vary from run to run.
"""

import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .decision import Decision, _eval_set_hash
from .evaluation.eval_set import EvalSet
from .execution.errors import SERVICE_ERRORS, STEP_FAILED
from .execution.executor import AgentExecutor
from .optimization.architecture_candidate import ArchitectureCandidate
from .stats.intervals import Interval, mean_interval, paired_difference
from .store.answer_log import LoggedBackend, identify

VALID = "valid"
QUALITY_DROPPED = "quality_dropped"
SLOWER = "slower"
UNAVAILABLE = "unavailable"
NOT_COMPARABLE = "not_comparable"

# Process exit codes for the command line. 2 is left to argument errors.
EXIT_CODES = {VALID: 0, QUALITY_DROPPED: 1, SLOWER: 3, UNAVAILABLE: 4, NOT_COMPARABLE: 5}

# Stop asking when this many examples in a row could not be run at all.
_GIVE_UP_AFTER = 3


@dataclass(frozen=True)
class Verification:
    """The outcome of one check.

    Attributes:
        status: ``"valid"``, ``"quality_dropped"``, ``"slower"``,
            ``"unavailable"`` or ``"not_comparable"``.
        message: One sentence saying what was found.
        components: The pipeline that was checked.
        asked: Examples run in this check.
        examples: Examples the decision was made on.
        quality_then: Recorded mean score on the examples asked.
        quality_now: Fresh mean score on the same examples.
        change: The average change in score, now minus then, with its
            range; ``None`` when nothing could be compared.
        confirmed: True when a ``valid`` result was shown to be within
            the tolerance, not merely not disproved.
        detectable: The smallest drop this check could have caught with
            the examples it asked; ``None`` when nothing was compared.
        latency_then_ms: Recorded median time per example.
        latency_now_ms: Fresh median time per example.
        errors: Failed runs by kind.
        label: The name the fresh answers carry in the answer log.
        time: When the check ran.
    """

    status: str
    message: str
    components: Tuple[str, ...]
    asked: int
    examples: int
    quality_then: Optional[float]
    quality_now: Optional[float]
    change: Optional[Interval]
    confirmed: bool
    detectable: Optional[float]
    latency_then_ms: Optional[float]
    latency_now_ms: Optional[float]
    errors: Dict[str, int]
    label: str
    time: str

    @property
    def ok(self) -> bool:
        """True when the status is ``"valid"``."""
        return self.status == VALID

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "components": list(self.components),
            "asked": self.asked,
            "examples": self.examples,
            "quality_then": self.quality_then,
            "quality_now": self.quality_now,
            "change": self.change.to_dict() if self.change else None,
            "confirmed": self.confirmed,
            "detectable": self.detectable,
            "latency_then_ms": self.latency_then_ms,
            "latency_now_ms": self.latency_now_ms,
            "errors": dict(self.errors),
            "label": self.label,
            "time": self.time,
        }


def _names(candidate: ArchitectureCandidate) -> Tuple[str, ...]:
    return tuple(metadata.name for _, metadata in candidate.graph.iter_components())


def _same(recorded: Any, current: Any) -> bool:
    return json.dumps(recorded, sort_keys=True, default=repr) == json.dumps(
        current, sort_keys=True, default=repr
    )


def _identity_change(
    recorded: Sequence[Dict[str, Any]], candidate: ArchitectureCandidate, backend: Any
) -> Optional[str]:
    """A sentence naming the first step that no longer runs on what was recorded."""
    current = [identify(backend, metadata) for _, metadata in candidate.graph.iter_components()]
    for before, now in zip(recorded, current):
        for key, value in before.items():
            if key not in now:
                continue  # the backend given now does not describe this
            if not _same(value, now[key]):
                return (
                    f"step '{before.get('component')}' ran on {key} {value!r} when the "
                    f"decision was recorded and would run on {now[key]!r} now"
                )
    return None


def verify_decision(
    decision: Decision,
    pipelines: Sequence[ArchitectureCandidate],
    executor: AgentExecutor,
    eval_set: EvalSet,
    tolerance: float = 0.05,
    confidence: float = 0.99,
    slower_factor: float = 2.0,
    max_latency_ms: Optional[float] = None,
    min_slowdown_ms: float = 100.0,
    min_examples: int = 5,
    seed: int = 0,
    label: Optional[str] = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Verification:
    """Run the chosen pipeline again and compare it with the record.

    Args:
        decision: The record, from :func:`~chatcortex.record_decision`
            or :func:`~chatcortex.load_decision`.
        pipelines: Pipelines to look the chosen one up in, by its
            component names; pass what you compared originally.
        executor: Runs the pipeline. If its backend is a
            :class:`~chatcortex.LoggedBackend`, the fresh answers are
            written to the same log under ``label`` and earlier answers
            are not reused; a log wrapped inside another backend is not
            detected, and would make this check read old answers.
        eval_set: The examples and scorer the decision was made on.
        tolerance: A drop in mean score up to this size is accepted.
        confidence: How sure the check must be before it reports a drop
            or a slowdown, and before it confirms there is none.
        slower_factor: A run is slow when it takes more than this many
            times the recorded median.
        max_latency_ms: If given, a run is also slow above this time.
        min_slowdown_ms: A run must also take this much longer than the
            recorded median to count as slow.
        min_examples: Ask at least this many before stopping early.
        seed: Fixes the order in which examples are asked.
        label: Name for this check's answers in the log. Defaults to
            ``"verify <time>"``.
        clock: Returns the current time; replace it in tests.

    Raises:
        ValueError: If the record was written by an older version that
            did not keep per-example scores, or an argument is out of
            range.
    """
    if not 0 <= tolerance < 1:
        raise ValueError(f"tolerance must be at least 0 and below 1, {tolerance!r} given")
    if slower_factor <= 1:
        raise ValueError(f"slower_factor must be above 1, {slower_factor!r} given")
    if min_examples < 1:
        raise ValueError(f"min_examples must be at least 1, {min_examples!r} given")

    chosen = decision.chosen
    recorded = chosen.get("examples")
    if not recorded or "scores" not in recorded:
        raise ValueError(
            "this decision record does not hold per-example scores; "
            "record it again with this version of chatcortex"
        )
    then_scores: List[float] = list(recorded["scores"])
    then_latencies: List[float] = list(recorded.get("latencies_ms") or [])
    components = tuple(chosen["components"])
    now = clock()
    stamp = now.isoformat(timespec="seconds")
    label = label if label is not None else f"verify {stamp}"

    def result(status: str, message: str, **found: Any) -> Verification:
        fields = {
            "asked": 0,
            "quality_then": None,
            "quality_now": None,
            "change": None,
            "confirmed": False,
            "detectable": None,
            "latency_then_ms": statistics.median(then_latencies) if then_latencies else None,
            "latency_now_ms": None,
            "errors": {},
        }
        fields.update(found)
        return Verification(
            status=status,
            message=message,
            components=components,
            examples=len(then_scores),
            label=label,
            time=stamp,
            **fields,
        )

    # --- is this the comparison that was recorded? ---------------------
    if len(eval_set.examples) != len(then_scores):
        return result(
            NOT_COMPARABLE,
            f"The decision was made on {len(then_scores)} examples and "
            f"{len(eval_set.examples)} were given.",
        )
    recorded_hash = decision.data.get("eval_set", {}).get("hash")
    if recorded_hash is not None and _eval_set_hash(eval_set) != recorded_hash:
        return result(
            NOT_COMPARABLE,
            "The examples given are not the ones the decision was made on.",
        )
    matches = [candidate for candidate in pipelines if _names(candidate) == components]
    if not matches:
        return result(
            NOT_COMPARABLE,
            f"No pipeline with the steps {list(components)} was given.",
        )
    candidate = matches[0]

    backend = executor.backend
    if isinstance(backend, LoggedBackend):
        backend = LoggedBackend(
            backend.backend, backend.log, versions=backend.versions, label=label
        )
        executor = AgentExecutor(backend=backend)
    changed = _identity_change(chosen.get("identities") or [], candidate, backend)
    if changed:
        return result(NOT_COMPARABLE, f"The setup changed: {changed}.")

    # --- ask the examples one at a time --------------------------------
    order = list(range(len(then_scores)))
    random.Random(seed).shuffle(order)
    now_scores: List[float] = []
    asked_then: List[float] = []
    now_latencies: List[float] = []
    errors: Dict[str, int] = {}
    service_failures = 0
    failures_in_a_row = 0
    change: Optional[Interval] = None
    settled: Optional[str] = None

    for index in order:
        example = eval_set.examples[index]
        run = executor.execute(candidate.graph, task_input=example.input)
        latency = sum(record["latency_ms"] for record in run.telemetry.records)
        if run.success:
            now_scores.append(eval_set.score(run.output, example.expected))
            now_latencies.append(latency)
            failures_in_a_row = 0
        else:
            kind = getattr(run, "error", None) or STEP_FAILED
            errors[kind] = errors.get(kind, 0) + 1
            now_scores.append(0.0)
            if kind in SERVICE_ERRORS:
                service_failures += 1
                failures_in_a_row += 1
            else:
                failures_in_a_row = 0
        asked_then.append(then_scores[index])

        if failures_in_a_row >= _GIVE_UP_AFTER and service_failures == len(now_scores):
            settled = UNAVAILABLE
            break
        if len(now_scores) < min_examples:
            continue
        change = paired_difference(now_scores, asked_then, confidence=confidence)
        if change.high < -tolerance:
            settled = QUALITY_DROPPED
            break
        if change.low > -tolerance:
            settled = VALID
            break

    asked = len(now_scores)
    if change is None and asked:
        change = paired_difference(now_scores, asked_then, confidence=confidence)
    found = {
        "asked": asked,
        "quality_then": sum(asked_then) / asked,
        "quality_now": sum(now_scores) / asked,
        "change": change,
        "detectable": None
        if change is None
        else tolerance + max(0.0, change.high - change.estimate),
        "latency_now_ms": statistics.median(now_latencies) if now_latencies else None,
        "errors": errors,
    }

    if settled == UNAVAILABLE or service_failures * 2 >= asked:
        kinds = ", ".join(f"{count} {kind}" for kind, count in sorted(errors.items()))
        reason = (
            "the model or address was not found, so it may have been retired or renamed"
            if errors.get("not_found")
            else "the service did not answer"
        )
        return result(
            UNAVAILABLE,
            f"The pipeline could not be run: {reason} ({kinds} in {asked} "
            f"example{'s' if asked != 1 else ''}).",
            **found,
        )

    if settled == QUALITY_DROPPED:
        return result(
            QUALITY_DROPPED,
            f"Quality dropped from {found['quality_then']:.2f} to {found['quality_now']:.2f} "
            f"on {asked} examples; the drop is between {-change.high:.2f} and "
            f"{-change.low:.2f}, above the tolerance of {tolerance:.2f}.",
            **found,
        )

    # --- quality held; is it slower? -----------------------------------
    if now_latencies and (then_latencies or max_latency_ms is not None):
        limits = []
        if then_latencies:
            usual = statistics.median(then_latencies)
            limits.append(max(slower_factor * usual, usual + min_slowdown_ms))
        if max_latency_ms is not None:
            limits.append(max_latency_ms)
        limit = min(limits)
        slow = [1.0 if latency > limit else 0.0 for latency in now_latencies]
        if mean_interval(slow, confidence=confidence).low > 0.5:
            was = (
                f" It took {statistics.median(then_latencies):.0f} ms when recorded."
                if then_latencies
                else ""
            )
            return result(
                SLOWER,
                f"Quality held, but it is slower: {sum(slow):.0f} of {len(slow)} runs took "
                f"more than {limit:.0f} ms, and the median is now "
                f"{found['latency_now_ms']:.0f} ms.{was}",
                **found,
            )

    confirmed = settled == VALID
    if confirmed:
        message = (
            f"Still valid: on {asked} examples the change in quality is between "
            f"{change.low:+.2f} and {change.high:+.2f}, within the tolerance of {tolerance:.2f}."
        )
    else:
        message = (
            f"No drop found: quality was {found['quality_then']:.2f} and is "
            f"{found['quality_now']:.2f} on {asked} examples. With this many examples a "
            f"drop smaller than about {found['detectable']:.2f} would go unnoticed."
        )
    return result(VALID, message, confirmed=confirmed, **found)
