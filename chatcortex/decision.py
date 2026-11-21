"""Write down a choice between pipelines, with the evidence behind it.

After comparing pipelines you pick one. :func:`record_decision` turns
that comparison into a small JSON document that says:

- which pipeline was chosen, and exactly what each of its steps ran on
  (endpoint, model, options, or the version of your own function);
- what it was compared with, on which examples and with which scorer;
- how sure the result is: a range for every score, each pipeline
  compared with the chosen one example by example, and the chosen
  pipeline's score corrected for having been picked from many;
- what failed along the way, by kind;
- the chosen pipeline's score and time on every example, which is what
  :func:`~chatcortex.verify_decision` later compares a fresh run with.

::

    decision = record_decision(evaluations, eval_set, backend=backend)
    decision.save("decision.json")
    decision.chosen["components"], decision.contenders

Keep the file next to your results. It answers "what did you compare,
on what, and how do you know" without rerunning anything, and
``decision.shared`` lists the steps every compared pipeline had in
common, which is what shows that two approaches were tested on the same
model with the same settings.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .evaluation.eval_set import EvalSet, PipelineEvaluation
from .stats.intervals import mean_interval, paired_difference
from .stats.selection import best_of
from .store.answer_log import identify

FORMAT = 2


def _eval_set_hash(eval_set: EvalSet) -> Optional[str]:
    """A fingerprint of the examples, or ``None`` if they are not JSON data."""
    try:
        text = json.dumps(
            [[example.input, example.expected] for example in eval_set.examples],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _identities(evaluation: PipelineEvaluation, backend: Any) -> List[Dict[str, Any]]:
    return [
        identify(backend, metadata)
        for _, metadata in evaluation.candidate.graph.iter_components()
    ]


class Decision:
    """A recorded choice. Build one with :func:`record_decision`.

    Attributes:
        data: The whole record as JSON data.
    """

    def __init__(self, data: Dict[str, Any]):
        if not isinstance(data, dict) or "chosen" not in data or "compared" not in data:
            raise ValueError("not a decision record")
        self.data = data

    @property
    def chosen(self) -> Dict[str, Any]:
        """The chosen pipeline: components, identities, scores, failures."""
        return self.data["chosen"]

    @property
    def compared(self) -> List[Dict[str, Any]]:
        """Every other pipeline, each with its comparison to the chosen one."""
        return self.data["compared"]

    @property
    def contenders(self) -> List[List[str]]:
        """Components of the pipelines too close to the chosen one to call."""
        return [
            entry["components"]
            for entry in self.compared
            if entry["against_chosen"]["verdict"] == "too close to call"
        ]

    @property
    def shared(self) -> List[Dict[str, Any]]:
        """The steps every compared pipeline ran, identical in each."""
        return self.data["shared"]

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    def save(self, path: Union[str, Path]) -> Path:
        """Write the record as indented JSON and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path


def load_decision(path: Union[str, Path]) -> Decision:
    """Read a record written by :meth:`Decision.save`.

    Raises:
        ValueError: If the file is not a decision record.
    """
    return Decision(json.loads(Path(path).read_text(encoding="utf-8")))


def record_decision(
    evaluations: Sequence[PipelineEvaluation],
    eval_set: EvalSet,
    backend: Any = None,
    chosen: Optional[PipelineEvaluation] = None,
    confidence: float = 0.95,
    method: str = "betting",
    note: Optional[str] = None,
    resamples: int = 2000,
    seed: int = 0,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Decision:
    """Record which pipeline was chosen and the evidence for it.

    Args:
        evaluations: Every pipeline that was compared, from
            :func:`~chatcortex.evaluate_pipelines` on ``eval_set``.
        eval_set: The examples and scorer they were compared on.
        backend: The backend the pipelines ran on. When it describes
            itself (``OpenAIBackend``, ``RoutedBackend``,
            ``LoggedBackend``) the record says what each step ran on;
            without it, steps are identified by name only.
        chosen: The pipeline you picked. Defaults to the one with the
            highest quality.
        confidence: How often each reported range should hold the truth.
        method: ``"betting"`` or ``"bootstrap"``, as in
            :func:`~chatcortex.stats.mean_interval`.
        note: Free text kept in the record, e.g. why this pipeline.
        resamples, seed: For the resampling steps; the same seed gives
            the same record.
        clock: Returns the current time; replace it in tests.

    Raises:
        ValueError: If ``evaluations`` is empty, lacks per-example
            scores, or ``chosen`` is not one of them.
    """
    evaluations = list(evaluations)
    if not evaluations:
        raise ValueError("record_decision needs at least one evaluation")
    for evaluation in evaluations:
        if len(evaluation.scores) != len(eval_set.examples):
            raise ValueError(
                f"pipeline {list(evaluation.components)} has {len(evaluation.scores)} "
                f"scores for {len(eval_set.examples)} examples; pass evaluations "
                "made on this eval set"
            )

    selection = best_of(
        {index: evaluation.scores for index, evaluation in enumerate(evaluations)},
        confidence=confidence,
        method=method,
        resamples=resamples,
        seed=seed,
    )
    if chosen is None:
        chosen_index = selection.winner
    else:
        matches = [i for i, evaluation in enumerate(evaluations) if evaluation is chosen]
        if not matches:
            raise ValueError("chosen must be one of the evaluations passed in")
        chosen_index = matches[0]
    picked = evaluations[chosen_index]

    identities = [_identities(evaluation, backend) for evaluation in evaluations]

    def entry(index: int) -> Dict[str, Any]:
        evaluation = evaluations[index]
        return {
            "components": list(evaluation.components),
            "identities": identities[index],
            "runs": evaluation.runs,
            "quality": evaluation.quality,
            "interval": mean_interval(
                evaluation.scores,
                confidence=confidence,
                method=method,
                resamples=resamples,
                seed=seed,
            ).to_dict(),
            "quality_when_answered": evaluation.quality_when_answered,
            "success_rate": evaluation.success_rate,
            "errors": evaluation.error_counts,
            "latency_ms": evaluation.latency_ms,
            "cost": evaluation.cost,
            "metrics": dict(evaluation.metrics),
        }

    compared = []
    for index, evaluation in enumerate(evaluations):
        if index == chosen_index:
            continue
        difference = paired_difference(
            picked.scores,
            evaluation.scores,
            confidence=confidence,
            method=method,
            resamples=resamples,
            seed=seed,
        ).to_dict()
        difference["verdict"] = {
            "first is better": "chosen is better",
            "second is better": "chosen is worse",
        }.get(difference["verdict"], difference["verdict"])
        compared.append({**entry(index), "against_chosen": difference})

    # A step is shared when every pipeline ran it with the same identity.
    def frozen(identity: Dict[str, Any]) -> str:
        return json.dumps(identity, sort_keys=True, default=repr)

    shared = [
        identity
        for identity in identities[chosen_index]
        if all(
            frozen(identity) in {frozen(other) for other in others}
            for others in identities
        )
    ]

    from . import __version__

    chosen_entry = entry(chosen_index)
    chosen_entry["examples"] = {
        "scores": list(picked.scores),
        "latencies_ms": list(picked.latencies),
    }

    by_index: Dict[int, List[str]] = {
        index: list(evaluation.components) for index, evaluation in enumerate(evaluations)
    }
    best: Tuple[str, ...] = tuple(by_index[selection.winner])
    data = {
        "format": FORMAT,
        "chatcortex": __version__,
        "created": clock().isoformat(timespec="seconds"),
        "note": note,
        "eval_set": {
            "examples": len(eval_set.examples),
            "hash": _eval_set_hash(eval_set),
            "scorer": getattr(eval_set.scorer, "__name__", type(eval_set.scorer).__name__),
            "metrics": sorted(eval_set.metrics),
        },
        "confidence": confidence,
        "method": method,
        "chosen": chosen_entry,
        "selection": {
            "candidates": selection.candidates,
            "best": list(best),
            "chosen_is_best": chosen_index == selection.winner,
            "best_measured": selection.measured,
            "best_corrected": selection.corrected,
            "best_interval": selection.interval.to_dict(),
        },
        "compared": compared,
        "shared": shared,
    }
    return Decision(data)
