"""Score pipelines on example inputs.

Declared cost, latency and reliability say nothing about whether a
pipeline's answers are any good. An :class:`EvalSet` pairs example inputs
with a scoring function, and :func:`evaluate_pipelines` runs each
candidate pipeline on every example and reports what was measured::

    eval_set = EvalSet(
        examples=[Example("What is 2 + 2?", expected="4")],
        scorer=lambda output, expected: 1.0 if expected in str(output) else 0.0,
    )
    evaluations = evaluate_pipelines(frontier, executor, eval_set)
    best = max(evaluations, key=lambda e: e.quality)

Quality is the mean score over all examples, counting a run that failed
part-way as 0. Cost and latency are means of what the executor recorded,
so they are measured values when the backend measures them.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..execution.executor import AgentExecutor
from ..optimization.architecture_candidate import ArchitectureCandidate

Scorer = Callable[[Any, Any], float]


@dataclass(frozen=True)
class Example:
    """One evaluation case: an input, and what a good output looks like."""

    input: Any
    expected: Any = None


@dataclass(frozen=True)
class EvalSet:
    """Example inputs plus the function that grades an output.

    Args:
        examples: The cases to run; at least one.
        scorer: ``scorer(output, expected) -> float`` in ``[0, 1]``,
            where 1 is a fully correct output.

    Raises:
        ValueError: If ``examples`` is empty.
        TypeError: If ``scorer`` is not callable.
    """

    examples: Sequence[Example]
    scorer: Scorer

    def __post_init__(self):
        if not self.examples:
            raise ValueError("an EvalSet needs at least one example")
        if not callable(self.scorer):
            raise TypeError(f"scorer must be callable, {self.scorer!r} given")

    def score(self, output: Any, expected: Any) -> float:
        """Grade one output; ``True``/``False`` count as 1 and 0.

        Raises:
            ValueError: If the scorer returns anything but a bool or a
                finite number in ``[0, 1]``.
        """
        value = self.scorer(output, expected)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            or value > 1.0
        ):
            raise ValueError(
                f"scorer must return a number in [0, 1], {value!r} returned"
            )
        return float(value)


@dataclass(frozen=True)
class PipelineEvaluation:
    """What one pipeline did across an :class:`EvalSet`.

    Attributes:
        candidate: The pipeline, with its declared metrics.
        components: Component names in execution order.
        runs: Number of examples it was run on.
        quality: Mean score in ``[0, 1]``; failed runs count as 0.
        success_rate: Share of runs in which every step succeeded.
        cost: Mean recorded cost per run.
        latency_ms: Mean recorded latency per run.
        traces: Every per-step telemetry record, for calibration.
    """

    candidate: ArchitectureCandidate
    components: Tuple[str, ...]
    runs: int
    quality: float
    success_rate: float
    cost: float
    latency_ms: float
    traces: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> dict:
        """JSON-safe summary (without the candidate graph or raw traces)."""
        return {
            "components": list(self.components),
            "runs": self.runs,
            "quality": self.quality,
            "success_rate": self.success_rate,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
        }


def evaluate_pipeline(
    candidate: ArchitectureCandidate,
    executor: AgentExecutor,
    eval_set: EvalSet,
) -> PipelineEvaluation:
    """Run one pipeline on every example and summarise what was measured."""
    if executor.backend is None:
        raise ValueError(
            "scoring needs real outputs: build the AgentExecutor with a "
            "backend (for example CallableBackend); the simulator produces none"
        )

    scores: List[float] = []
    successes = 0
    total_cost = 0.0
    total_latency = 0.0
    traces: List[Dict[str, Any]] = []

    for example in eval_set.examples:
        result = executor.execute(candidate.graph, task_input=example.input)
        records = result.telemetry.records
        traces.extend(dict(record) for record in records)
        total_cost += sum(record["cost"] for record in records)
        total_latency += sum(record["latency_ms"] for record in records)
        if result.success:
            successes += 1
            scores.append(eval_set.score(result.output, example.expected))
        else:
            scores.append(0.0)

    runs = len(eval_set.examples)
    return PipelineEvaluation(
        candidate=candidate,
        components=tuple(
            metadata.name for _, metadata in candidate.graph.iter_components()
        ),
        runs=runs,
        quality=sum(scores) / runs,
        success_rate=successes / runs,
        cost=total_cost / runs,
        latency_ms=total_latency / runs,
        traces=tuple(traces),
    )


def evaluate_pipelines(
    candidates: Sequence[ArchitectureCandidate],
    executor: AgentExecutor,
    eval_set: EvalSet,
) -> List[PipelineEvaluation]:
    """Evaluate each candidate on ``eval_set``, in the order given.

    Only the pipelines you pass are scored: a pipeline the synthesizer
    did not propose is never run, however good its answers would be.
    """
    return [
        evaluate_pipeline(candidate, executor, eval_set) for candidate in candidates
    ]


def _dominates(a: PipelineEvaluation, b: PipelineEvaluation) -> bool:
    no_worse = (
        a.cost <= b.cost and a.latency_ms <= b.latency_ms and a.quality >= b.quality
    )
    better = a.cost < b.cost or a.latency_ms < b.latency_ms or a.quality > b.quality
    return no_worse and better


def quality_frontier(
    evaluations: Sequence[PipelineEvaluation],
) -> List[PipelineEvaluation]:
    """The evaluations not beaten on measured cost, latency and quality.

    Returned highest quality first, then cheapest.
    """
    kept = [
        evaluation
        for evaluation in evaluations
        if not any(_dominates(other, evaluation) for other in evaluations)
    ]
    return sorted(kept, key=lambda e: (-e.quality, e.cost, e.latency_ms))
