"""The closed loop: synthesize -> execute -> calibrate -> re-synthesize.

:func:`run_closed_loop` glues three already-shipped mechanisms into one
iterative drive: a synthesizer maps the *current* registry to a Pareto
frontier, an executor runs the frontier's architectures and produces
telemetry records, and :func:`frontierkit.telemetry.calibration.calibrate`
blends the measured success/latency/cost back into the registry — so the
next iteration synthesizes against a registry that remembers what actually
happened, not only what was declared.

Determinism: given a deterministic-or-seeded synthesizer, a seeded (or
deterministic) executor, and seeded synthesis budgets, two calls with the
same inputs produce iteration-for-iteration identical records. The seeded
executor's RNG advances across iterations, which is still deterministic —
it is state, not scheduling.

Component mapping: telemetry records carry graph *node ids*;
``calibrate`` resolves them through
:func:`frontierkit.graph.node_ids.component_name_from_node_id` and silently
ignores names it cannot match, so traces from graphs built from the
previous registry still calibrate the components they were produced from.

Calibration feedback is *iterative*: each iteration blends only the
latest traces into the previous iteration's calibration, so priors
compound toward empirical values across iterations at roughly
``1 - (1 - alpha) ** k`` after ``k`` iterations.

Evidence-weighted mode: with ``prior_strength`` set, every iteration
instead recalibrates the *declared* registry from all traces collected so
far, weighing each component by its record count (see ``calibrate``). The
estimate for a component is then its declared value pulled toward the
mean of everything observed, by exactly as much as the evidence supports.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from frontierkit.evaluation.eval_set import (
    EvalSet,
    PipelineEvaluation,
    evaluate_pipelines,
)
from frontierkit.execution.executor import AgentExecutor
from frontierkit.registry.capability_registry import CapabilityRegistry
from frontierkit.synthesis.base import Synthesizer
from frontierkit.synthesis.budget import SynthesisBudget
from frontierkit.synthesis.enumeration import enumerate_pipelines
from frontierkit.synthesis.task_specification import TaskSpecification
from frontierkit.telemetry.calibration import calibrate, calibrate_quality


@dataclass
class LoopIteration:
    """Everything one loop iteration observed, kept and serializable.

    ``registry`` holds the *calibrated* registry this iteration produced
    (the next iteration's synthesis input and the value users typically
    keep); ``to_dict()`` embeds it as the loader-shaped
    ``registry_snapshot`` so a full loop can be checkpointed to JSON.
    ``traces`` are the raw telemetry records behind the calibration.
    """

    iteration: int
    frontier_size: int
    frontier_objectives: List[Dict[str, float]]
    executed_candidates: int
    trace_count: int
    updated_components: List[str]
    traces: List[Dict[str, Any]]
    registry: CapabilityRegistry
    prior_registry: CapabilityRegistry
    # One entry per executed pipeline when the loop ran with an eval_set;
    # empty otherwise.
    evaluations: List[PipelineEvaluation] = field(default_factory=list)
    # Untried pipelines scored this round beyond the frontier (``explore``).
    explored_candidates: int = 0

    def to_dict(self) -> dict:
        """JSON-safe record including the calibrated registry snapshot."""
        return {
            "iteration": self.iteration,
            "frontier_size": self.frontier_size,
            "frontier_objectives": self.frontier_objectives,
            "executed_candidates": self.executed_candidates,
            "trace_count": self.trace_count,
            "updated_components": list(self.updated_components),
            "traces": [dict(trace) for trace in self.traces],
            "registry_snapshot": self.registry.to_dict(),
            "evaluations": [e.to_dict() for e in self.evaluations],
            "explored_candidates": self.explored_candidates,
        }


def _make_synthesizer(
    synthesizer: Union[type, Callable[[CapabilityRegistry], Synthesizer]],
    registry: CapabilityRegistry,
) -> Synthesizer:
    """Instantiate the synthesis strategy against the *current* registry.

    Accepts a synthesizer *class* (constructed with its defaults) or a
    one-argument factory such as
    ``lambda reg: BeamSynthesizer(reg, beam_width=5)``. A pre-built
    instance is rejected: its constructor bound it to the previous
    registry, so reusing it would silently synthesize against stale
    metadata and the loop would learn nothing.
    """
    if isinstance(synthesizer, type) and issubclass(synthesizer, Synthesizer):
        return synthesizer(registry)
    if callable(synthesizer):
        return synthesizer(registry)
    raise TypeError(
        "synthesizer must be a Synthesizer subclass or a "
        f"factory(registry) callable, {synthesizer!r} given; passing a "
        "pre-built instance would bind the loop to stale metadata"
    )


def run_closed_loop(
    registry: CapabilityRegistry,
    task: TaskSpecification,
    synthesizer: Union[type, Callable[[CapabilityRegistry], Synthesizer]],
    executor: AgentExecutor,
    iterations: int = 3,
    alpha: float = 0.3,
    prior_strength: Optional[float] = None,
    eval_set: Optional[EvalSet] = None,
    budget: Optional[SynthesisBudget] = None,
    candidates_per_iteration: Optional[int] = None,
    explore: int = 0,
) -> List[LoopIteration]:
    """Run the synthesize -> execute -> calibrate cycle ``iterations`` times.

    Args:
        registry: The declared (prior) registry; iteration 0 synthesizes
            against it unchanged.
        task: The synthesis problem, as for any synthesizer.
        synthesizer: A :class:`~frontierkit.synthesis.base.Synthesizer`
            subclass or ``factory(registry) -> Synthesizer`` callable. It
            is instantiated fresh against the *current* registry every
            iteration, because calibrate returns a new registry.
        executor: A shared :class:`AgentExecutor` instance (simulated or
            backend-backed); its seeded RNG advances across iterations,
            which keeps the whole run deterministic.
        iterations: How many loop rounds to run (>= 1).
        alpha: Blend weight passed to :func:`calibrate`; validated, so
            every round only executes under a valid blend. Non-finite
            values are rejected like any out-of-range value.
        prior_strength: Evidence-weighted calibration. When given, each
            round calibrates the declared ``registry`` from all traces
            so far with weight ``n / (n + prior_strength)`` per
            component, and ``alpha`` is ignored. Must be finite and
            ``>= 0``.
        eval_set: Example inputs and a scorer. When given, every selected
            pipeline is run on every example (the executor needs a
            backend), all of those runs feed calibration, and each
            iteration's ``evaluations`` holds the measured quality, cost
            and latency per pipeline. Measured quality also steers the
            search: each component's ``quality_score`` is estimated from
            every scored run so far (see ``calibrate_quality``; prior
            strength is ``prior_strength``, or 1 when that is unset), and
            the next round's frontier is synthesized with quality as an
            objective. A pipeline scored in an earlier round is not run
            again; its earlier evaluation is reused. Build the
            synthesizer with ``keep_ties=True`` (``ExhaustiveSynthesizer``)
            so look-alike alternatives are all tried.
        explore: With an ``eval_set``, also score this many untried
            pipelines per round beyond the frontier: those with the highest
            predicted quality, then the lowest predicted latency. Without
            it the loop stops trying new pipelines once every frontier
            member has been scored, even if an untried one would be
            better. ``0`` (default) disables it.
        candidates_per_iteration: Execute only the first ``N`` frontier
            members (objective-sorted, deterministic) instead of all of
            them; useful when executions are expensive. Must be ``>= 1``
            when given, and is validated before anything executes.
        budget: Optional per-iteration :class:`SynthesisBudget`; it is a
            frozen dataclass, so the same constraints apply every round
            without state leaking between iterations.
        candidates_per_iteration: Execute only the first ``N`` frontier
            members (objective-sorted, deterministic) instead of all of
            them; useful when executions are expensive.

    Returns:
        One :class:`LoopIteration` per round; iteration ``k``'s
        ``frontier_size``/``frontier_objectives`` describe the frontier
        synthesized from iteration ``k-1``'s calibrated registry, and
        ``registry`` is the registry calibrated *during* round ``k``.

    Raises:
        ValueError: If ``iterations < 1``, ``alpha`` is outside ``[0, 1]``,
            or ``candidates_per_iteration`` is given below 1 — all before
            any round runs — or if a round synthesizes an empty frontier
            (nothing to execute would stall the loop).
        TypeError: If ``synthesizer`` is neither a Synthesizer subclass
            nor a factory callable.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, {iterations} given")
    if candidates_per_iteration is not None and candidates_per_iteration < 1:
        raise ValueError(
            "candidates_per_iteration must be >= 1 when given, "
            f"{candidates_per_iteration} given"
        )
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], {alpha!r} given")
    if isinstance(explore, bool) or not isinstance(explore, int) or explore < 0:
        raise ValueError(f"explore must be an int >= 0, {explore!r} given")
    if explore and eval_set is None:
        raise ValueError("explore needs an eval_set: untried pipelines are judged by score")
    if prior_strength is not None and (
        not math.isfinite(prior_strength) or prior_strength < 0.0
    ):
        raise ValueError(
            f"prior_strength must be a finite number >= 0, {prior_strength!r} given"
        )

    all_traces: List[Dict[str, Any]] = []
    scored: Dict[tuple, PipelineEvaluation] = {}
    records: List[LoopIteration] = []
    current = registry
    for iteration in range(iterations):
        engine = _make_synthesizer(synthesizer, current)
        frontier = engine.synthesize(task, budget)
        if not frontier:
            raise ValueError(
                f"iteration {iteration}: the synthesizer returned an "
                "empty frontier; the task may be infeasible under the "
                "current registry and constraints"
            )

        selected = frontier
        if candidates_per_iteration is not None:
            selected = sorted(
                frontier,
                key=lambda c: (c.total_cost, c.total_latency, -c.total_reliability),
            )[:candidates_per_iteration]

        explored = 0
        if explore:
            chosen = {_components(c) for c in selected}
            untried = [
                c
                for c in enumerate_pipelines(current, task, seed=iteration)
                if _components(c) not in scored and _components(c) not in chosen
            ]
            untried.sort(
                key=lambda c: (
                    -c.total_quality,
                    c.total_latency,
                    c.total_cost,
                    -c.total_reliability,
                    _components(c),
                )
            )
            extra = untried[:explore]
            explored = len(extra)
            selected = list(selected) + extra

        traces: List[Dict[str, Any]] = []
        evaluations: List[PipelineEvaluation] = []
        if eval_set is None:
            for candidate in selected:
                traces.extend(
                    executor.execute(candidate.graph).telemetry.records
                )
        else:
            fresh = [c for c in selected if _components(c) not in scored]
            for evaluation in evaluate_pipelines(fresh, executor, eval_set):
                scored[evaluation.components] = evaluation
                traces.extend(evaluation.traces)
            evaluations = [scored[_components(c)] for c in selected]

        if prior_strength is None:
            calibrated = calibrate(current, traces, alpha=alpha)
        else:
            all_traces.extend(traces)
            calibrated = calibrate(
                registry, all_traces, prior_strength=prior_strength
            )
        if eval_set is not None:
            calibrated = calibrate_quality(
                calibrated,
                scored.values(),
                prior_strength=1.0 if prior_strength is None else prior_strength,
            )
        updated = [
            prior.name
            for prior in current.list_all()
            if _differs(calibrated.get(prior.name), prior)
        ]

        records.append(
            LoopIteration(
                iteration=iteration,
                frontier_size=len(frontier),
                frontier_objectives=[
                    {
                        "cost": c.total_cost,
                        "latency": c.total_latency,
                        "reliability": c.total_reliability,
                        "quality": c.total_quality,
                    }
                    for c in frontier
                ],
                executed_candidates=(
                    len(selected) if eval_set is None else len(fresh)
                ),
                trace_count=len(traces),
                updated_components=updated,
                traces=[dict(trace) for trace in traces],
                registry=calibrated,
                prior_registry=current,
                evaluations=evaluations,
                explored_candidates=explored,
            )
        )
        current = calibrated
    return records


def _components(candidate) -> tuple:
    return tuple(m.name for _, m in candidate.graph.iter_components())


def _differs(calibrated, prior) -> bool:
    """True when calibration changed any performance field.

    Exact comparison is safe: ``blend == prior`` exactly when the
    empirical value equals the prior, with no rounding involved.
    """
    return (
        calibrated.reliability_score != prior.reliability_score
        or calibrated.avg_latency_ms != prior.avg_latency_ms
        or calibrated.cost_per_call != prior.cost_per_call
        or calibrated.quality_score != prior.quality_score
    )
