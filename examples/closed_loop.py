"""The closed loop against a hidden ground truth - no API keys, no network.

Sixteen rounds of synthesize -> execute (simulated) -> calibrate ->
re-synthesize, with evidence-weighted calibration. The registry the loop starts from is *wrong* about four
components; the simulator draws outcomes from a separate ground-truth
registry that the synthesizer and the calibration never see. The final
table compares declared, calibrated and true values, so the demo shows
both what the loop learns and where it does not.

Run from the repo root:

    python examples/closed_loop.py
"""

import dataclasses

from frontierkit.execution.executor import AgentExecutor
from frontierkit.loop import run_closed_loop
from frontierkit.registry.capability_registry import CapabilityRegistry
from frontierkit.synthesis.progressive_pareto_beam_synthesizer import (
    ProgressiveParetoBeamSynthesizer,
)
from frontierkit.synthesis.task_specification import TaskSpecification

ITERATIONS = 16
# The declared value counts as this many observations: a component seen
# a few times barely moves, one seen often converges on what was measured.
PRIOR_STRENGTH = 20
# The earlier fixed-weight rule, kept for comparison.
FIXED_ALPHA = 0.4
SEEDS = range(20)

# Where the declared registry is wrong: component -> its true values.
# Every other component really behaves as declared.
HIDDEN_TRUTH = {
    "retrieval_speedster": {"reliability_score": 0.55},   # declared 0.78
    "retrieval_budget": {"reliability_score": 0.95},      # declared 0.82
    "generation_balanced": {"avg_latency_ms": 320.0},     # declared 180
    "verification_reliable": {"cost_per_call": 0.009},    # declared 0.006
}

FIELDS = ("reliability_score", "avg_latency_ms", "cost_per_call")


def build_ground_truth(declared):
    """The registry the simulator samples from; the loop never reads it."""
    truth = CapabilityRegistry()
    for metadata in declared.list_all():
        truth.register(
            dataclasses.replace(metadata, **HIDDEN_TRUTH.get(metadata.name, {}))
        )
    return truth


def mean_abs_error(registry, truth, field):
    components = truth.list_all()
    return sum(
        abs(getattr(registry.get(m.name), field) - getattr(m, field))
        for m in components
    ) / len(components)


def run_loop(registry, truth, task, seed, **calibration):
    return run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=lambda reg: ProgressiveParetoBeamSynthesizer(
            reg, base_beam_width=3, growth_factor=2.0
        ),
        executor=AgentExecutor(
            mode="probabilistic", seed=seed, ground_truth=truth
        ),
        iterations=ITERATIONS,
        **calibration,
    )


def main():
    registry = build_demo_registry()
    truth = build_ground_truth(registry)
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"],
        max_latency=1500.0,
    )

    records = run_loop(registry, truth, task, seed=42, prior_strength=PRIOR_STRENGTH)

    print(
        f"Closed loop, {ITERATIONS} iterations "
        f"(prior_strength={PRIOR_STRENGTH}, seed=42)"
    )
    print()
    print("round  probes  frontier  reliability range")
    for record in records:
        reliabilities = [o["reliability"] for o in record.frontier_objectives]
        print(
            f"{record.iteration:5d}  {record.trace_count:6d}  "
            f"{record.frontier_size:8d}  "
            f"[{min(reliabilities):.3f}, {max(reliabilities):.3f}]"
        )
    print(f"total probes: {sum(r.trace_count for r in records)}")
    print()

    final = records[-1].registry

    print("Mis-declared components: declared -> calibrated (truth)")
    for name, overrides in HIDDEN_TRUTH.items():
        for field, true_value in overrides.items():
            print(
                f"  {name:24s} {field:18s} "
                f"{getattr(registry.get(name), field):<8.4g} -> "
                f"{getattr(final.get(name), field):<8.4g} ({true_value:.4g})"
            )

    print()
    print("Mean absolute error vs truth, all components: declared -> calibrated")
    for field in FIELDS:
        before = mean_abs_error(registry, truth, field)
        after = mean_abs_error(final, truth, field)
        verdict = "better" if after < before else "worse"
        print(f"  {field:18s} {before:.4g} -> {after:.4g}  ({verdict})")

    # One seed is one draw. Reliability comes from pass/fail probes, so
    # judge the calibration rule over many seeds, not the run above.
    declared_error = mean_abs_error(registry, truth, "reliability_score")
    print()
    print(
        f"Reliability error over {len(SEEDS)} seeds "
        f"(declared registry: {declared_error:.4f})"
    )
    for label, calibration in (
        (f"fixed alpha={FIXED_ALPHA}", {"alpha": FIXED_ALPHA}),
        (f"prior_strength={PRIOR_STRENGTH}", {"prior_strength": PRIOR_STRENGTH}),
    ):
        errors = [
            mean_abs_error(
                run_loop(registry, truth, task, seed=seed, **calibration)[-1].registry,
                truth,
                "reliability_score",
            )
            for seed in SEEDS
        ]
        wins = sum(error < declared_error for error in errors)
        print(
            f"  {label:20s} mean {sum(errors) / len(errors):.4f}, "
            f"beats declared in {wins}/{len(errors)} runs"
        )


if __name__ == "__main__":
    from frontierkit.registry.demo_components import build_demo_registry

    main()
