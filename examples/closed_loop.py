"""The closed loop against a hidden ground truth - no API keys, no network.

Four rounds of synthesize -> execute (simulated) -> calibrate ->
re-synthesize. The registry the loop starts from is *wrong* about four
components; the simulator draws outcomes from a separate ground-truth
registry that the synthesizer and the calibration never see. The final
table compares declared, calibrated and true values, so the demo shows
both what the loop learns and where it does not.

Run from the repo root:

    python examples/closed_loop.py
"""

import dataclasses

from chatcortex.execution.executor import AgentExecutor
from chatcortex.loop import run_closed_loop
from chatcortex.registry.capability_registry import CapabilityRegistry
from chatcortex.synthesis.progressive_pareto_beam_synthesizer import (
    ProgressiveParetoBeamSynthesizer,
)
from chatcortex.synthesis.task_specification import TaskSpecification

ITERATIONS = 4
ALPHA = 0.4

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


def main():
    registry = build_demo_registry()
    truth = build_ground_truth(registry)
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"],
        max_latency=1500.0,
    )

    records = run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=lambda reg: ProgressiveParetoBeamSynthesizer(
            reg, base_beam_width=3, growth_factor=2.0
        ),
        executor=AgentExecutor(
            mode="probabilistic", seed=42, ground_truth=truth
        ),
        iterations=ITERATIONS,
        alpha=ALPHA,
    )

    print(f"Closed loop, {ITERATIONS} iterations (alpha={ALPHA}, seed=42)\n")
    previous = None
    for record in records:
        objectives = record.frontier_objectives
        min_cost = min(o["cost"] for o in objectives)
        max_cost = max(o["cost"] for o in objectives)
        min_lat = min(o["latency"] for o in objectives)
        max_lat = max(o["latency"] for o in objectives)
        min_rel = min(o["reliability"] for o in objectives)
        max_rel = max(o["reliability"] for o in objectives)
        steps = record.trace_count

        print(f"=== iteration {record.iteration} "
              f"(probes this round: {steps}) ===")
        print(
            f"  frontier: {record.frontier_size} architectures "
            f"({record.executed_candidates} executed)"
        )
        print(
            f"    cost         = [{min_cost:.4f}, {max_cost:.4f}]"
        )
        print(
            f"    latency_ms   = [{min_lat:.0f}, {max_lat:.0f}]"
        )
        print(
            f"    reliability  = [{min_rel:.3f}, {max_rel:.3f}]"
        )

        if previous is not None:
            d_cost = min_cost - previous[0]
            d_rel = max_rel - previous[1]
            print(
                f"    movement     : cheapest cost {d_cost:+.4f}, "
                f"best reliability {d_rel:+.3f} vs last round"
            )
        if record.updated_components:
            names = ", ".join(record.updated_components)
            truncated = names if len(names) <= 72 else names[:69] + "..."
            print(f"    calibrated   : {truncated}")
        else:
            print("    calibrated   : every measurement matched its prior")
        print()

        previous = (min_cost, max_rel)

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


if __name__ == "__main__":
    from chatcortex.registry.demo_components import build_demo_registry

    main()
