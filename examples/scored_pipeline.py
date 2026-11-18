"""Rank pipelines by how good their answers are - no API keys.

Uses the same five plain-function components as callable_pipeline.py.
There, two of the three frontier pipelines answered "Water boils." and
still counted as successes. Here an EvalSet supplies four questions and a
scorer, so each pipeline gets a measured quality, and the one that
actually answers the questions comes out on top.

Run from the repo root:

    python examples/scored_pipeline.py
"""

from callable_pipeline import IMPLEMENTATIONS, build_registry

from chatcortex import (
    AgentExecutor,
    CallableBackend,
    EvalSet,
    Example,
    ExhaustiveSynthesizer,
    TaskSpecification,
    evaluate_pipelines,
    quality_frontier,
)

# Each expected value is the fact a correct answer must contain. The
# third and fourth questions are reworded, which the strict retriever
# cannot handle.
EXAMPLES = [
    Example("What is the boiling point of water?", expected="100"),
    Example("What is the speed of light?", expected="299,792"),
    Example("At what temperature does water boil?", expected="100"),
    Example("How quickly does light travel?", expected="299,792"),
]


def contains_fact(output, expected):
    return expected in str(output)


def main():
    registry = build_registry()
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"]
    )
    frontier = ExhaustiveSynthesizer(registry).synthesize(task)
    executor = AgentExecutor(backend=CallableBackend(IMPLEMENTATIONS))
    eval_set = EvalSet(examples=EXAMPLES, scorer=contains_fact)

    evaluations = evaluate_pipelines(frontier, executor, eval_set)

    print(f"{len(evaluations)} pipelines, {len(EXAMPLES)} questions each")
    print()
    print(f"{'pipeline':58s} declared  measured  measured")
    print(f"{'':58s} reliab.   success   quality")
    for evaluation in sorted(evaluations, key=lambda e: -e.quality):
        print(
            f"{' -> '.join(evaluation.components):58s} "
            f"{evaluation.candidate.total_reliability:7.3f}   "
            f"{evaluation.success_rate:7.2f}   "
            f"{evaluation.quality:7.2f}"
        )

    best = quality_frontier(evaluations)[0]
    print()
    print(f"Best by measured quality: {' -> '.join(best.components)}")
    print(f"  quality {best.quality:.2f} at ${best.cost:.3f} per question")


if __name__ == "__main__":
    main()
