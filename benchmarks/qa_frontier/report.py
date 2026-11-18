"""Score every pipeline from the response cache and print the true frontier.

Runs offline: no keys, no network. It reads the replies collect.py stored
and evaluates every combination of context, model and postprocess step
with ChatCortex's own executor and scoring, as a full grid search would.
Pipelines whose replies are not all collected yet are listed as
incomplete.

    python -m benchmarks.qa_frontier.report --data path/to/squad-dev.json --questions 30

Use the same --data, --questions and --seed as for collect.py.
"""

import argparse
import itertools
import sys
from typing import Any, List, Sequence, Tuple

from chatcortex import (
    AgentExecutor,
    AgentGraph,
    BackendConfigurationError,
    CallableBackend,
    ComponentMetadata,
    EvalSet,
    Example,
    PipelineEvaluation,
    RoutedBackend,
    evaluate_pipelines,
    quality_frontier,
)
from chatcortex.execution.executor import ProbeResult
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate

from .qa import (
    CONTEXT_VARIANTS,
    DEFAULT_CACHE,
    FREE_MODELS,
    POSTPROCESS_VARIANTS,
    SAMPLE_DATA,
    ModelSpec,
    Question,
    ResponseCache,
    Retriever,
    context_step,
    load_questions,
    score,
)


class CachedModelBackend:
    """Serves model replies from the response cache, counting distinct lookups."""

    def __init__(self, cache: ResponseCache):
        self.cache = cache
        self.used = set()

    def __call__(self, metadata: ComponentMetadata, node_input: Any) -> ProbeResult:
        prompt = str(node_input)
        record = self.cache.get(metadata.name, prompt)
        if record is None:
            raise BackendConfigurationError(
                f"no collected reply for {metadata.name}; run collect.py first"
            )
        self.used.add((metadata.name, prompt))
        return ProbeResult(
            success=True,
            latency_ms=record["latency_ms"],
            cost=0.0,
            output=record["output"],
        )


def _metadata(name: str, capability: str, latency_ms: float, reliability: float):
    return ComponentMetadata(
        name=name,
        component_type="model" if capability == "answer" else "tool",
        capabilities=[capability],
        cost_per_call=0.0,
        avg_latency_ms=latency_ms,
        reliability_score=reliability,
        privacy_level="external" if capability == "answer" else "internal",
    )


def build_components(questions: Sequence[Question], models: Sequence[ModelSpec]):
    """Metadata per step, and the local functions for the non-model steps."""
    retriever = Retriever(q.context for q in questions)
    contexts = [_metadata(f"ctx_{n}", "context", 1.0, 1.0) for n in CONTEXT_VARIANTS]
    answers = [_metadata(m.name, "answer", m.declared_latency_ms, 0.9) for m in models]
    posts = [_metadata(f"post_{n}", "postprocess", 0.0, 1.0) for n in POSTPROCESS_VARIANTS]
    functions = {f"ctx_{n}": context_step(retriever, k) for n, k in CONTEXT_VARIANTS.items()}
    functions.update({f"post_{n}": f for n, f in POSTPROCESS_VARIANTS.items()})
    return contexts, answers, posts, functions


def all_pipelines(contexts, answers, posts) -> List[ArchitectureCandidate]:
    """Every combination, as a full grid search would try them."""
    candidates = []
    for steps in itertools.product(contexts, answers, posts):
        graph = AgentGraph()
        for index, metadata in enumerate(steps):
            graph.chain_append(f"{metadata.name}_{index}", metadata)
        candidates.append(
            ArchitectureCandidate(
                graph=graph,
                total_cost=graph.total_cost(),
                total_latency=graph.total_latency(),
                total_reliability=graph.aggregate_reliability(),
            )
        )
    return candidates


def eval_set_for(questions: Sequence[Question]) -> EvalSet:
    return EvalSet(
        examples=[Example(q.question, expected=q.answers) for q in questions],
        scorer=score,
    )


def grid_report(
    questions: Sequence[Question],
    cache: ResponseCache,
    models: Sequence[ModelSpec] = FREE_MODELS,
) -> Tuple[List[PipelineEvaluation], List[str]]:
    """Evaluate every complete pipeline; return evaluations and incomplete names."""
    contexts, answers, posts, functions = build_components(questions, models)
    cached = CachedModelBackend(cache)
    backend = RoutedBackend(
        {m.name: cached for m in models}, default=CallableBackend(functions)
    )
    executor = AgentExecutor(backend=backend)
    eval_set = eval_set_for(questions)

    complete, incomplete = [], []
    for candidate in all_pipelines(contexts, answers, posts):
        names = [m.name for _, m in candidate.graph.iter_components()]
        context_fn, model_name = functions[names[0]], names[1]
        if all(cache.get(model_name, context_fn(q.question)) for q in questions):
            complete.append(candidate)
        else:
            incomplete.append(" -> ".join(names))
    return evaluate_pipelines(complete, executor, eval_set), incomplete


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(SAMPLE_DATA))
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    args = parser.parse_args(argv)

    questions = load_questions(args.data, limit=args.questions, seed=args.seed)
    evaluations, incomplete = grid_report(questions, ResponseCache(args.cache))
    total = len(evaluations) + len(incomplete)
    print(f"{len(questions)} questions; {len(evaluations)} of {total} pipelines fully collected")
    if not evaluations:
        print("Nothing to score yet: run collect.py first.")
        return 0

    frontier = {id(e) for e in quality_frontier(evaluations)}
    print()
    print(f"  {'pipeline':56s} quality  latency     frontier")
    for evaluation in sorted(evaluations, key=lambda e: (-e.quality, e.latency_ms)):
        mark = "yes" if id(evaluation) in frontier else ""
        print(
            f"  {' -> '.join(evaluation.components):56s} {evaluation.quality:7.3f}  "
            f"{evaluation.latency_ms:8.0f} ms  {mark}"
        )
    if incomplete:
        print(f"\n{len(incomplete)} pipelines still need replies; rerun collect.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
