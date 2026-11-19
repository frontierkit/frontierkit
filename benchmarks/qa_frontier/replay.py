"""Compare search strategies on the collected replies, without new calls.

How many model calls does each strategy spend before it has found the
pipelines worth choosing between? Every strategy is replayed against the
response cache, so this costs nothing to rerun. A call is one distinct
(model, prompt) pair: a real run would cache replies too, so asking the
same model the same prompt twice counts once.

Strategies:

- grid: try every pipeline, in a fixed order.
- random: try pipelines in random order (averaged over 20 seeds).
- chatcortex: ChatCortex's closed loop with an eval set. Each round it
  synthesizes a frontier from the registry, with measured quality as an
  objective, scores the pipelines it has not scored yet, and calibrates
  the registry from what it measured. Look-alike pipelines are all kept.
- chatcortex, explore N: the same loop also scoring the N untried
  pipelines with the highest predicted quality each round, so it keeps
  looking after its frontier is fully scored. --explore sets the values
  shown (default 1, 2 and 4).

The score at a budget is coverage: the share of the true quality frontier
(found by scoring every pipeline) that the strategy's own frontier
contains after spending at most that many calls.

    python -m benchmarks.qa_frontier.replay

Only models whose replies are fully collected take part. Use the same
--data, --questions and --seed as for collect.py.
"""

import argparse
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from chatcortex import (
    AgentExecutor,
    CapabilityRegistry,
    ExhaustiveSynthesizer,
    PipelineEvaluation,
    RoutedBackend,
    TaskSpecification,
    quality_frontier,
    run_closed_loop,
)

from .qa import (
    DEFAULT_CACHE,
    FREE_MODELS,
    SQUAD_SUBSET,
    ModelSpec,
    Question,
    ResponseCache,
    load_questions,
    needed_calls,
)
from .report import (
    CachedModelBackend,
    LocalStepBackend,
    build_components,
    eval_set_for,
    grid_report,
)

Components = Tuple[str, ...]

BUDGET_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
RANDOM_SEEDS = range(20)


@dataclass(frozen=True)
class Point:
    """A strategy's state after spending ``calls`` calls."""

    calls: int
    coverage: float
    best_quality: float


def complete_models(
    questions: Sequence[Question], cache: ResponseCache, models: Sequence[ModelSpec]
) -> List[ModelSpec]:
    """Models with every reply the grid needs already collected."""
    missing = {
        spec.name
        for spec, prompt in needed_calls(questions, models)
        if cache.get(spec.name, prompt) is None
    }
    return [model for model in models if model.name not in missing]


class Replay:
    """The true frontier and a call counter for any set of pipelines."""

    def __init__(self, questions, cache, models):
        self.questions = list(questions)
        self.cache = cache
        self.models = list(models)
        evaluations, incomplete = grid_report(self.questions, cache, self.models)
        assert not incomplete, "replay needs fully collected models"
        self.table: Dict[Components, PipelineEvaluation] = {
            e.components: e for e in evaluations
        }
        self.truth = {e.components for e in quality_frontier(evaluations)}
        self.true_best = max(e.quality for e in evaluations)
        *_, self.functions = build_components(self.questions, self.models)

    def calls_for(self, pipelines: Sequence[Components]) -> int:
        prompts = set()
        for components in pipelines:
            context = self.functions[components[0]]
            for question in self.questions:
                prompts.add((components[1], context(question.question)))
        return len(prompts)

    @property
    def total_calls(self) -> int:
        return self.calls_for(list(self.table))

    def point(self, tried: Sequence[Components]) -> Point:
        evaluations = [self.table[c] for c in tried]
        found = {e.components for e in quality_frontier(evaluations)}
        return Point(
            calls=self.calls_for(tried),
            coverage=len(found & self.truth) / len(self.truth),
            best_quality=max(e.quality for e in evaluations),
        )


def sequential(replay: Replay, order: Sequence[Components]) -> List[Point]:
    """Points after each pipeline, trying them in ``order``."""
    return [replay.point(order[: i + 1]) for i in range(len(order))]


def grid_strategy(replay: Replay) -> List[Point]:
    return sequential(replay, list(replay.table))


def random_strategy(replay: Replay, seed: int) -> List[Point]:
    order = list(replay.table)
    random.Random(seed).shuffle(order)
    return sequential(replay, order)


def chatcortex_strategy(
    replay: Replay,
    iterations: int = 8,
    prior_strength: float = 20.0,
    keep_ties: bool = True,
    explore: int = 0,
) -> List[Point]:
    """ChatCortex's closed loop with an eval set; a point per round."""
    contexts, answers, posts, functions = build_components(replay.questions, replay.models)
    registry = CapabilityRegistry()
    for metadata in (*contexts, *answers, *posts):
        registry.register(metadata)
    cached = CachedModelBackend(replay.cache)
    executor = AgentExecutor(
        backend=RoutedBackend(
            {m.name: cached for m in replay.models},
            default=LocalStepBackend(functions),
        )
    )
    records = run_closed_loop(
        registry=registry,
        task=TaskSpecification(required_capabilities=["context", "answer", "postprocess"]),
        synthesizer=lambda reg: ExhaustiveSynthesizer(reg, keep_ties=keep_ties),
        executor=executor,
        iterations=iterations,
        prior_strength=prior_strength,
        eval_set=eval_set_for(replay.questions),
        explore=explore,
    )
    tried: List[Components] = []
    points = []
    for record in records:
        for evaluation in record.evaluations:
            if evaluation.components not in tried:
                tried.append(evaluation.components)
        points.append(replay.point(tried))
    return points


def at_budget(points: Sequence[Point], budget: int) -> Point:
    """The best state reachable without spending more than ``budget``."""
    affordable = [p for p in points if p.calls <= budget]
    if not affordable:
        return Point(0, 0.0, 0.0)
    return max(affordable, key=lambda p: (p.coverage, p.best_quality, -p.calls))


def compare(
    replay: Replay, explore_values: Sequence[int] = (2,), iterations: int = 12
) -> Dict[str, Callable[[int], Tuple[float, float]]]:
    """Strategy name -> f(budget) giving (coverage, best quality found)."""
    grid = grid_strategy(replay)
    randoms = [random_strategy(replay, seed) for seed in RANDOM_SEEDS]
    loop = chatcortex_strategy(replay, iterations=iterations)
    exploring = {
        n: chatcortex_strategy(replay, iterations=iterations, explore=n)
        for n in explore_values
        if n > 0
    }

    def single(points):
        def at(budget):
            point = at_budget(points, budget)
            return point.coverage, point.best_quality

        return at

    def averaged(budget):
        points = [at_budget(run, budget) for run in randoms]
        return (
            sum(p.coverage for p in points) / len(points),
            sum(p.best_quality for p in points) / len(points),
        )

    strategies = {
        "grid": single(grid),
        "random (mean of 20)": averaged,
        "chatcortex": single(loop),
    }
    for n, points in exploring.items():
        strategies[f"chatcortex, explore {n}"] = single(points)
    return strategies, loop


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(SQUAD_SUBSET))
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--explore", default="1,2,4",
        help="comma-separated explore values for extra ChatCortex rows",
    )
    args = parser.parse_args(argv)
    explore_values = [int(v) for v in args.explore.split(",") if v.strip()]

    questions = load_questions(args.data, limit=args.questions, seed=args.seed)
    cache = ResponseCache(args.cache)
    models = complete_models(questions, cache, FREE_MODELS)
    if len(models) < 2:
        print(
            f"{len(models)} model(s) fully collected for these {len(questions)} questions; "
            "at least 2 are needed. Run collect.py first."
        )
        return 0

    replay = Replay(questions, cache, models)
    strategies, loop = compare(replay, explore_values)
    total = replay.total_calls
    print(
        f"{len(questions)} questions, {len(models)} models "
        f"({', '.join(m.name for m in models)}), {len(replay.table)} pipelines"
    )
    print(
        f"True frontier: {len(replay.truth)} pipelines; best quality {replay.true_best:.3f}; "
        f"scoring every pipeline takes {total} calls"
    )
    print()
    print("Share of the true frontier found, and best quality found, within a call budget")
    header = f"  {'strategy':24s}" + "".join(
        f"{f'{int(f * 100)}% ({round(total * f)})':>18s}" for f in BUDGET_FRACTIONS
    )
    print(header)
    for name, at in strategies.items():
        cells = ""
        for fraction in BUDGET_FRACTIONS:
            coverage, best = at(round(total * fraction))
            cells += f"{f'{coverage:.2f} / {best:.3f}':>18s}"
        print(f"  {name:24s}{cells}")
    print()
    final = loop[-1]
    print(
        f"Without exploration, ChatCortex's loop stopped after {final.calls} calls "
        f"({final.calls / total:.0%} of the grid) with coverage {final.coverage:.2f} "
        f"and best quality {final.best_quality:.3f}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
