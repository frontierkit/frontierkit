"""Measured quality as a search objective, and keeping tied candidates."""

import json
import math

import pytest

from chatcortex import (
    AgentExecutor,
    AgentGraph,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    EvalSet,
    Example,
    ExhaustiveSynthesizer,
    TaskSpecification,
    build_demo_registry,
    run_closed_loop,
)
from chatcortex.optimization.architecture_candidate import ArchitectureCandidate
from chatcortex.optimization.pareto import ParetoSet, dominates
from chatcortex.registry.loader import load_registry
from chatcortex.registry.package import save_registry
from chatcortex.telemetry.calibration import calibrate_quality


def _component(name, capability, latency=10.0, quality=None):
    return ComponentMetadata(
        name=name,
        component_type="tool",
        capabilities=[capability],
        cost_per_call=0.0,
        avg_latency_ms=latency,
        reliability_score=1.0,
        privacy_level="internal",
        quality_score=quality,
    )


def _candidate(names, cost=0.0, latency=1.0, reliability=1.0, quality=1.0):
    graph = AgentGraph()
    for index, name in enumerate(names):
        graph.chain_append(f"{name}_{index}", _component(name, name))
    return ArchitectureCandidate(graph, cost, latency, reliability, quality)


class _Evaluation:
    def __init__(self, components, quality, runs):
        self.components, self.quality, self.runs = components, quality, runs


def test_unmeasured_quality_counts_as_one_and_measured_multiplies():
    graph = AgentGraph()
    graph.chain_append("a_0", _component("a", "x"))
    graph.chain_append("b_1", _component("b", "y", quality=0.5))
    graph.chain_append("c_2", _component("c", "z", quality=0.8))
    assert graph.aggregate_quality() == pytest.approx(0.4)


def test_higher_quality_dominates_when_all_else_is_equal():
    better = _candidate(["a"], quality=0.9)
    worse = _candidate(["b"], quality=0.5)
    assert dominates(better, worse)
    assert not dominates(worse, better)
    faster_but_worse = _candidate(["c"], latency=0.5, quality=0.5)
    assert not dominates(better, faster_but_worse)
    assert not dominates(faster_but_worse, better)


def test_pareto_set_keeps_one_of_identical_candidates_unless_asked():
    a, b = _candidate(["a"]), _candidate(["b"])
    default = ParetoSet()
    assert default.add(a) and not default.add(b)
    assert len(default) == 1

    tied = ParetoSet(keep_ties=True)
    assert tied.add(a) and tied.add(b)
    assert [c.graph.list_nodes() for c in tied] == [["a_0"], ["b_0"]]


def test_calibrate_quality_splits_credit_and_weighs_by_runs():
    registry = CapabilityRegistry()
    for name in ("ctx", "model", "untouched"):
        registry.register(_component(name, name))
    # A two-component pipeline scoring 0.64 credits each part 0.8.
    calibrated = calibrate_quality(
        registry, [_Evaluation(("ctx", "model"), 0.64, 10)], prior_strength=10
    )
    assert calibrated.get("model").quality_score == pytest.approx((10 * 1.0 + 10 * 0.8) / 20)
    assert calibrated.get("untouched").quality_score is None
    assert registry.get("model").quality_score is None  # not mutated

    exact = calibrate_quality(
        registry, [_Evaluation(("ctx", "model"), 0.64, 10)], prior_strength=0
    )
    product = exact.get("ctx").quality_score * exact.get("model").quality_score
    assert product == pytest.approx(0.64)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"prior_strength": -1}, "prior_strength"),
        ({"prior_strength": math.nan}, "prior_strength"),
        ({"prior_quality": 1.5}, "prior_quality"),
    ],
)
def test_calibrate_quality_rejects_bad_priors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        calibrate_quality(CapabilityRegistry(), [], **kwargs)


def test_measured_quality_survives_save_and_load(tmp_path):
    registry = CapabilityRegistry()
    registry.register(_component("a", "x", quality=0.7))
    registry.register(_component("b", "y"))
    path = tmp_path / "registry.json"
    save_registry(registry, path)
    text = path.read_text(encoding="utf-8")
    assert text.count("quality_score") == 1  # only the measured one
    reloaded = load_registry(path)
    assert reloaded.get("a").quality_score == 0.7
    assert reloaded.get("b").quality_score is None


def test_out_of_range_quality_in_a_file_is_rejected(tmp_path):
    path = tmp_path / "registry.json"
    document = CapabilityRegistry()
    document.register(_component("a", "x"))
    data = document.to_dict()
    data["components"][0]["quality_score"] = 1.5
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="quality_score"):
        load_registry(path)


def _loop_setup():
    """A context step, then an answer step. The fastest pair is wrong."""
    registry = CapabilityRegistry()
    for name in ("ctx_none", "ctx_passage"):
        registry.register(_component(name, "context", latency=1.0))
    registry.register(_component("model_fast", "answer", latency=10.0))
    registry.register(_component("model_good", "answer", latency=50.0))
    calls = []

    def answer(name, right_with_passage):
        def run(prompt):
            calls.append((name, prompt))
            if right_with_passage and "passage" in prompt:
                return "42"
            return "no idea"

        return run

    backend = CallableBackend(
        {
            "ctx_none": lambda q: q,
            "ctx_passage": lambda q: q + " passage",
            "model_fast": answer("model_fast", False),
            "model_good": answer("model_good", True),
        }
    )
    eval_set = EvalSet(
        [Example("q1", "42"), Example("q2", "42")], lambda out, exp: out == exp
    )
    task = TaskSpecification(required_capabilities=["context", "answer"])
    return registry, task, AgentExecutor(backend=backend), eval_set, calls


def _run(iterations):
    registry, task, executor, eval_set, calls = _loop_setup()
    records = run_closed_loop(
        registry=registry,
        task=task,
        synthesizer=lambda reg: ExhaustiveSynthesizer(reg, keep_ties=True),
        executor=executor,
        iterations=iterations,
        prior_strength=1.0,
        eval_set=eval_set,
    )
    return records, calls


def test_quality_steers_the_loop_to_the_best_pipeline():
    records, _ = _run(4)
    scored = {e.components: e for r in records for e in r.evaluations}
    best = max(scored.values(), key=lambda e: e.quality)
    assert best.components == ("ctx_passage", "model_good")
    assert best.quality == 1.0
    final = records[-1].registry
    assert final.get("model_good").quality_score > final.get("model_fast").quality_score


def test_a_scored_pipeline_is_never_run_twice():
    _, calls = _run(5)
    assert calls and len(calls) == len(set(calls))


def test_without_quality_measurements_the_search_is_unchanged():
    task = TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"],
        max_latency=1500.0,
    )
    frontier = ExhaustiveSynthesizer(build_demo_registry()).synthesize(task)
    assert all(c.total_quality == 1.0 for c in frontier)
    keys = [(c.total_cost, c.total_latency, c.total_reliability) for c in frontier]
    assert keys == sorted(set(keys))
