"""EvalSet scoring: pipelines ranked by what their outputs are worth."""

import json

import pytest

from frontierkit import (
    AgentExecutor,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    EvalSet,
    Example,
    ExhaustiveSynthesizer,
    StepResult,
    TaskSpecification,
    evaluate_pipelines,
    quality_frontier,
    run_closed_loop,
)

# A toy task with a known best pipeline. "double" is the correct first
# step; "add_one" is cheaper but wrong, and "flaky" fails on odd inputs.
# For the second step "exact" keeps the number and "rounded" loses it.
IMPLEMENTATIONS = {
    "double": lambda n: n * 2,
    "add_one": lambda n: n + 1,
    "flaky": lambda n: n * 2 if n % 2 == 0 else StepResult(None, success=False),
    "exact": lambda n: f"answer {n}",
    "rounded": lambda n: f"answer {round(n, -1)}",
}

# name, capability, declared cost, latency, reliability
DECLARED = [
    ("double", "compute", 0.003, 30, 0.95),
    ("add_one", "compute", 0.001, 10, 0.90),
    ("flaky", "compute", 0.002, 20, 0.93),
    ("exact", "format", 0.002, 20, 0.95),
    ("rounded", "format", 0.001, 10, 0.90),
]

BEST = ("double", "exact")


def _registry(flaky_declared=0.93):
    registry = CapabilityRegistry()
    for name, capability, cost, latency, reliability in DECLARED:
        if name == "flaky":
            reliability = flaky_declared
        registry.register(
            ComponentMetadata(
                name=name,
                component_type="tool",
                capabilities=[capability],
                cost_per_call=cost,
                avg_latency_ms=latency,
                reliability_score=reliability,
                privacy_level="internal",
            )
        )
    return registry


def _task():
    return TaskSpecification(required_capabilities=["compute", "format"])


def _eval_set():
    return EvalSet(
        examples=[Example(n, expected=f"answer {n * 2}") for n in (1, 2, 3, 4)],
        scorer=lambda output, expected: output == expected,
    )


def _executor():
    return AgentExecutor(backend=CallableBackend(IMPLEMENTATIONS))


def _by_components(evaluations):
    return {evaluation.components: evaluation for evaluation in evaluations}


def test_quality_is_the_mean_score_and_failed_runs_count_as_zero():
    frontier = ExhaustiveSynthesizer(_registry()).synthesize(_task())
    scored = _by_components(evaluate_pipelines(frontier, _executor(), _eval_set()))

    assert scored[BEST].quality == 1.0
    assert scored[BEST].success_rate == 1.0
    assert scored[BEST].runs == 4

    # flaky fails on the two odd inputs: half the runs score, half are 0.
    flaky = scored[("flaky", "exact")]
    assert flaky.success_rate == 0.5
    assert flaky.quality == 0.5

    # add_one runs fine and is always wrong.
    assert scored[("add_one", "rounded")].success_rate == 1.0
    assert scored[("add_one", "rounded")].quality == 0.0


def test_each_example_keeps_its_own_score_and_output():
    frontier = ExhaustiveSynthesizer(_registry()).synthesize(_task())
    eval_set = _eval_set()
    scored = _by_components(evaluate_pipelines(frontier, _executor(), eval_set))

    # In example order (inputs 1, 2, 3, 4); flaky fails on the odd ones.
    flaky = scored[("flaky", "exact")]
    assert flaky.scores == (0.0, 1.0, 0.0, 1.0)
    assert flaky.outputs == (None, "answer 4", None, "answer 8")
    assert scored[BEST].outputs == ("answer 2", "answer 4", "answer 6", "answer 8")

    for evaluation in scored.values():
        assert len(evaluation.scores) == len(evaluation.outputs) == evaluation.runs
        assert evaluation.quality == pytest.approx(sum(evaluation.scores) / evaluation.runs)

    # The kept outputs can be graded again without running the pipeline.
    # add_one is right only for input 1, where 1 + 1 equals 1 * 2.
    wrong = scored[("add_one", "exact")]
    assert wrong.scores == (1.0, 0.0, 0.0, 0.0)
    lenient = EvalSet(eval_set.examples, scorer=lambda output, expected: output[:6] == expected[:6])
    regraded = [
        lenient.score(output, example.expected)
        for output, example in zip(wrong.outputs, eval_set.examples)
    ]
    assert regraded == [1.0, 1.0, 1.0, 1.0]


def test_cost_is_the_mean_recorded_cost_per_run():
    frontier = ExhaustiveSynthesizer(_registry()).synthesize(_task())
    scored = _by_components(evaluate_pipelines(frontier, _executor(), _eval_set()))
    assert scored[BEST].cost == pytest.approx(0.005)
    # flaky: two full runs (0.002 + 0.002) and two that stop after step one.
    assert scored[("flaky", "exact")].cost == pytest.approx((0.004 * 2 + 0.002 * 2) / 4)


def test_quality_frontier_drops_beaten_pipelines_and_leads_with_the_best():
    frontier = ExhaustiveSynthesizer(_registry()).synthesize(_task())
    evaluations = evaluate_pipelines(frontier, _executor(), _eval_set())
    kept = quality_frontier(evaluations)

    assert kept[0].components == BEST
    qualities = [evaluation.quality for evaluation in kept]
    assert qualities == sorted(qualities, reverse=True)
    for evaluation in evaluations:
        if evaluation not in kept:
            assert any(
                other.cost <= evaluation.cost
                and other.latency_ms <= evaluation.latency_ms
                and other.quality >= evaluation.quality
                for other in kept
            )


def test_closed_loop_with_an_eval_set_finds_the_known_best_pipeline():
    # Declared at 0.99, flaky is cheaper, faster and "more reliable" than
    # double, so it hides the best pipeline from the first frontier.
    registry = _registry(flaky_declared=0.99)
    records = run_closed_loop(
        registry=registry,
        task=_task(),
        synthesizer=ExhaustiveSynthesizer,
        executor=_executor(),
        iterations=3,
        prior_strength=10,
        eval_set=_eval_set(),
    )
    first, final = records[0], records[-1]
    assert BEST not in _by_components(first.evaluations)

    # Running the examples shows flaky failing half the time; calibration
    # lowers its reliability and the best pipeline enters the frontier.
    assert final.registry.get("flaky").reliability_score < 0.95
    best = max(final.evaluations, key=lambda evaluation: evaluation.quality)
    assert best.components == BEST
    assert best.quality == 1.0
    # Each pipeline is run once; later rounds reuse its evaluation.
    distinct = {e.components: e for r in records for e in r.evaluations}
    assert sum(r.trace_count for r in records) == sum(
        len(e.traces) for e in distinct.values()
    )


def test_loop_record_with_evaluations_is_json_serializable():
    records = run_closed_loop(
        registry=_registry(),
        task=_task(),
        synthesizer=ExhaustiveSynthesizer,
        executor=_executor(),
        iterations=1,
        eval_set=_eval_set(),
    )
    payload = json.loads(json.dumps(records[0].to_dict()))
    assert payload["evaluations"]
    assert set(payload["evaluations"][0]) == {
        "components", "runs", "quality", "success_rate", "cost", "latency_ms",
    }


def test_loop_without_an_eval_set_has_no_evaluations():
    records = run_closed_loop(
        registry=_registry(),
        task=_task(),
        synthesizer=ExhaustiveSynthesizer,
        executor=AgentExecutor(),
        iterations=1,
    )
    assert records[0].evaluations == []


def test_scoring_with_the_simulator_is_rejected():
    frontier = ExhaustiveSynthesizer(_registry()).synthesize(_task())
    with pytest.raises(ValueError, match="backend"):
        evaluate_pipelines(frontier, AgentExecutor(), _eval_set())


def test_eval_set_validation():
    with pytest.raises(ValueError, match="at least one example"):
        EvalSet(examples=[], scorer=lambda output, expected: 1.0)
    with pytest.raises(TypeError, match="callable"):
        EvalSet(examples=[Example(1)], scorer="not a function")


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), "high", None])
def test_scorer_must_return_a_score_between_zero_and_one(bad):
    eval_set = EvalSet(examples=[Example(1)], scorer=lambda output, expected: bad)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        eval_set.score("anything", None)


def test_boolean_scores_count_as_one_and_zero():
    eval_set = EvalSet(examples=[Example(1)], scorer=lambda output, expected: output)
    assert eval_set.score(True, None) == 1.0
    assert eval_set.score(False, None) == 0.0
