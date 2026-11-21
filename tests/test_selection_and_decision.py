"""Best-of-many correction and the saved decision record."""

import json
import random
from datetime import datetime, timezone

import pytest

from frontierkit import (
    AgentExecutor,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    EvalSet,
    Example,
    TaskSpecification,
    best_of,
    enumerate_pipelines,
    evaluate_pipelines,
    load_decision,
    record_decision,
)


def coin_scores(rng, rate, n):
    return [1.0 if rng.random() < rate else 0.0 for _ in range(n)]


def test_correction_removes_the_luck_of_the_pick():
    """Twenty equal pipelines: the winner looks good, the correction does not."""
    rng = random.Random(7)
    measured, corrected, held = [], [], 0
    runs = 120
    for run in range(runs):
        scores = {name: coin_scores(rng, 0.5, 30) for name in range(20)}
        result = best_of(scores, resamples=200, seed=run)
        measured.append(result.measured)
        corrected.append(result.corrected)
        held += result.interval.contains(0.5)

    assert sum(measured) / runs > 0.62  # flattered by being picked
    assert abs(sum(corrected) / runs - 0.5) < 0.03
    assert held / runs >= 0.95


def test_a_clear_winner_keeps_its_score_and_has_few_contenders():
    rng = random.Random(3)
    scores = {"strong": coin_scores(rng, 0.95, 80)}
    scores.update({f"weak{i}": coin_scores(rng, 0.3, 80) for i in range(6)})

    result = best_of(scores, resamples=300)

    assert result.winner == "strong"
    assert result.clear and result.contenders == ()
    assert result.measured - result.corrected < 0.03
    assert result.optimism == pytest.approx(result.measured - result.corrected)


def test_contenders_are_the_pipelines_too_close_to_call():
    rng = random.Random(5)
    base = coin_scores(rng, 0.8, 40)
    twin = list(base)
    twin[0] = 1.0 - twin[0]
    scores = {"a": base, "b": twin, "bad": [0.0] * 40}

    result = best_of(scores, resamples=200)

    assert result.winner in ("a", "b")
    assert set(result.contenders) == {"a", "b"} - {result.winner}


def test_one_pipeline_needs_no_correction():
    result = best_of({"only": [1.0, 0.0, 1.0, 1.0]})
    assert result.corrected == result.measured == 0.75
    assert result.contenders == ()


def test_best_of_is_repeatable_and_checks_its_input():
    rng = random.Random(1)
    scores = {i: coin_scores(rng, 0.5, 20) for i in range(5)}
    assert best_of(scores, seed=4).to_dict() == best_of(scores, seed=4).to_dict()

    with pytest.raises(ValueError, match="same examples"):
        best_of({"a": [1.0, 0.0], "b": [1.0]})
    with pytest.raises(ValueError, match="at least one pipeline"):
        best_of({})


# --- decision record -------------------------------------------------------


def component(name, capability):
    return ComponentMetadata(
        name=name,
        component_type="tool",
        capabilities=[capability],
        cost_per_call=0.01,
        avg_latency_ms=10,
        reliability_score=0.9,
        privacy_level="internal",
    )


def compared():
    registry = CapabilityRegistry()
    for name, capability in (("double", "compute"), ("add_one", "compute"), ("show", "format")):
        registry.register(component(name, capability))
    task = TaskSpecification(required_capabilities=["compute", "format"])
    backend = CallableBackend(
        {"double": lambda n: n * 2, "add_one": lambda n: n + 1, "show": lambda n: str(n)}
    )
    eval_set = EvalSet(
        examples=[Example(n, expected=str(n * 2)) for n in range(2, 22)],
        scorer=lambda output, expected: output == expected,
    )
    evaluations = evaluate_pipelines(
        enumerate_pipelines(registry, task), AgentExecutor(backend=backend), eval_set
    )
    return evaluations, eval_set, backend


FIXED = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def test_decision_names_the_choice_and_the_evidence(tmp_path):
    evaluations, eval_set, backend = compared()

    decision = record_decision(evaluations, eval_set, backend=backend, clock=lambda: FIXED)

    assert decision.chosen["components"] == ["double", "show"]
    assert decision.chosen["quality"] == 1.0
    assert decision.chosen["interval"]["low"] > 0.5
    (other,) = decision.compared
    assert other["components"] == ["add_one", "show"]
    assert other["against_chosen"]["verdict"] == "chosen is better"
    assert decision.contenders == []
    assert decision.shared == [{"component": "show"}]
    assert decision.data["eval_set"]["examples"] == 20
    assert len(decision.data["eval_set"]["hash"]) == 32
    assert decision.data["selection"]["candidates"] == 2
    assert decision.data["selection"]["chosen_is_best"] is True
    assert decision.data["created"] == "2026-11-21T12:00:00+00:00"

    path = decision.save(tmp_path / "out" / "decision.json")
    assert json.loads(path.read_text(encoding="utf-8")) == decision.to_dict()
    assert load_decision(path).chosen == decision.chosen


def test_decision_records_a_choice_that_is_not_the_best():
    evaluations, eval_set, backend = compared()
    worse = next(e for e in evaluations if e.components[0] == "add_one")

    decision = record_decision(
        evaluations, eval_set, backend=backend, chosen=worse, note="cheaper"
    )

    assert decision.chosen["components"] == ["add_one", "show"]
    assert decision.data["selection"]["chosen_is_best"] is False
    assert decision.data["selection"]["best"] == ["double", "show"]
    assert decision.compared[0]["against_chosen"]["verdict"] == "chosen is worse"
    assert decision.data["note"] == "cheaper"


def test_same_examples_give_the_same_hash_and_different_ones_do_not():
    evaluations, eval_set, backend = compared()
    again = EvalSet(examples=list(eval_set.examples), scorer=eval_set.scorer)
    shorter = EvalSet(examples=list(eval_set.examples)[:-1], scorer=eval_set.scorer)

    first = record_decision(evaluations, eval_set).data["eval_set"]["hash"]
    assert record_decision(evaluations, again).data["eval_set"]["hash"] == first
    with pytest.raises(ValueError, match="scores for 19 examples"):
        record_decision(evaluations, shorter)


def test_decision_rejects_bad_input(tmp_path):
    evaluations, eval_set, _ = compared()
    with pytest.raises(ValueError, match="at least one evaluation"):
        record_decision([], eval_set)
    stranger = compared()[0][0]
    with pytest.raises(ValueError, match="one of the evaluations"):
        record_decision(evaluations, eval_set, chosen=stranger)
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a decision record"):
        load_decision(path)
