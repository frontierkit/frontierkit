"""Whole-set metrics: figures that are not a mean of per-example scores."""

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
    enumerate_pipelines,
    evaluate_pipelines,
    label_macro_f1,
    pooled_triple_f1,
    pooled_triple_precision,
    pooled_triple_recall,
    run_closed_loop,
    triple_f1,
    triple_precision,
)

# Two documents of very different size: the first has one reference
# triple and it is found; the second has four and one is found, with one
# extra triple invented.
EXPECTEDS = [
    [("a", "must", "b")],
    [("c", "must", "d"), ("e", "should", "f"), ("g", "may", "h"), ("i", "must", "j")],
]
OUTPUTS = [
    [("a", "must", "b")],
    [("c", "must", "d"), ("x", "must", "y")],
]


def test_pooled_scores_count_triples_not_examples():
    # 2 of 3 extracted triples are right; 2 of 5 reference triples are found.
    assert pooled_triple_precision(OUTPUTS, EXPECTEDS) == pytest.approx(2 / 3)
    assert pooled_triple_recall(OUTPUTS, EXPECTEDS) == pytest.approx(2 / 5)
    assert pooled_triple_f1(OUTPUTS, EXPECTEDS) == pytest.approx(2 * 2 / (3 + 5))

    # The mean of per-example precision weighs both documents the same.
    per_example = [triple_precision(o, e) for o, e in zip(OUTPUTS, EXPECTEDS)]
    assert sum(per_example) / 2 == pytest.approx((1.0 + 0.5) / 2)


def test_a_triple_only_matches_the_reference_of_its_own_example():
    swapped = [EXPECTEDS[1], EXPECTEDS[0]]
    assert pooled_triple_precision(swapped, EXPECTEDS) == 0.0


def test_pooled_scores_treat_failed_runs_and_empty_sets_like_the_scorers():
    assert pooled_triple_recall([None, OUTPUTS[1]], EXPECTEDS) == pytest.approx(1 / 5)
    for metric in (pooled_triple_precision, pooled_triple_recall, pooled_triple_f1):
        assert metric([[], None], [[], []]) == 1.0
        assert metric([None, None], EXPECTEDS) == 0.0
        assert metric(OUTPUTS, [[], []]) == 0.0
    with pytest.raises(ValueError, match="same length"):
        pooled_triple_f1(OUTPUTS, EXPECTEDS[:1])


# Triples with a modality as the last field.
LABELLED_EXPECTED = [[
    ("p", "document", "data", "mandatory"),
    ("p", "monitor", "system", "mandatory"),
    ("u", "request", "explanation", "optional"),
    ("d", "inform", "persons", "recommended"),
]]
LABELLED_OUTPUT = [[
    ("p", "document", "data", "mandatory"),       # right triple, right label
    ("p", "monitor", "system", "recommended"),    # right triple, wrong label
    ("u", "request", "explanation", "optional"),  # right triple, right label
    ("z", "invent", "something", "optional"),     # not in the reference
]]


def test_label_macro_f1_covers_extraction_and_labelling_by_default():
    # mandatory: 1 right of 1 predicted and 2 wanted -> 2/3
    # optional:  1 right of 2 predicted and 1 wanted -> 2/3
    # recommended: 0 right of 1 predicted and 1 wanted -> 0
    assert label_macro_f1(LABELLED_OUTPUT, LABELLED_EXPECTED) == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)


def test_label_macro_f1_matched_only_scores_the_labels_alone():
    # Only the three triples found in both count. "d inform persons" was
    # missed and "z invent something" was invented, so both drop out.
    # mandatory: 1 right of 1 predicted and 2 wanted -> 2/3
    # optional:  1 right of 1 predicted and 1 wanted -> 1
    # recommended: 0 right of 1 predicted and 0 wanted -> 0
    score = label_macro_f1(LABELLED_OUTPUT, LABELLED_EXPECTED, matched_only=True)
    assert score == pytest.approx((2 / 3 + 1 + 0) / 3)


def test_label_macro_f1_takes_the_label_from_any_field():
    outputs = [[("mandatory", "p", "document", "data")]]
    expecteds = [[("Mandatory", "p", "document", "the data")]]
    assert label_macro_f1(outputs, expecteds, field=0) == 1.0
    assert label_macro_f1(outputs, [[("optional", "p", "document", "data")]], field=0) == 0.0


def test_label_macro_f1_edge_cases():
    assert label_macro_f1([[], None], [[], []]) == 1.0
    assert label_macro_f1([None], LABELLED_EXPECTED) == 0.0
    # Nothing was found in both, so there is no labelling to score.
    nothing_shared = [[("q", "r", "s", "mandatory")]]
    assert label_macro_f1(nothing_shared, LABELLED_EXPECTED, matched_only=True) == 0.0
    with pytest.raises(ValueError, match="no label at position 3"):
        label_macro_f1([[("a", "b", "c")]], LABELLED_EXPECTED, field=3)


# --- metrics on an eval set ----------------------------------------------

DOCUMENTS = {"short": EXPECTEDS[0], "long": EXPECTEDS[1]}


def _registry():
    registry = CapabilityRegistry()
    for name in ("careful", "sloppy", "broken"):
        registry.register(
            ComponentMetadata(
                name=name, component_type="tool", capabilities=["extract"],
                cost_per_call=0.0, avg_latency_ms=1.0, reliability_score=1.0,
                privacy_level="internal",
            )
        )
    return registry


def _executor():
    return AgentExecutor(
        backend=CallableBackend({
            "careful": lambda name: DOCUMENTS[name],
            "sloppy": lambda name: OUTPUTS[0] if name == "short" else OUTPUTS[1],
            "broken": lambda name: StepResult(None, success=False),
        })
    )


def _eval_set(**kwargs):
    return EvalSet(
        examples=[Example(name, expected=triples) for name, triples in DOCUMENTS.items()],
        scorer=triple_f1,
        **kwargs,
    )


TASK = TaskSpecification(required_capabilities=["extract"])


def test_metrics_are_computed_once_per_pipeline_over_all_examples():
    eval_set = _eval_set(metrics={
        "precision": pooled_triple_precision,
        "recall": pooled_triple_recall,
        "count": lambda outputs, expecteds: sum(len(o or []) for o in outputs),
    })
    scored = {
        e.components[0]: e
        for e in evaluate_pipelines(enumerate_pipelines(_registry(), TASK), _executor(), eval_set)
    }
    assert scored["careful"].metrics == {"precision": 1.0, "recall": 1.0, "count": 5.0}
    assert scored["sloppy"].metrics["precision"] == pytest.approx(2 / 3)
    assert scored["sloppy"].metrics["recall"] == pytest.approx(2 / 5)
    # A failed run extracts nothing.
    assert scored["broken"].metrics == {"precision": 0.0, "recall": 0.0, "count": 0.0}

    # Quality is still the mean of the per-example scorer.
    assert scored["sloppy"].quality == pytest.approx((1.0 + 1 / 3) / 2)
    assert json.loads(json.dumps(scored["sloppy"].to_dict()))["metrics"]["count"] == 3.0


def test_without_metrics_evaluations_are_unchanged():
    evaluation = evaluate_pipelines(
        enumerate_pipelines(_registry(), TASK), _executor(), _eval_set()
    )[0]
    assert evaluation.metrics == {}
    assert "metrics" not in evaluation.to_dict()


def test_closed_loop_reports_metrics():
    records = run_closed_loop(
        registry=_registry(), task=TASK, synthesizer=lambda reg: ExhaustiveSynthesizer(reg, keep_ties=True),
        executor=_executor(), iterations=1,
        eval_set=_eval_set(metrics={"f1": pooled_triple_f1}),
    )
    assert records[0].evaluations
    assert all("f1" in evaluation.metrics for evaluation in records[0].evaluations)
    assert "f1" in records[0].to_dict()["evaluations"][0]["metrics"]


def test_metric_validation():
    with pytest.raises(TypeError, match="callable"):
        _eval_set(metrics={"f1": "pooled"})
    with pytest.raises(TypeError, match="strings"):
        _eval_set(metrics={1: pooled_triple_f1})
    with pytest.raises(TypeError, match="map a name"):
        _eval_set(metrics=[pooled_triple_f1])

    for bad in (float("nan"), "high", None, True):
        eval_set = _eval_set(metrics={"bad": lambda outputs, expecteds, bad=bad: bad})
        with pytest.raises(ValueError, match="finite number"):
            eval_set.measure([None, None])
    with pytest.raises(ValueError, match="one output per example"):
        _eval_set(metrics={"f1": pooled_triple_f1}).measure([None])
