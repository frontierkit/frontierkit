"""Answer scorers, pipeline enumeration, and the model-picking example's helpers."""

import sys
from pathlib import Path

import pytest

from chatcortex import (
    CapabilityRegistry,
    ComponentMetadata,
    TaskSpecification,
    contains_answer,
    count_pipelines,
    enumerate_pipelines,
    exact_match,
    token_f1,
    triple_f1,
    triple_precision,
    triple_recall,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import pick_models  # noqa: E402


def test_scorers_normalise_case_punctuation_and_articles():
    assert exact_match("The Pacific.", "pacific") == 1.0
    assert exact_match("Pacific Ocean", "Pacific") == 0.0
    assert contains_answer("It is the Pacific Ocean.", ["Atlantic", "Pacific"]) == 1.0
    assert contains_answer("No idea", "Pacific") == 0.0
    assert token_f1("local granite", "granite") == pytest.approx(2 / 3)
    assert token_f1("granite", ["basalt", "Granite"]) == 1.0


def test_scorers_reject_an_empty_reference_list():
    with pytest.raises(ValueError, match="reference"):
        token_f1("x", [])


REFERENCE_TRIPLES = [
    ("provider", "must document", "training data"),
    ("provider", "should monitor", "the system"),
    ("user", "may request", "an explanation"),
    ("deployer", "must inform", "affected persons"),
]


def test_triple_scorers_count_exact_matches_after_normalisation():
    extracted = [
        ("Provider", "must document", "the training data."),  # matches
        ["user", "may request", "explanation"],                # matches, as a list
        ("provider", "must monitor", "the system"),            # wrong predicate
    ]
    assert triple_precision(extracted, REFERENCE_TRIPLES) == pytest.approx(2 / 3)
    assert triple_recall(extracted, REFERENCE_TRIPLES) == pytest.approx(2 / 4)
    assert triple_f1(extracted, REFERENCE_TRIPLES) == pytest.approx(4 / 7)


def test_triple_scorers_count_duplicates_once_and_compare_every_field():
    repeated = [("user", "may request", "an explanation")] * 3
    assert triple_precision(repeated, REFERENCE_TRIPLES) == 1.0
    assert triple_recall(repeated, REFERENCE_TRIPLES) == pytest.approx(1 / 4)

    # A fourth field makes the match stricter: same triple, other modality.
    assert triple_f1([("a", "does", "b", "mandatory")], [("a", "does", "b", "optional")]) == 0.0
    assert triple_f1([("a", "does", "b", "mandatory")], [("a", "does", "b", "Mandatory")]) == 1.0


def test_triple_scorers_on_empty_collections():
    for scorer in (triple_precision, triple_recall, triple_f1):
        assert scorer([], []) == 1.0
        assert scorer(None, []) == 1.0
        assert scorer([], REFERENCE_TRIPLES) == 0.0
        assert scorer(REFERENCE_TRIPLES, []) == 0.0
        assert scorer([("x", "y", "z")], REFERENCE_TRIPLES) == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        "provider must document training data",
        ["provider must document training data"],
        [{"subject": "provider", "predicate": "must document", "object": "training data"}],
        [42],
    ],
)
def test_triple_scorers_reject_unparsed_output(bad):
    with pytest.raises(TypeError, match="triple"):
        triple_f1(bad, REFERENCE_TRIPLES)


def _component(name, capability, cost=0.0, latency=1.0):
    return ComponentMetadata(
        name=name, component_type="tool", capabilities=[capability],
        cost_per_call=cost, avg_latency_ms=latency, reliability_score=1.0,
        privacy_level="internal",
    )


def _registry():
    registry = CapabilityRegistry()
    for name in ("a1", "a2", "a3"):
        registry.register(_component(name, "a"))
    for name, latency in (("b1", 1.0), ("b2", 100.0)):
        registry.register(_component(name, "b", latency=latency))
    return registry


def test_enumerate_lists_every_combination_in_order():
    task = TaskSpecification(required_capabilities=["a", "b"])
    pipelines = enumerate_pipelines(_registry(), task)
    names = [[m.name for _, m in c.graph.iter_components()] for c in pipelines]
    assert names[:2] == [["a1", "b1"], ["a1", "b2"]]
    assert len(names) == count_pipelines(_registry(), task) == 6


def test_enumerate_respects_limits_and_samples_reproducibly():
    capped = TaskSpecification(required_capabilities=["a", "b"], max_latency=50)
    assert len(enumerate_pipelines(_registry(), capped)) == 3

    task = TaskSpecification(required_capabilities=["a", "b"])
    sample = enumerate_pipelines(_registry(), task, limit=4, seed=1)
    again = enumerate_pipelines(_registry(), task, limit=4, seed=1)
    assert len(sample) == 4
    assert [c.graph.list_nodes() for c in sample] == [c.graph.list_nodes() for c in again]


def test_example_reads_questions_with_alternative_answers(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text(
        'question,answer,context\nWhat gas?,carbon dioxide|CO2,"Plants take in carbon dioxide."\n'
        "Second?,two,\n",
        encoding="utf-8",
    )
    questions = pick_models.load_questions(path)
    assert questions[0]["answers"] == ["carbon dioxide", "CO2"]
    assert questions[1]["context"] == ""
    assert len(pick_models.load_questions(path, limit=1)) == 1


def test_example_prompt_steps_add_context_only_when_asked():
    questions = [{"question": "Q?", "answers": ["a"], "context": "Some text."}]
    steps = pick_models.prompt_steps(questions)
    assert set(steps) == {
        "plain+no_context", "plain+with_context", "short+no_context", "short+with_context",
    }
    assert steps["plain+no_context"]("Q?") == "Q?"
    assert steps["plain+with_context"]("Q?").startswith("Context:\nSome text.")


def test_example_skips_context_options_when_no_question_has_context():
    steps = pick_models.prompt_steps([{"question": "Q?", "answers": ["a"], "context": ""}])
    assert set(steps) == {"plain+no_context", "short+no_context"}


def test_example_spaces_calls_that_share_a_provider():
    waits, now = [], [0.0]
    pick_models.SpacedBackend.last_call = {}
    make = lambda: pick_models.SpacedBackend(  # noqa: E731
        lambda metadata, prompt: prompt, provider="KEY", interval_s=5.0,
        sleep=waits.append, clock=lambda: now[0],
    )
    first, second = make(), make()
    first(None, "x")
    now[0] = 2.0
    second(None, "y")
    assert waits == [3.0]
