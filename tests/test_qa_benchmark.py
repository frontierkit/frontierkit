"""QA frontier benchmark: scoring, retrieval, collection and the grid report.

Runs on the bundled sample with stub model callers: no keys, no network.
"""

import pytest

from benchmarks.qa_frontier.collect import collect
from benchmarks.qa_frontier.qa import (
    CONTEXT_VARIANTS,
    SAMPLE_DATA,
    SQUAD_SUBSET,
    ModelSpec,
    ResponseCache,
    Retriever,
    first_clause,
    load_questions,
    needed_calls,
    score,
)
from benchmarks.qa_frontier.replay import (
    Point,
    Replay,
    at_budget,
    chatcortex_strategy,
    complete_models,
    grid_strategy,
    random_strategy,
)
from benchmarks.qa_frontier.report import grid_report

MODELS = [
    ModelSpec("fast", "p1", "KEY_P1", "https://p1.test/v1", "fast-model", 100),
    ModelSpec("good", "p2", "KEY_P2", "https://p2.test/v1", "good-model", 900),
]
KEYS = {"KEY_P1": "k1", "KEY_P2": "k2"}


def _questions():
    return load_questions(SAMPLE_DATA)


def _answer_from_prompt(prompt):
    """A stub 'good' model: answers right only when the context holds the fact."""
    for question in _questions():
        if question.question in prompt and question.context in prompt:
            return question.answers[0] + ". It is stated in the context."
    return "I am not sure."


def _callers(calls, rate_limited=()):
    def make(spec, api_key):
        def call(prompt):
            calls.append(spec.name)
            if spec.provider in rate_limited:
                return False, "openai call failed: RateLimitError: Error code: 429", 5.0
            if spec.name == "fast":
                return True, "unknown", 50.0
            return True, _answer_from_prompt(prompt), 800.0

        return call

    return make


def _collect(tmp_path, **kwargs):
    calls = []
    cache = ResponseCache(tmp_path / "cache.jsonl")
    summary = collect(
        _questions(),
        cache,
        models=MODELS,
        env=kwargs.pop("env", KEYS),
        make_caller=_callers(calls, kwargs.pop("rate_limited", ())),
        sleep=lambda seconds: None,
        log=lambda line: None,
        **kwargs,
    )
    return cache, calls, summary


def test_score_uses_squad_normalisation_and_best_answer():
    assert score("The local granite.", ("local granite",)) == 1.0
    assert score("granite", ("local granite",)) == pytest.approx(2 / 3)
    assert score("basalt", ("local granite", "granite")) == 0.0
    assert score("Granite", ("local granite", "granite")) == 1.0


def test_first_clause_keeps_the_answer_and_drops_the_explanation():
    assert first_clause("Carbon dioxide. The yeast makes it.") == "Carbon dioxide"
    assert first_clause("\n") == ""


def test_retriever_ranks_the_matching_paragraph_first():
    questions = _questions()
    retriever = Retriever(q.context for q in questions)
    for question in questions:
        assert retriever.top(question.question, 1) == [question.context]


def test_sampling_is_deterministic():
    first = load_questions(SAMPLE_DATA, limit=3, seed=7)
    again = load_questions(SAMPLE_DATA, limit=3, seed=7)
    assert first == again and len(first) == 3


def test_grid_needs_one_call_per_model_per_distinct_prompt():
    calls = needed_calls(_questions(), MODELS)
    assert len(calls) == len(_questions()) * len(CONTEXT_VARIANTS) * len(MODELS)


def test_collect_respects_the_budget_and_resumes_without_repeating(tmp_path):
    cache, calls, _ = _collect(tmp_path, budget=5)
    assert len(calls) == 5 and len(cache) == 5

    resumed = ResponseCache(tmp_path / "cache.jsonl")
    assert len(resumed) == 5
    more_calls = []
    summary = collect(
        _questions(), resumed, models=MODELS, env=KEYS,
        make_caller=_callers(more_calls), sleep=lambda s: None, log=lambda line: None,
    )
    total = len(needed_calls(_questions(), MODELS))
    assert len(more_calls) == total - 5
    assert all(c["cached"] == c["needed"] for c in summary.values())


def test_rate_limited_provider_is_paused_and_others_continue(tmp_path):
    _, calls, summary = _collect(tmp_path, rate_limited=("p2",))
    assert calls.count("good") == 1
    assert summary["fast"]["cached"] == summary["fast"]["needed"]
    assert summary["good"]["cached"] == 0


def test_models_without_a_key_are_skipped(tmp_path):
    _, calls, summary = _collect(tmp_path, env={"KEY_P1": "k1"})
    assert set(calls) == {"fast"}
    assert summary["good"]["cached"] == 0


def test_provider_spacing_is_enforced(tmp_path):
    waits = []
    collect(
        _questions(), ResponseCache(tmp_path / "c.jsonl"), models=MODELS[:1], env=KEYS,
        make_caller=_callers([]), sleep=waits.append, clock=lambda: 0.0,
        log=lambda line: None, budget=3,
    )
    # p1 has no configured spacing, so nothing waits; a known provider does.
    assert waits == []
    spaced = [ModelSpec("g", "groq", "KEY_P1", "u", "m", 1)]
    collect(
        _questions(), ResponseCache(tmp_path / "d.jsonl"), models=spaced, env=KEYS,
        make_caller=_callers([]), sleep=waits.append, clock=lambda: 0.0,
        log=lambda line: None, budget=3,
    )
    assert len(waits) == 2 and all(w > 0 for w in waits)


def test_grid_report_finds_the_known_best_pipeline(tmp_path):
    cache, _, _ = _collect(tmp_path)
    evaluations, incomplete = grid_report(_questions(), cache, models=MODELS)
    assert incomplete == []
    assert len(evaluations) == len(CONTEXT_VARIANTS) * len(MODELS) * 2

    best = max(evaluations, key=lambda e: e.quality)
    assert best.components == ("ctx_top1", "good", "post_first_clause")
    assert best.quality == 1.0
    # Keeping the explanation lowers the F1 score.
    as_is = next(
        e for e in evaluations
        if e.components == ("ctx_top1", "good", "post_as_is")
    )
    assert as_is.quality < best.quality


def test_pipelines_differing_only_in_a_local_step_tie_exactly(tmp_path):
    cache, _, _ = _collect(tmp_path)
    evaluations, _ = grid_report(_questions(), cache, models=MODELS)
    by_name = {e.components: e for e in evaluations}
    a = by_name[("ctx_top1", "fast", "post_as_is")]
    b = by_name[("ctx_top1", "fast", "post_first_clause")]
    assert a.latency_ms == b.latency_ms == 50.0


def test_grid_report_lists_pipelines_still_missing_replies(tmp_path):
    cache, _, _ = _collect(tmp_path, env={"KEY_P1": "k1"})
    evaluations, incomplete = grid_report(_questions(), cache, models=MODELS)
    assert {e.components[1] for e in evaluations} == {"fast"}
    assert incomplete and all("good" in name for name in incomplete)


def test_squad_subset_is_intact():
    questions = load_questions(SQUAD_SUBSET)
    assert len(questions) == 30
    assert len({q.id for q in questions}) == 30
    for question in questions:
        assert any(answer in question.context for answer in question.answers)


def _replay(tmp_path):
    cache, _, _ = _collect(tmp_path)
    return Replay(_questions(), cache, MODELS)


def test_replay_counts_one_call_per_model_and_distinct_prompt(tmp_path):
    replay = _replay(tmp_path)
    assert replay.total_calls == len(needed_calls(_questions(), MODELS))
    one = ("ctx_top1", "good", "post_as_is")
    same_prompts = ("ctx_top1", "good", "post_first_clause")
    assert replay.calls_for([one]) == len(_questions())
    assert replay.calls_for([one, same_prompts]) == len(_questions())


def test_grid_and_random_find_the_whole_frontier_by_the_full_budget(tmp_path):
    replay = _replay(tmp_path)
    total = replay.total_calls
    assert at_budget(grid_strategy(replay), total).coverage == 1.0
    for seed in range(3):
        points = random_strategy(replay, seed)
        assert at_budget(points, total).coverage == 1.0
        assert [p.calls for p in points] == sorted(p.calls for p in points)


def test_chatcortex_strategy_reports_a_point_per_round_within_the_grid(tmp_path):
    replay = _replay(tmp_path)
    points = chatcortex_strategy(replay, iterations=3)
    assert len(points) == 3
    assert all(0 < p.calls <= replay.total_calls for p in points)
    assert [p.calls for p in points] == sorted(p.calls for p in points)


def test_at_budget_is_zero_before_anything_is_affordable():
    points = [Point(10, 0.5, 0.9), Point(20, 1.0, 1.0)]
    assert at_budget(points, 5) == Point(0, 0.0, 0.0)
    assert at_budget(points, 15) == Point(10, 0.5, 0.9)
    assert at_budget(points, 25) == Point(20, 1.0, 1.0)


def test_complete_models_excludes_partly_collected_ones(tmp_path):
    cache, _, _ = _collect(tmp_path, env={"KEY_P1": "k1"})
    assert [m.name for m in complete_models(_questions(), cache, MODELS)] == ["fast"]
