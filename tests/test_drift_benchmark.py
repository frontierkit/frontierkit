"""The daily collector and its report, with stand-in models."""

from benchmarks.drift.collect import collect_sample, prompts_for
from benchmarks.drift.report import changes, choices, failures_by_sample, scores_by_sample
from benchmarks.qa_frontier.qa import SAMPLE_DATA, SQUAD_SUBSET, ModelSpec, load_questions
from chatcortex.execution.executor import ProbeResult
from chatcortex.store import AnswerLog

QUESTIONS = load_questions(SAMPLE_DATA)
MODELS = [
    ModelSpec("good", "alpha", "ALPHA_KEY", "https://alpha.test/v1", "good-model", 100),
    ModelSpec("poor", "beta", "BETA_KEY", "https://beta.test/v1", "poor-model", 100),
]
ENV = {"ALPHA_KEY": "x", "BETA_KEY": "y"}


class RateLimited(Exception):
    status_code = 429


def answering(replies, calls):
    """Backend factory: ``replies[spec.name](prompt)`` gives the reply or raises."""

    def make(spec, api_key):
        def backend(metadata, prompt):
            calls.append(spec.name)
            return ProbeResult(
                success=True, latency_ms=5.0, cost=0.0, output=replies[spec.name](prompt)
            )

        return backend

    return make


def right_answer(prompt, questions=QUESTIONS):
    for question in questions:
        if question.question in prompt:
            return question.answers[0]
    return "?"


def collect(log, label, replies, calls=None, models=MODELS, env=ENV, questions=QUESTIONS):
    return collect_sample(
        questions,
        questions,
        log,
        label,
        models=models,
        env=env,
        make_backend=answering(replies, [] if calls is None else calls),
        sleep=lambda seconds: None,
        out=lambda line: None,
    )


def test_a_sample_is_collected_once_and_resumed_for_free(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    replies = {"good": right_answer, "poor": lambda prompt: "no idea"}
    needed = len(prompts_for(QUESTIONS, QUESTIONS))

    calls = []
    counts = collect(log, "day1", replies, calls)
    assert counts["good"] == {"answered": needed, "failed": 0, "needed": needed}
    assert len(calls) == 2 * needed

    again = []
    collect(log, "day1", replies, again)
    assert again == []

    next_day = []
    collect(log, "day2", replies, next_day)
    assert len(next_day) == 2 * needed


def test_models_without_a_key_are_skipped(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    calls = []
    counts = collect(log, "day1", {"good": right_answer}, calls, env={"ALPHA_KEY": "x"})

    assert set(calls) == {"good"}
    assert counts["poor"]["answered"] == 0


def test_a_rate_limit_stops_that_model_and_is_logged(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")

    def limited(prompt):
        raise RateLimited("slow down")

    calls = []
    counts = collect(log, "day1", {"good": right_answer, "poor": limited}, calls)

    assert calls.count("poor") == 1
    assert counts["poor"]["failed"] == 1
    assert failures_by_sample(log.records()) == {"day1": {"poor": {"rate_limited": 1}}}


def test_a_broken_model_is_left_alone_after_three_failures(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")

    def broken(prompt):
        raise ValueError("bad reply")

    calls = []
    collect(log, "day1", {"good": right_answer, "poor": broken}, calls)

    assert calls.count("poor") == 3


def test_report_finds_the_day_the_choice_stopped_holding(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")
    thirty = load_questions(SQUAD_SUBSET)

    def right(prompt):
        return right_answer(prompt, thirty)

    def wrong(prompt):
        return "no idea"

    collect(log, "day1", {"good": right, "poor": wrong}, questions=thirty)
    collect(log, "day2", {"good": right, "poor": wrong}, questions=thirty)
    collect(log, "day3", {"good": wrong, "poor": right}, questions=thirty)  # the two swap

    samples = scores_by_sample(log.records(), thirty, thirty)
    picked = choices(samples)

    assert list(picked) == ["day1", "day2", "day3"]
    assert picked["day1"].winner[0] == "good"
    assert picked["day3"].winner[0] == "poor"
    moved = changes(picked)
    assert [(label, before[0], after[0]) for label, before, after in moved] == [
        ("day3", "good", "poor")
    ]


def test_six_questions_are_too_few_to_call_a_change(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")

    def wrong(prompt):
        return "no idea"

    collect(log, "day1", {"good": right_answer, "poor": wrong})
    collect(log, "day2", {"good": wrong, "poor": right_answer})

    picked = choices(scores_by_sample(log.records(), QUESTIONS, QUESTIONS))

    assert picked["day1"].winner[0] == "good" and picked["day2"].winner[0] == "poor"
    assert changes(picked) == []


def test_report_ignores_setups_that_were_not_finished(tmp_path):
    log = AnswerLog(tmp_path / "answers.jsonl")

    def limited(prompt):
        raise RateLimited("slow down")

    collect(log, "day1", {"good": right_answer, "poor": limited})
    samples = scores_by_sample(log.records(), QUESTIONS, QUESTIONS)

    assert {setup[0] for setup in samples["day1"]} == {"good"}
