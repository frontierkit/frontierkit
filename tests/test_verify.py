"""verify_decision and the frontierkit verify command."""

import json
from datetime import datetime, timezone

import pytest

from frontierkit import (
    AgentExecutor,
    AnswerLog,
    CapabilityRegistry,
    ComponentMetadata,
    Decision,
    EvalSet,
    Example,
    LoggedBackend,
    TaskSpecification,
    enumerate_pipelines,
    evaluate_pipelines,
    load_decision,
    record_decision,
    verify_decision,
)
from frontierkit.cli import history_path, load_setup, main
from frontierkit.execution.executor import ProbeResult

FIXED = datetime(2026, 11, 21, 8, 0, tzinfo=timezone.utc)


class NotFound(Exception):
    status_code = 404


class Model:
    """A stand-in model whose behaviour, speed and name can be changed."""

    def __init__(self, name="model-a"):
        self.name = name
        self.wrong = set()       # example inputs it now gets wrong
        self.latency_ms = 200.0
        self.stalled = set()     # inputs that take very long
        self.retired = False
        self.calls = 0

    def identity(self, metadata):
        return {"endpoint": "https://example.test/v1", "model": self.name}

    def __call__(self, metadata, node_input):
        self.calls += 1
        if self.retired:
            raise NotFound("no such model")
        latency = 60000.0 if node_input in self.stalled else self.latency_ms
        output = "no" if node_input in self.wrong else "yes"
        return ProbeResult(success=True, latency_ms=latency, cost=0.0, output=output)


def world(n=40, model=None):
    model = model or Model()
    registry = CapabilityRegistry()
    registry.register(
        ComponentMetadata(
            name="answerer",
            component_type="model",
            capabilities=["answer"],
            cost_per_call=0.0,
            avg_latency_ms=200,
            reliability_score=0.9,
            privacy_level="external",
        )
    )
    pipelines = enumerate_pipelines(registry, TaskSpecification(required_capabilities=["answer"]))
    eval_set = EvalSet(
        examples=[Example(i, expected="yes") for i in range(n)],
        scorer=lambda output, expected: output == expected,
    )
    executor = AgentExecutor(backend=model)
    evaluations = evaluate_pipelines(pipelines, executor, eval_set)
    decision = record_decision(evaluations, eval_set, backend=model, resamples=100)
    return model, pipelines, executor, eval_set, decision


def check(model, pipelines, executor, eval_set, decision, **options):
    return verify_decision(
        decision, pipelines, executor, eval_set, clock=lambda: FIXED, **options
    )


def test_record_keeps_scores_and_times_for_each_example():
    _, _, _, _, decision = world(n=10)
    kept = decision.chosen["examples"]
    assert kept["scores"] == [1.0] * 10
    assert kept["latencies_ms"] == [200.0] * 10
    assert decision.data["format"] == 2


def test_nothing_changed_is_valid_but_not_confirmed_with_few_examples():
    result = check(*world(n=40))

    assert result.status == "valid" and result.ok and result.exit_code == 0
    assert result.confirmed is False
    assert result.asked == result.examples == 40
    assert result.quality_then == result.quality_now == 1.0
    assert 0.1 < result.detectable < 0.5
    assert "No drop found" in result.message and "would go unnoticed" in result.message


def test_many_stable_examples_confirm_and_stop_early():
    result = check(*world(n=400))

    assert result.status == "valid" and result.confirmed is True
    assert result.asked < 400
    assert result.change.low > -0.05
    assert "Still valid" in result.message


def test_a_large_drop_is_caught_before_all_examples_are_asked():
    model, *rest = world(n=60)
    model.wrong = set(range(0, 60, 2)) | set(range(1, 60, 4))  # three in four now wrong
    before = model.calls

    result = check(model, *rest)

    assert result.status == "quality_dropped" and result.exit_code == 1
    assert result.asked < 60 and model.calls - before == result.asked
    assert result.quality_now < result.quality_then
    assert result.change.high < -0.05
    assert "Quality dropped from 1.00" in result.message


def test_a_small_drop_within_few_examples_is_not_an_alarm():
    model, *rest = world(n=40)
    model.wrong = {0, 1}

    result = check(model, *rest)

    assert result.status == "valid" and result.confirmed is False
    assert result.quality_now == pytest.approx(0.95)


def test_a_retired_model_is_unavailable_after_three_calls():
    model, *rest = world(n=40)
    model.retired = True
    before = model.calls

    result = check(model, *rest)

    assert result.status == "unavailable" and result.exit_code == 4
    assert model.calls - before == 3
    assert result.errors == {"not_found": 3}
    assert "retired or renamed" in result.message


def test_slower_needs_most_runs_to_be_slow():
    model, *rest = world(n=40)
    model.latency_ms = 900.0
    slow = check(model, *rest)

    model.latency_ms = 200.0
    model.stalled = {3}
    one_stall = check(model, *rest)

    model.stalled = set()
    model.latency_ms = 260.0  # 30% slower: under twice the recorded median
    a_little = check(model, *rest)

    assert slow.status == "slower" and slow.exit_code == 3
    assert slow.latency_then_ms == 200.0 and slow.latency_now_ms == 900.0
    assert "It took 200 ms when recorded" in slow.message
    assert one_stall.status == "valid"
    assert a_little.status == "valid"


def test_max_latency_sets_an_absolute_limit():
    model, *rest = world(n=40)
    model.latency_ms = 350.0

    assert check(model, *rest).status == "valid"
    assert check(model, *rest, max_latency_ms=300).status == "slower"


def test_tiny_steps_are_never_called_slower():
    model = Model()
    model.latency_ms = 0.01
    model, *rest = world(n=40, model=model)
    model.latency_ms = 0.05  # five times slower, and nobody would notice

    assert check(model, *rest).status == "valid"


def test_other_examples_or_pipelines_are_not_comparable():
    model, pipelines, executor, eval_set, decision = world(n=20)
    other = EvalSet(
        examples=[Example(i + 100, expected="yes") for i in range(20)], scorer=eval_set.scorer
    )
    shorter = EvalSet(examples=list(eval_set.examples)[:10], scorer=eval_set.scorer)
    before = model.calls

    assert check(model, pipelines, executor, other, decision).status == "not_comparable"
    assert check(model, pipelines, executor, shorter, decision).status == "not_comparable"
    missing = check(model, [], executor, eval_set, decision)
    assert missing.status == "not_comparable" and missing.exit_code == 5
    assert model.calls == before  # nothing was run


def test_a_changed_model_name_is_not_comparable():
    model, *rest = world(n=20)
    model.name = "model-b"

    result = check(model, *rest)

    assert result.status == "not_comparable"
    assert "model 'model-a'" in result.message and "'model-b' now" in result.message


def test_fresh_answers_are_logged_and_old_ones_not_reused(tmp_path):
    model = Model()
    logged = LoggedBackend(model, tmp_path / "answers.jsonl")
    registry = CapabilityRegistry()
    registry.register(
        ComponentMetadata(
            name="answerer", component_type="model", capabilities=["answer"],
            cost_per_call=0.0, avg_latency_ms=200, reliability_score=0.9,
            privacy_level="external",
        )
    )
    pipelines = enumerate_pipelines(registry, TaskSpecification(required_capabilities=["answer"]))
    eval_set = EvalSet(
        examples=[Example(i, expected="yes") for i in range(30)],
        scorer=lambda output, expected: output == expected,
    )
    executor = AgentExecutor(backend=logged)
    decision = record_decision(
        evaluate_pipelines(pipelines, executor, eval_set), eval_set, backend=logged, resamples=100
    )
    assert model.calls == 30

    model.wrong = set(range(30))  # the model now fails; the log still holds good answers
    result = check(model, pipelines, executor, eval_set, decision)

    assert result.status == "quality_dropped"
    assert model.calls == 30 + result.asked
    labels = {record["label"] for record in AnswerLog(tmp_path / "answers.jsonl").records()}
    assert labels == {"", "verify 2026-11-21T08:00:00+00:00"}
    assert result.label == "verify 2026-11-21T08:00:00+00:00"


def test_a_record_without_per_example_scores_is_refused():
    model, pipelines, executor, eval_set, decision = world(n=10)
    old = json.loads(json.dumps(decision.to_dict()))
    del old["chosen"]["examples"]

    with pytest.raises(ValueError, match="record it again"):
        verify_decision(Decision(old), pipelines, executor, eval_set)


def test_result_round_trips_through_json():
    result = check(*world(n=20))
    data = json.loads(json.dumps(result.to_dict()))
    assert data["status"] == "valid" and data["change"]["n"] == 20
    assert data["time"] == "2026-11-21T08:00:00+00:00"


# --- the command -----------------------------------------------------------

SETUP = '''
import json
from pathlib import Path

from frontierkit import (
    AgentExecutor, CallableBackend, CapabilityRegistry, ComponentMetadata, EvalSet, Example,
    TaskSpecification, enumerate_pipelines,
)

STATE = Path(__file__).with_name("state.json")


class Gone(Exception):
    status_code = 404


def answer(n):
    state = json.loads(STATE.read_text())
    if state == "retired":
        raise Gone("no such model")
    return "no" if state == "worse" else "yes"


def setup():
    registry = CapabilityRegistry()
    registry.register(ComponentMetadata(
        name="answerer", component_type="model", capabilities=["answer"], cost_per_call=0.0,
        avg_latency_ms=1, reliability_score=0.9, privacy_level="internal"))
    return {
        "pipelines": enumerate_pipelines(
            registry, TaskSpecification(required_capabilities=["answer"])),
        "executor": AgentExecutor(backend=CallableBackend({"answerer": answer})),
        "eval_set": EvalSet(
            examples=[Example(i, expected="yes") for i in range(40)],
            scorer=lambda output, expected: output == expected),
    }


def broken():
    return ["not", "a", "dict"]
'''


@pytest.fixture
def project(tmp_path):
    setup_file = tmp_path / "compare.py"
    setup_file.write_text(SETUP, encoding="utf-8")
    state = tmp_path / "state.json"
    state.write_text('"good"', encoding="utf-8")
    given = load_setup(str(setup_file))
    evaluations = evaluate_pipelines(given["pipelines"], given["executor"], given["eval_set"])
    decision_file = record_decision(evaluations, given["eval_set"], resamples=100).save(
        tmp_path / "decision.json"
    )
    return setup_file, state, decision_file


def run(*arguments):
    lines = []
    code = main(list(arguments), out=lines.append)
    return code, "\n".join(lines)


def test_command_exit_codes_follow_what_happened(project):
    setup_file, state, decision_file = project

    code, text = run("verify", str(decision_file), "--setup", str(setup_file))
    assert code == 0 and text.startswith("valid: answerer")

    state.write_text('"worse"', encoding="utf-8")
    code, text = run("verify", str(decision_file), "--setup", str(setup_file))
    assert code == 1 and text.startswith("quality_dropped: answerer")

    state.write_text('"retired"', encoding="utf-8")
    code, text = run("verify", str(decision_file), "--setup", f"{setup_file}:setup")
    assert code == 4 and "retired or renamed" in text

    history = [
        json.loads(line)
        for line in history_path(decision_file).read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["status"] for entry in history] == ["valid", "quality_dropped", "unavailable"]


def test_command_prints_json_and_can_skip_the_history(project):
    setup_file, _, decision_file = project

    code, text = run(
        "verify", str(decision_file), "--setup", str(setup_file), "--json", "--no-history"
    )

    assert code == 0 and json.loads(text)["status"] == "valid"
    assert not history_path(decision_file).exists()
    assert load_decision(decision_file).chosen["components"] == ["answerer"]


def test_command_reports_what_it_cannot_run(project, tmp_path):
    setup_file, _, decision_file = project

    assert run("verify", str(tmp_path / "none.json"), "--setup", str(setup_file))[0] == 2
    code, text = run("verify", str(decision_file), "--setup", str(tmp_path / "none.py"))
    assert code == 2 and "setup file not found" in text
    code, text = run("verify", str(decision_file), "--setup", f"{setup_file}:missing")
    assert code == 2 and "does not define a function named missing" in text
    code, text = run("verify", str(decision_file), "--setup", f"{setup_file}:broken")
    assert code == 2 and "must return a dict" in text
    not_a_record = tmp_path / "other.json"
    not_a_record.write_text("{}", encoding="utf-8")
    assert run("verify", str(not_a_record), "--setup", str(setup_file))[0] == 2
