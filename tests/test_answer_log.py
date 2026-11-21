"""The answer log keeps every answer and reuses the successful ones."""

import json
import warnings
from datetime import datetime, timezone

import pytest

from chatcortex import (
    AgentExecutor,
    AnswerLog,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    EvalSet,
    Example,
    LoggedBackend,
    RoutedBackend,
    StepResult,
    TaskSpecification,
    enumerate_pipelines,
    evaluate_pipelines,
)
from chatcortex.adapters.openai_backend import OpenAIBackend
from chatcortex.store import fingerprint, identify


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


def two_step_pipelines():
    registry = CapabilityRegistry()
    for name, capability in (("upper", "shout"), ("lower", "shout"), ("exclaim", "finish")):
        registry.register(component(name, capability))
    task = TaskSpecification(required_capabilities=["shout", "finish"])
    return enumerate_pipelines(registry, task)


class Counting:
    """Callable implementations that count how often each one ran."""

    def __init__(self):
        self.calls = {"upper": 0, "lower": 0, "exclaim": 0}

    def backend(self):
        def make(name, fn):
            def run(value):
                self.calls[name] += 1
                return fn(value)

            return run

        return CallableBackend(
            {
                "upper": make("upper", str.upper),
                "lower": make("lower", str.lower),
                "exclaim": make("exclaim", lambda text: text + "!"),
            }
        )


EVAL_SET = EvalSet(
    examples=[Example("Hi", expected="HI!"), Example("Yo", expected="YO!")],
    scorer=lambda output, expected: output == expected,
)


def test_second_run_makes_no_calls_and_gives_the_same_result(tmp_path):
    path = tmp_path / "answers.jsonl"
    pipelines = two_step_pipelines()

    first_work = Counting()
    first = LoggedBackend(first_work.backend(), path)
    before = evaluate_pipelines(pipelines, AgentExecutor(backend=first), EVAL_SET)

    second_work = Counting()
    second = LoggedBackend(second_work.backend(), path)
    after = evaluate_pipelines(pipelines, AgentExecutor(backend=second), EVAL_SET)

    assert sum(first_work.calls.values()) == first.calls == 8
    assert sum(second_work.calls.values()) == second.calls == 0
    assert second.reused == 8
    assert [e.to_dict() for e in before] == [e.to_dict() for e in after]
    assert [e.scores for e in before] == [e.scores for e in after]


def test_log_records_what_was_asked_and_what_ran(tmp_path):
    path = tmp_path / "answers.jsonl"
    fixed = datetime(2026, 11, 21, 9, 30, tzinfo=timezone.utc)
    backend = LoggedBackend(
        CallableBackend({"upper": str.upper}),
        path,
        versions={"upper": "v2"},
        label="monday",
        clock=lambda: fixed,
    )

    backend(component("upper", "shout"), "hi")

    (record,) = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert record["component"] == "upper"
    assert record["identity"] == {"component": "upper", "version": "v2"}
    assert record["input"] == "hi"
    assert record["output"] == "HI"
    assert record["label"] == "monday"
    assert record["time"] == "2026-11-21T09:30:00+00:00"
    assert record["success"] is True and record["error"] is None
    assert record["key"] == fingerprint(record["identity"], "hi")


def test_changing_the_version_asks_again(tmp_path):
    path = tmp_path / "answers.jsonl"
    metadata = component("upper", "shout")
    LoggedBackend(CallableBackend({"upper": str.upper}), path)(metadata, "hi")

    same = LoggedBackend(CallableBackend({"upper": str.title}), path)
    changed = LoggedBackend(CallableBackend({"upper": str.title}), path, versions={"upper": 2})

    assert same(metadata, "hi").output == "HI"  # old answer, the code change went unseen
    assert changed(metadata, "hi").output == "Hi"
    assert (same.calls, changed.calls) == (0, 1)


def test_labels_keep_separate_samples(tmp_path):
    path = tmp_path / "answers.jsonl"
    metadata = component("upper", "shout")
    monday = LoggedBackend(CallableBackend({"upper": str.upper}), path, label="mon")
    tuesday = LoggedBackend(CallableBackend({"upper": str.upper}), path, label="tue")

    monday(metadata, "hi")
    tuesday(metadata, "hi")
    monday(metadata, "hi")

    assert (monday.calls, monday.reused, tuesday.calls) == (1, 1, 1)
    assert len(AnswerLog(path)) == 2


def test_reuse_off_always_calls_and_logs(tmp_path):
    path = tmp_path / "answers.jsonl"
    backend = LoggedBackend(CallableBackend({"upper": str.upper}), path, reuse=False)
    metadata = component("upper", "shout")

    backend(metadata, "hi")
    backend(metadata, "hi")

    assert backend.calls == 2
    assert len(AnswerLog(path).records()) == 2


def test_failures_are_logged_but_tried_again(tmp_path):
    path = tmp_path / "answers.jsonl"
    metadata = component("upper", "shout")
    attempts = []

    def flaky(text):
        attempts.append(text)
        if len(attempts) == 1:
            raise TimeoutError("too slow")
        return text.upper()

    backend = LoggedBackend(CallableBackend({"upper": flaky}), path)
    first = backend(metadata, "hi")
    second = backend(metadata, "hi")
    third = backend(metadata, "hi")

    assert (first.success, first.error) == (False, "timeout")
    assert (second.success, second.output) == (True, "HI")
    assert third.output == "HI" and len(attempts) == 2
    assert [r["error"] for r in AnswerLog(path).records()] == ["timeout", None]


def test_values_that_are_not_json_run_unlogged_with_one_warning(tmp_path):
    path = tmp_path / "answers.jsonl"
    metadata = component("upper", "shout")
    backend = LoggedBackend(CallableBackend({"upper": lambda _: {1, 2}}), path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = backend(metadata, "hi")
        second = backend(metadata, "hi")

    assert first.output == second.output == {1, 2}
    assert backend.calls == 2 and len(AnswerLog(path)) == 0
    assert len(caught) == 1 and "not JSON data" in str(caught[0].message)


def test_step_cost_and_latency_are_kept_for_reuse(tmp_path):
    path = tmp_path / "answers.jsonl"
    metadata = component("upper", "shout")
    first = LoggedBackend(
        CallableBackend({"upper": lambda t: StepResult(t.upper(), cost=0.5)}), path
    )(metadata, "hi")
    again = LoggedBackend(CallableBackend({"upper": str.upper}), path)(metadata, "hi")

    assert again.cost == first.cost == 0.5
    assert again.latency_ms == first.latency_ms


def test_log_rejects_a_file_that_is_not_a_log(tmp_path):
    path = tmp_path / "other.jsonl"
    path.write_text('{"something": "else"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not an answer record"):
        AnswerLog(path)


class _Client:
    base_url = "https://user:secret@example.test:8443/v1/?token=abc"


def test_openai_identity_names_endpoint_model_and_options_without_secrets():
    backend = OpenAIBackend(
        model="small",
        client=_Client(),
        request_options={"temperature": 0, "timeout": 30, "extra_headers": {"X-Key": "s3cret"}},
    )
    identity = identify(backend, component("writer", "write"))

    assert identity == {
        "component": "writer",
        "endpoint": "https://example.test:8443/v1",
        "model": "small",
        "options": {"temperature": 0},
    }
    assert "secret" not in json.dumps(identity) and "s3cret" not in json.dumps(identity)


def test_a_different_model_does_not_reuse_answers(tmp_path):
    path = tmp_path / "answers.jsonl"
    metadata = component("writer", "write")

    def backend(model):
        inner = OpenAIBackend(model=model, client=_Client())
        return identify(inner, metadata)

    assert fingerprint(backend("small"), "q") != fingerprint(backend("large"), "q")
    assert fingerprint(backend("small"), "q") == fingerprint(backend("small"), "q")
    assert not path.exists()


def test_routed_backend_passes_identity_through():
    routed = RoutedBackend(
        {"writer": OpenAIBackend(model="small", client=_Client())},
        default=CallableBackend({"upper": str.upper}),
    )

    assert identify(routed, component("writer", "write"))["model"] == "small"
    assert identify(routed, component("upper", "shout")) == {"component": "upper"}
