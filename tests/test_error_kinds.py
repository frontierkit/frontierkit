"""Failed steps are recorded with what went wrong, not as wrong answers."""

import time

import pytest

from chatcortex import (
    AgentExecutor,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    EvalSet,
    Example,
    StepResult,
    TaskSpecification,
    classify_error,
    enumerate_pipelines,
    evaluate_pipelines,
)
from chatcortex.adapters.openai_backend import OpenAIBackend
from chatcortex.execution.errors import SERVICE_ERRORS
from chatcortex.execution.executor import ProbeResult


class RateLimitError(Exception):
    status_code = 429


class APITimeoutError(ConnectionError):
    """Named and derived the way SDK timeouts usually are."""


class NotFoundError(Exception):
    status_code = 404


class _Response:
    status_code = 503


class WrappedServerError(Exception):
    def __init__(self):
        super().__init__("upstream")
        self.response = _Response()


@pytest.mark.parametrize(
    "exc, kind",
    [
        (RateLimitError("slow down"), "rate_limited"),
        (APITimeoutError("too slow"), "timeout"),
        (TimeoutError("too slow"), "timeout"),
        (NotFoundError("no such model"), "not_found"),
        (WrappedServerError(), "unavailable"),
        (ConnectionRefusedError("refused"), "unavailable"),
        (type("AuthenticationError", (Exception,), {})("bad key"), "auth"),
        (type("Gone", (Exception,), {"status_code": 410})("retired"), "not_found"),
        (type("Forbidden", (Exception,), {"status_code": 403})("no"), "auth"),
        (ValueError("bug in my step"), "error"),
    ],
)
def test_classify_error(exc, kind):
    assert classify_error(exc) == kind


def test_service_errors_exclude_answer_problems():
    assert "rate_limited" in SERVICE_ERRORS
    assert "step_failed" not in SERVICE_ERRORS
    assert "empty_output" not in SERVICE_ERRORS
    assert "error" not in SERVICE_ERRORS


def _one_step(name="model"):
    registry = CapabilityRegistry()
    registry.register(
        ComponentMetadata(
            name=name,
            component_type="tool",
            capabilities=["answer"],
            cost_per_call=0.01,
            avg_latency_ms=10,
            reliability_score=0.9,
            privacy_level="internal",
        )
    )
    task = TaskSpecification(required_capabilities=["answer"])
    (pipeline,) = enumerate_pipelines(registry, task)
    return pipeline


def test_callable_backend_names_the_kind_of_a_raised_error():
    def limited(_):
        raise RateLimitError("slow down")

    pipeline = _one_step()
    result = AgentExecutor(backend=CallableBackend({"model": limited})).execute(
        pipeline.graph, task_input="q"
    )

    assert result.success is False
    assert result.error == "rate_limited"
    assert result.telemetry.records[0]["error"] == "rate_limited"


def test_reported_failure_is_step_failed_unless_named():
    pipeline = _one_step()

    plain = AgentExecutor(
        backend=CallableBackend({"model": lambda _: StepResult(None, success=False)})
    ).execute(pipeline.graph, task_input="q")
    named = AgentExecutor(
        backend=CallableBackend(
            {"model": lambda _: StepResult(None, success=False, error="timeout")}
        )
    ).execute(pipeline.graph, task_input="q")

    assert plain.error == "step_failed"
    assert named.error == "timeout"


def test_success_has_no_error_even_if_a_step_names_one():
    pipeline = _one_step()
    result = AgentExecutor(
        backend=CallableBackend({"model": lambda _: StepResult("ok", error="timeout")})
    ).execute(pipeline.graph, task_input="q")

    assert result.success is True
    assert result.error is None
    assert result.telemetry.records[0]["error"] is None


def test_executor_measures_time_and_kind_when_a_backend_raises():
    def backend(metadata, node_input):
        time.sleep(0.02)
        raise APITimeoutError("too slow")

    pipeline = _one_step()
    result = AgentExecutor(backend=backend).execute(pipeline.graph, task_input="q")

    (record,) = result.telemetry.records
    assert record["success"] is False
    assert record["error"] == "timeout"
    assert record["latency_ms"] >= 15
    assert record["cost"] == 0.0


def test_backend_written_before_error_kinds_still_works():
    """A failed probe that names no kind is recorded as step_failed."""

    def backend(metadata, node_input):
        return ProbeResult(success=False, latency_ms=5.0, cost=0.0, output=None)

    pipeline = _one_step()
    result = AgentExecutor(backend=backend).execute(pipeline.graph, task_input="q")

    assert result.error == "step_failed"


class _Completions:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def create(self, **_):
        return self._behaviour()


class _Client:
    def __init__(self, behaviour):
        self.chat = type("Chat", (), {"completions": _Completions(behaviour)})()


def _empty_response():
    message = type("Message", (), {"content": ""})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice], "usage": None})()


def test_openai_backend_names_rate_limits_and_empty_answers():
    def limited():
        raise RateLimitError("slow down")

    ((_, metadata),) = _one_step().graph.iter_components()

    refused = OpenAIBackend(model="m", client=_Client(limited))(metadata, "q")
    empty = OpenAIBackend(model="m", client=_Client(_empty_response))(metadata, "q")

    assert (refused.success, refused.error) == (False, "rate_limited")
    assert (empty.success, empty.error) == (False, "empty_output")


def test_evaluation_separates_outages_from_wrong_answers():
    # Right on every question it answers, but rate limited on half of them.
    def limited_half(question):
        if question in ("q2", "q4"):
            raise RateLimitError("slow down")
        return "yes"

    pipeline = _one_step()
    eval_set = EvalSet(
        examples=[Example(f"q{i}", expected="yes") for i in range(1, 5)],
        scorer=lambda output, expected: output == expected,
    )
    (evaluation,) = evaluate_pipelines(
        [pipeline], AgentExecutor(backend=CallableBackend({"model": limited_half})), eval_set
    )

    assert evaluation.errors == (None, "rate_limited", None, "rate_limited")
    assert evaluation.error_counts == {"rate_limited": 2}
    assert evaluation.quality == 0.5
    assert evaluation.quality_when_answered == 1.0
    assert evaluation.success_rate == 0.5
    assert evaluation.to_dict()["errors"] == {"rate_limited": 2}


def test_evaluation_without_failures_reports_no_errors():
    pipeline = _one_step()
    eval_set = EvalSet(
        examples=[Example("q1", expected="yes"), Example("q2", expected="no")],
        scorer=lambda output, expected: output == expected,
    )
    (evaluation,) = evaluate_pipelines(
        [pipeline], AgentExecutor(backend=CallableBackend({"model": lambda _: "yes"})), eval_set
    )

    assert evaluation.errors == (None, None)
    assert evaluation.error_counts == {}
    assert evaluation.quality_when_answered == evaluation.quality == 0.5
    assert "errors" not in evaluation.to_dict()


def test_quality_when_answered_is_none_when_nothing_finished():
    def down(_):
        raise ConnectionRefusedError("refused")

    pipeline = _one_step()
    eval_set = EvalSet(examples=[Example("q1", expected="yes")], scorer=lambda o, e: o == e)
    (evaluation,) = evaluate_pipelines(
        [pipeline], AgentExecutor(backend=CallableBackend({"model": down})), eval_set
    )

    assert evaluation.quality == 0.0
    assert evaluation.quality_when_answered is None
    assert evaluation.error_counts == {"unavailable": 1}
