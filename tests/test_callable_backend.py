"""CallableBackend: pipelines of plain Python functions run for real."""

import pytest

from chatcortex import (
    AgentExecutor,
    AgentGraph,
    BackendConfigurationError,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    StepResult,
)


def _component(name, cost=0.01):
    return ComponentMetadata(
        name=name,
        component_type="tool",
        capabilities=[name],
        cost_per_call=cost,
        avg_latency_ms=10,
        reliability_score=0.9,
        privacy_level="internal",
    )


def _chain(*names):
    graph = AgentGraph()
    for index, name in enumerate(names):
        graph.chain_append(f"{name}_{index}", _component(name))
    return graph


def test_each_step_receives_the_previous_output():
    backend = CallableBackend(
        {
            "split": lambda text: text.split(),
            "count": lambda words: len(words),
            "label": lambda n: f"{n} words",
        }
    )
    result = AgentExecutor(backend=backend).execute(
        _chain("split", "count", "label"), task_input="one two three"
    )
    assert result.success is True
    assert result.output == "3 words"
    assert [r["success"] for r in result.telemetry.records] == [True, True, True]


def test_latency_is_measured_and_cost_defaults_to_declared():
    backend = CallableBackend({"a": lambda value: value})
    (record,) = AgentExecutor(backend=backend).execute(_chain("a")).telemetry.records
    assert record["latency_ms"] >= 0.0
    assert record["cost"] == 0.01


def test_step_result_reports_cost_and_failure_without_raising():
    backend = CallableBackend(
        {
            "a": lambda value: StepResult(output="x", cost=0.5),
            "b": lambda value: StepResult(output=None, success=False),
            "c": lambda value: pytest.fail("must not run after a failure"),
        }
    )
    result = AgentExecutor(backend=backend).execute(_chain("a", "b", "c"))
    records = result.telemetry.records
    assert [r["success"] for r in records] == [True, False]
    assert records[0]["cost"] == 0.5
    assert result.success is False
    assert result.output is None


def test_exception_in_a_component_is_a_failed_probe_and_stops_the_chain():
    def boom(_):
        raise RuntimeError("no index")

    backend = CallableBackend({"a": boom, "b": lambda value: value})
    result = AgentExecutor(backend=backend).execute(_chain("a", "b"))
    (record,) = result.telemetry.records
    assert record["success"] is False
    assert record["cost"] == 0.01
    assert result.success is False


def test_missing_implementation_raises_instead_of_recording_a_failure():
    backend = CallableBackend({"a": lambda value: value})
    with pytest.raises(BackendConfigurationError, match="'b'"):
        AgentExecutor(backend=backend).execute(_chain("a", "b"))


def test_missing_lists_unimplemented_registry_components():
    registry = CapabilityRegistry()
    for name in ("a", "b", "c"):
        registry.register(_component(name))
    backend = CallableBackend({"a": lambda value: value, "c": lambda value: value})
    assert backend.missing(registry) == ["b"]


def test_non_callable_implementation_is_rejected():
    with pytest.raises(TypeError, match="'a'"):
        CallableBackend({"a": "not a function"})


def test_simulator_result_has_no_output_and_ignores_task_input():
    result = AgentExecutor().execute(_chain("a"), task_input="ignored")
    assert result.output is None
    assert result.success is True
