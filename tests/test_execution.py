import pytest

from chatcortex import AgentGraph, CapabilityRegistry, ComponentMetadata
from chatcortex.execution.executor import AgentExecutor


def build_graph():
    g = AgentGraph()
    g.add_component(
        "a",
        ComponentMetadata(
            name="a",
            component_type="tool",
            capabilities=["x"],
            cost_per_call=0.001,
            avg_latency_ms=1,
            reliability_score=0.5,
            privacy_level="internal",
        ),
    )
    return g


def test_deterministic_execution():
    result = AgentExecutor(mode="deterministic").execute(build_graph())
    assert result.summary() is not None


def _truth(reliability=0.5, latency=1, cost=0.001, name="a"):
    registry = CapabilityRegistry()
    registry.register(
        ComponentMetadata(
            name=name,
            component_type="tool",
            capabilities=["x"],
            cost_per_call=cost,
            avg_latency_ms=latency,
            reliability_score=reliability,
            privacy_level="internal",
        )
    )
    return registry


def _records(executor, runs=1):
    records = []
    for _ in range(runs):
        records.extend(executor.execute(build_graph()).telemetry.records)
    return records


def test_ground_truth_latency_and_cost_replace_declared_values():
    executor = AgentExecutor(ground_truth=_truth(latency=250, cost=0.02))
    (record,) = _records(executor)
    assert record["latency_ms"] == 250
    assert record["cost"] == 0.02


def test_ground_truth_reliability_drives_success_not_the_declaration():
    # Declared reliability is 0.5; the hidden truth says the component
    # never works, then that it always works.
    never = AgentExecutor(
        mode="probabilistic", seed=7, ground_truth=_truth(reliability=0.0)
    )
    # random() <= 0.0 is possible only at exactly 0.0, so allow none.
    assert not any(r["success"] for r in _records(never, runs=200))

    always = AgentExecutor(
        mode="probabilistic", seed=7, ground_truth=_truth(reliability=1.0)
    )
    assert all(r["success"] for r in _records(always, runs=200))


def test_component_missing_from_ground_truth_behaves_as_declared():
    executor = AgentExecutor(ground_truth=_truth(latency=999, name="other"))
    (record,) = _records(executor)
    assert record["latency_ms"] == 1
    assert record["cost"] == 0.001


def test_no_ground_truth_keeps_the_declared_simulation():
    with_none = _records(AgentExecutor(mode="probabilistic", seed=3), runs=50)
    with_same = _records(
        AgentExecutor(mode="probabilistic", seed=3, ground_truth=_truth()),
        runs=50,
    )
    assert with_none == with_same


def test_ground_truth_with_backend_is_rejected():
    with pytest.raises(ValueError, match="ground_truth"):
        AgentExecutor(backend=lambda metadata, node_input: None, ground_truth=_truth())


def test_probabilistic_execution_is_seeded():
    r1 = AgentExecutor(mode="probabilistic", seed=42).execute(build_graph())
    r2 = AgentExecutor(mode="probabilistic", seed=42).execute(build_graph())
    assert r1.summary() == r2.summary()
