"""OpenAI-compatible endpoints and per-component backend routing.

No key or network: a fake ``openai`` module records how the SDK client
was built, and stub clients stand in for the vendors.
"""

import sys
from types import SimpleNamespace

import pytest

from frontierkit import (
    AgentExecutor,
    AgentGraph,
    BackendConfigurationError,
    CallableBackend,
    CapabilityRegistry,
    ComponentMetadata,
    RoutedBackend,
)
from frontierkit.adapters.openai_backend import OpenAIBackend


def _component(name):
    return ComponentMetadata(
        name=name,
        component_type="model",
        capabilities=[name],
        cost_per_call=0.01,
        avg_latency_ms=10,
        reliability_score=0.9,
        privacy_level="external",
    )


def _stub_client(reply, seen=None):
    def create(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=None,
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture
def fake_sdk(monkeypatch):
    built = []

    def client_factory(**kwargs):
        built.append(kwargs)
        return _stub_client("ok")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client_factory))
    return built


def test_base_url_and_api_key_reach_the_sdk_client(fake_sdk):
    OpenAIBackend(model="m", base_url="https://example.test/v1", api_key="k")
    assert fake_sdk == [{"base_url": "https://example.test/v1", "api_key": "k"}]


def test_sdk_defaults_are_untouched_when_nothing_is_given(fake_sdk):
    OpenAIBackend(model="m")
    assert fake_sdk == [{}]


def test_request_options_are_sent_with_every_call():
    seen = []
    backend = OpenAIBackend(
        model="m",
        client=_stub_client("hi", seen),
        request_options={"max_tokens": 50, "temperature": 0},
    )
    backend(_component("a"), "question")
    assert seen[0]["max_tokens"] == 50
    assert seen[0]["temperature"] == 0
    assert seen[0]["model"] == "m"


@pytest.mark.parametrize("reserved", ["model", "messages"])
def test_request_options_cannot_override_model_or_messages(reserved):
    with pytest.raises(ValueError, match=reserved):
        OpenAIBackend(model="m", client=_stub_client("x"), request_options={reserved: 1})


def _graph(*names):
    graph = AgentGraph()
    for index, name in enumerate(names):
        graph.chain_append(f"{name}_{index}", _component(name))
    return graph


def test_routed_backend_sends_each_component_to_its_own_backend():
    backend = RoutedBackend(
        {
            "vendor_a": OpenAIBackend(model="a", client=_stub_client("from a")),
            "vendor_b": OpenAIBackend(model="b", client=_stub_client("from b")),
        }
    )
    executor = AgentExecutor(backend=backend)
    assert executor.execute(_graph("vendor_a"), task_input="q").output == "from a"
    assert executor.execute(_graph("vendor_b"), task_input="q").output == "from b"


def test_routed_backend_default_handles_unrouted_components():
    backend = RoutedBackend(
        {"model": OpenAIBackend(model="m", client=_stub_client("answer"))},
        default=CallableBackend({"prep": lambda text: text.upper()}),
    )
    result = AgentExecutor(backend=backend).execute(
        _graph("prep", "model"), task_input="q"
    )
    assert result.success is True
    assert result.output == "answer"


def test_unrouted_component_without_default_raises():
    backend = RoutedBackend({"a": _stub_backend()})
    with pytest.raises(BackendConfigurationError, match="'b'"):
        AgentExecutor(backend=backend).execute(_graph("b"))


def test_unrouted_lists_registry_components_without_a_route():
    registry = CapabilityRegistry()
    for name in ("a", "b"):
        registry.register(_component(name))
    assert RoutedBackend({"a": _stub_backend()}).unrouted(registry) == ["b"]
    assert RoutedBackend({"a": _stub_backend()}, default=_stub_backend()).unrouted(registry) == []


def test_non_callable_route_is_rejected():
    with pytest.raises(TypeError, match="'a'"):
        RoutedBackend({"a": "not a backend"})


def _stub_backend():
    return OpenAIBackend(model="m", client=_stub_client("x"))
