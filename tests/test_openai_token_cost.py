"""OpenAIBackend costs a call from reported token usage when given prices.

A stub client stands in for the SDK, so no key or network is needed.
"""

from types import SimpleNamespace

import pytest

from frontierkit import ComponentMetadata
from frontierkit.adapters.openai_backend import OpenAIBackend, TokenPrice

DECLARED_COST = 0.05


def _metadata():
    return ComponentMetadata(
        name="writer",
        component_type="model",
        capabilities=["generation"],
        cost_per_call=DECLARED_COST,
        avg_latency_ms=100,
        reliability_score=0.9,
        privacy_level="external",
    )


def _client(usage=None, content="hello", error=None):
    def create(**kwargs):
        if error is not None:
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _usage(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


def test_cost_is_usage_times_price():
    backend = OpenAIBackend(
        model="m",
        client=_client(usage=_usage(1_000, 500)),
        prices={"m": TokenPrice(input_per_million=2.0, output_per_million=8.0)},
    )
    probe = backend(_metadata(), "question")
    # 1,000 * 2 / 1e6 + 500 * 8 / 1e6
    assert probe.cost == pytest.approx(0.002 + 0.004)
    assert probe.success is True


def test_price_follows_the_model_each_component_runs_on():
    backend = OpenAIBackend(
        model_for=lambda metadata: "big",
        client=_client(usage=_usage(1_000_000, 0)),
        prices={"small": TokenPrice(1.0, 1.0), "big": TokenPrice(10.0, 30.0)},
    )
    assert backend(_metadata(), "q").cost == pytest.approx(10.0)


def test_without_prices_the_declared_cost_is_recorded():
    backend = OpenAIBackend(model="m", client=_client(usage=_usage(1_000, 500)))
    assert backend(_metadata(), "q").cost == DECLARED_COST


def test_unpriced_model_falls_back_to_declared_cost():
    backend = OpenAIBackend(
        model="other",
        client=_client(usage=_usage(1_000, 500)),
        prices={"m": TokenPrice(2.0, 8.0)},
    )
    assert backend(_metadata(), "q").cost == DECLARED_COST


def test_response_without_usage_falls_back_to_declared_cost():
    backend = OpenAIBackend(
        model="m", client=_client(usage=None), prices={"m": TokenPrice(2.0, 8.0)}
    )
    assert backend(_metadata(), "q").cost == DECLARED_COST


def test_failed_call_records_zero_cost():
    backend = OpenAIBackend(
        model="m",
        client=_client(error=RuntimeError("offline")),
        prices={"m": TokenPrice(2.0, 8.0)},
    )
    probe = backend(_metadata(), "q")
    assert probe.success is False
    assert probe.cost == 0.0


def test_invalid_prices_are_rejected():
    with pytest.raises(ValueError, match="input_per_million"):
        TokenPrice(input_per_million=-1.0, output_per_million=1.0)
    with pytest.raises(TypeError, match="TokenPrice"):
        OpenAIBackend(model="m", client=_client(), prices={"m": (2.0, 8.0)})
