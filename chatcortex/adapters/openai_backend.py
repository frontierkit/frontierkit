"""OpenAI chat-completions execution backend (optional ``real`` extra).

Installation: ``pip install chatcortex[real]`` — this module imports
``openai`` lazily inside :func:`OpenAIBackend.__init__`, so the package
never imports it unless a backend is actually built. The API key comes
from ``OPENAI_API_KEY`` (the SDK's standard env resolution); tests can
inject a stub ``client`` instead and need no key or network.

What is measured vs. declared:

- **Latency** and **success** are measured from the real SDK call.
- **Cost** is measured when you supply prices: pass ``prices`` mapping a
  model name to a :class:`TokenPrice`, and each call is costed from the
  token usage the API reports. Prices are yours to supply because they
  are model-specific and change over time. For a model without a price,
  or a response without usage, the component's declared ``cost_per_call``
  is recorded instead. A call that raises records zero cost.

Component metadata stays pure: nothing here mutates it, calibration is
the component that turns telemetry into updated metadata.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from ..execution.executor import ProbeResult
from ..registry.metadata import ComponentMetadata

DEFAULT_MODEL = "gpt-5.5-mini"


@dataclass(frozen=True)
class TokenPrice:
    """What one model charges, in your currency per million tokens."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self):
        for name in ("input_per_million", "output_per_million"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a number >= 0, {value!r} given")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000


class OpenAIBackend:
    """Runs one graph node as one chat-completions API call.

    Args:
        model: Model name used for every component; ignored when
            ``model_for`` is supplied.
        model_for: Optional ``Callable[[ComponentMetadata], str]`` that
            maps a component to the model it should run on (e.g. cheap
            models for retrieval tooling, stronger models for
            generation); takes precedence over ``model``.
        prices: Optional ``{model name: TokenPrice}``. When the model a
            call ran on has a price and the response reports token
            usage, the recorded cost is computed from that usage;
            otherwise the component's declared ``cost_per_call`` is used.
        client: Optional pre-built OpenAI client (or any object exposing
            ``chat.completions.create``). Meant for tests; when given,
            the ``openai`` package is never imported.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        model_for: Optional[Callable[[ComponentMetadata], str]] = None,
        client: Optional[Any] = None,
        prices: Optional[Mapping[str, TokenPrice]] = None,
    ):
        self._model = model
        self.model_for = model_for
        self._prices = dict(prices or {})
        for priced_model, price in self._prices.items():
            if not isinstance(price, TokenPrice):
                raise TypeError(
                    f"price for '{priced_model}' must be a TokenPrice, {price!r} given"
                )
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - exercised only without extra
                raise ImportError(
                    "OpenAIBackend requires the optional 'openai' dependency; "
                    "install it with: pip install chatcortex[real]"
                ) from exc
            self._client = openai.OpenAI()

    def _model_for_component(self, metadata: ComponentMetadata) -> str:
        if self.model_for is not None:
            return self.model_for(metadata)
        return self._model

    def __call__(
        self,
        metadata: ComponentMetadata,
        node_input: Any = None,
    ) -> ProbeResult:
        model = self._model_for_component(metadata)
        prompt = str(node_input) if node_input is not None else (
            f"Perform the '{metadata.capabilities[0]}' task as component "
            f"'{metadata.name}'."
        )

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        except Exception as exc:  # network/HTTP/key errors are probe failures
            latency_ms = (time.perf_counter() - started) * 1000.0
            return ProbeResult(
                success=False,
                latency_ms=latency_ms,
                cost=0.0,
                output=f"openai call failed: {type(exc).__name__}: {exc}",
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        output = response.choices[0].message.content
        return ProbeResult(
            success=bool(output),
            latency_ms=latency_ms,
            cost=self._cost(model, response, metadata),
            output=output,
        )

    def _cost(self, model: str, response: Any, metadata: ComponentMetadata) -> float:
        """Cost from reported token usage, or the declared cost without it."""
        price = self._prices.get(model)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if price is None or input_tokens is None or output_tokens is None:
            return metadata.cost_per_call
        return price.cost(input_tokens, output_tokens)
