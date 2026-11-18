"""Run graph nodes as plain Python callables.

:class:`CallableBackend` maps component names to functions and satisfies
:class:`~chatcortex.execution.executor.ExecutionProtocol`, so an
:class:`~chatcortex.execution.executor.AgentExecutor` built with it runs a
synthesized pipeline for real: each function receives the previous step's
output and returns its own::

    backend = CallableBackend({
        "retriever": lambda question: search(question),
        "writer": lambda passages: draft(passages),
    })
    result = AgentExecutor(backend=backend).execute(graph, task_input="...")
    result.output   # what the last step returned

What is measured vs. declared:

- **Latency** is measured around each call.
- **Success** is ``True`` unless the function raises, or returns a
  :class:`StepResult` with ``success=False``.
- **Cost** is the component's declared ``cost_per_call`` unless the
  function returns a :class:`StepResult` carrying its own ``cost``.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

from ..registry.capability_registry import CapabilityRegistry
from ..registry.metadata import ComponentMetadata
from .executor import BackendConfigurationError, ProbeResult


@dataclass(frozen=True)
class StepResult:
    """What a component function may return when it has more to report.

    Returning a bare value is the same as ``StepResult(output=value)``.
    Use this to mark a step as failed without raising, or to report what
    the call actually cost.
    """

    output: Any
    success: bool = True
    cost: Optional[float] = None


class CallableBackend:
    """Execution backend that runs one Python callable per component.

    Args:
        implementations: ``{component name: callable}``. Each callable
            takes one argument, the previous step's output (the
            ``task_input`` for the first step), and returns its output or
            a :class:`StepResult`.

    Raises:
        TypeError: If any value in ``implementations`` is not callable.
    """

    def __init__(self, implementations: Mapping[str, Callable[[Any], Any]]):
        for name, implementation in implementations.items():
            if not callable(implementation):
                raise TypeError(
                    f"implementation for '{name}' must be callable, "
                    f"{implementation!r} given"
                )
        self._implementations = dict(implementations)

    def missing(self, registry: CapabilityRegistry) -> List[str]:
        """Names of registry components that have no implementation.

        Call this before synthesizing: a pipeline that includes one of
        these components cannot be executed by this backend.
        """
        return [
            metadata.name
            for metadata in registry.list_all()
            if metadata.name not in self._implementations
        ]

    def __call__(self, metadata: ComponentMetadata, node_input: Any) -> ProbeResult:
        implementation = self._implementations.get(metadata.name)
        if implementation is None:
            raise BackendConfigurationError(
                f"no implementation registered for component '{metadata.name}'; "
                "pass one to CallableBackend, or check CallableBackend.missing(registry)"
            )

        started = time.perf_counter()
        try:
            returned = implementation(node_input)
        except Exception as exc:  # the component failed; that is a result
            return ProbeResult(
                success=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                cost=metadata.cost_per_call,
                output=f"{metadata.name} raised {type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - started) * 1000.0

        step = returned if isinstance(returned, StepResult) else StepResult(returned)
        return ProbeResult(
            success=bool(step.success),
            latency_ms=latency_ms,
            cost=metadata.cost_per_call if step.cost is None else step.cost,
            output=step.output,
        )
