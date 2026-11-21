"""Run graph nodes as plain Python callables.

:class:`CallableBackend` maps component names to functions and satisfies
:class:`~frontierkit.execution.executor.ExecutionProtocol`, so an
:class:`~frontierkit.execution.executor.AgentExecutor` built with it runs a
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
  function returns a :class:`StepResult` carrying its own ``cost``. A
  function that raises records zero cost, as a failed API call does; to
  report what a failure cost, return
  ``StepResult(output=None, success=False, cost=...)`` instead of raising.
- **Error kind.** A function that raises is recorded with the kind of its
  exception (see :mod:`frontierkit.execution.errors`); one that returns a
  failed :class:`StepResult` is ``"step_failed"`` unless it names a kind.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

from ..registry.capability_registry import CapabilityRegistry
from ..registry.metadata import ComponentMetadata
from .errors import STEP_FAILED, classify_error
from .executor import BackendConfigurationError, ProbeResult


@dataclass(frozen=True)
class StepResult:
    """What a component function may return when it has more to report.

    Returning a bare value is the same as ``StepResult(output=value)``.
    Use this to mark a step as failed without raising, or to report what
    the call actually cost. ``error`` names what went wrong for a failed
    step (one of the kinds in :mod:`frontierkit.execution.errors`, or
    your own word); it is ignored when ``success`` is true.
    """

    output: Any
    success: bool = True
    cost: Optional[float] = None
    error: Optional[str] = None


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
                cost=0.0,
                output=f"{metadata.name} raised {type(exc).__name__}: {exc}",
                error=classify_error(exc),
            )
        latency_ms = (time.perf_counter() - started) * 1000.0

        step = returned if isinstance(returned, StepResult) else StepResult(returned)
        succeeded = bool(step.success)
        return ProbeResult(
            success=succeeded,
            latency_ms=latency_ms,
            cost=metadata.cost_per_call if step.cost is None else step.cost,
            output=step.output,
            error=None if succeeded else (step.error or STEP_FAILED),
        )
