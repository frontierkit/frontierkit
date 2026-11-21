import random
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol

from frontierkit.graph.agent_graph import AgentGraph
from frontierkit.registry.capability_registry import CapabilityRegistry
from frontierkit.registry.metadata import ComponentMetadata
from frontierkit.telemetry.logger import TelemetryLogger

from .errors import STEP_FAILED, classify_error

ExecutionMode = Literal["deterministic", "probabilistic"]


class ExecutionProtocol(Protocol):
    """Optional real-execution backend protocol (see ``AgentExecutor``).

    Any callable with the signature ``backend(metadata, node_input) ->
    ProbeResult`` satisfies this protocol; a class only needs a matching
    ``__call__``. Implementations own the actual invocation (HTTP, SDK,
    stub); the executor only consumes the returned ProbeResult.
    """

    @dataclass(frozen=True)
    class ProbeResult:
        """Outcome of one real backend invocation of a graph node.

        A execution backend returns one of these per component call; the
        values are recorded verbatim in the execution's TelemetryLogger,
        so they should be *measured* values, not estimates.
        """

        success: bool
        latency_ms: float
        cost: float
        output: Any
        error: Optional[str] = None
        """For a failed call, what went wrong: one of the kinds in
        :mod:`frontierkit.execution.errors`. ``None`` on success."""

    def __call__(
        self,
        metadata: ComponentMetadata,
        node_input: Any,
    ) -> ProbeResult:  # pragma: no cover - protocol definition only
        ...


# Flat alias so the pre-nesting import path (`from ...executor import
# ProbeResult`) keeps working alongside ``ExecutionProtocol.ProbeResult``.
ProbeResult = ExecutionProtocol.ProbeResult


class BackendConfigurationError(Exception):
    """A backend cannot run a component at all (not: the component failed).

    Backend exceptions normally become failed probes, because a component
    that errors is an execution result worth recording. A backend raises
    this instead when the setup is wrong, for example no implementation
    exists for a component, and the executor lets it propagate so the
    mistake is not recorded as component unreliability.
    """


class ExecutionResult:
    """Outcome of executing one graph.

    Attributes:
        telemetry: The per-step records.
        output: What the last step returned when every step succeeded;
            ``None`` if the chain stopped on a failure, and always
            ``None`` for the simulator, which produces no outputs.
        success: ``True`` when every step succeeded.
        error: ``None`` when every step succeeded; otherwise what went
            wrong in the step that stopped the chain, one of the kinds
            in :mod:`frontierkit.execution.errors`.
    """

    def __init__(
        self,
        telemetry: TelemetryLogger,
        output: Any = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        self.telemetry = telemetry
        self.output = output
        self.success = success
        self.error = error

    def summary(self):
        return self.telemetry.summary()


class AgentExecutor:
    """
    Simulated execution engine for AgentGraph
    Supports deterministic and probabilistic modes,
    plus an optional real-execution backend.

    With ``backend=None`` (default)
    behavior is identical to the pure simulator. With a backend set,
    each node is invoked through ``backend(metadata, node_input) ->
    ProbeResult`` and the *measured* latency/cost/success from the
    ProbeResult (never the metadata estimates) are recorded in the
    telemetry; ``mode`` is ignored because the backend decides success.
    The chain stops on the first failure, exactly like the simulator.
    A backend exception is recorded as a failed probe (zero cost, the
    time the call took, and the kind of error) and also stops the chain,
    so partial telemetry survives the error.

    Hidden ground truth: by default the simulator draws outcomes from the
    metadata carried by the graph, i.e. from the *declared* values, so a
    calibration loop fed by it can only rediscover its own priors plus
    sampling noise. Pass ``ground_truth`` (a registry holding each
    component's *true* cost, latency and reliability) and the simulator
    draws from those instead, looked up by component name. The synthesizer
    and calibration never see that registry, which makes the simulator a
    real test of whether a loop converges on values it was not told.
    Components absent from ``ground_truth`` behave as declared.
    ``ground_truth`` cannot be combined with ``backend``: a backend
    already is the source of truth.
    """
    def __init__(
            self,
            mode: ExecutionMode = "deterministic",
            seed: Optional[int] = None,
            backend: Optional[ExecutionProtocol] = None,
            ground_truth: Optional[CapabilityRegistry] = None,
        ):
        if backend is not None and ground_truth is not None:
            raise ValueError(
                "ground_truth only applies to the simulator; a backend "
                "is its own source of truth, so pass one or the other"
            )
        self.mode = mode
        self.backend = backend
        self.ground_truth = ground_truth
        # local random generator - NO RNG pollution, no side-effects, clean isolation
        self.random = random.Random(seed)

    def execute(self, graph: AgentGraph, task_input: Any = None) -> ExecutionResult:
        """Run ``graph`` once.

        Args:
            graph: The pipeline to execute.
            task_input: Handed to the first step's backend call; each
                later step receives the previous step's output. Ignored
                by the simulator.
        """
        telemetry = TelemetryLogger()

        node_input: Any = task_input
        succeeded = True
        failure: Optional[str] = None
        for node_id in graph.get_execution_order():
            metadata = graph.get_metadata(node_id)
            latency_ms, cost, success, output, error = self._evaluate_node(
                metadata, node_input
            )
            if not success and error is None:
                error = STEP_FAILED

            # Shared semantics for both paths: record the probe, then
            # stop the chain on the first failure.
            telemetry.log(
                component_name=node_id,
                latency_ms=latency_ms,
                cost=cost,
                success=success,
                error=None if success else error,
            )

            if not success:
                succeeded = False
                failure = error
                break # Stop pipeline on failure
            node_input = output

        return ExecutionResult(
            telemetry=telemetry,
            output=node_input if succeeded and self.backend is not None else None,
            success=succeeded,
            error=failure,
        )

    def _evaluate_node(self, metadata: ComponentMetadata, node_input: Any):
        """Run one node through the backend (if set) or the simulator.

        Returns ``(latency_ms, cost, success, output, error)``; the shared
        log-then-stop-on-failure tail lives in ``execute``.
        """
        if self.backend is not None:
            probe = self._invoke_backend(metadata, node_input)
            return (
                probe.latency_ms,
                probe.cost,
                probe.success,
                probe.output,
                getattr(probe, "error", None),
            )

        # What the component really does: the hidden truth when one is
        # supplied, otherwise the declared metadata itself.
        actual = self._true_metadata(metadata)

        # Simulated latency
        simulated_latency = actual.avg_latency_ms

        # Optional: actually sleep (disable if too slow)
        # time.sleep(simulated_latency / 1000)

        # Determine success
        if self.mode == "deterministic":
            success = True
        elif self.mode == "probabilistic":
            success = self.random.random() <= actual.reliability_score
        else:
            raise ValueError("Invalid execution mode")

        return simulated_latency, actual.cost_per_call, success, None, None

    def _true_metadata(self, declared: ComponentMetadata) -> ComponentMetadata:
        """The metadata the simulator samples from for one component."""
        if self.ground_truth is None:
            return declared
        try:
            return self.ground_truth.get(declared.name)
        except KeyError:
            return declared

    def _invoke_backend(
        self, metadata: ComponentMetadata, node_input: Any
    ) -> ProbeResult:
        """Call the backend once, turning exceptions into failed probes."""
        started = time.perf_counter()
        try:
            return self.backend(metadata, node_input)
        except BackendConfigurationError:
            raise  # a setup mistake, not a component outcome
        except Exception as exc:  # backend failures are execution results
            return ProbeResult(
                success=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                cost=0.0,
                output=f"backend raised {type(exc).__name__}: {exc}",
                error=classify_error(exc),
            )
