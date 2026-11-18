"""Send different components to different backends.

A single search can mix components that live in different places: one
model at one vendor, another at a second vendor, a retriever that is a
local Python function. :class:`RoutedBackend` picks the backend for each
component by name::

    backend = RoutedBackend(
        {
            "writer_vendor_a": OpenAIBackend(model="...", base_url=URL_A, api_key=KEY_A),
            "writer_vendor_b": OpenAIBackend(model="...", base_url=URL_B, api_key=KEY_B),
        },
        default=CallableBackend({"retriever": search}),
    )
"""

from typing import Any, List, Mapping, Optional

from ..registry.capability_registry import CapabilityRegistry
from ..registry.metadata import ComponentMetadata
from .executor import BackendConfigurationError, ExecutionProtocol, ProbeResult


class RoutedBackend:
    """Execution backend that delegates each component to its own backend.

    Args:
        routes: ``{component name: backend}``.
        default: Backend for components not named in ``routes``. Without
            one, running an unrouted component raises
            :class:`~chatcortex.execution.executor.BackendConfigurationError`.

    Raises:
        TypeError: If a route or the default is not callable.
    """

    def __init__(
        self,
        routes: Mapping[str, ExecutionProtocol],
        default: Optional[ExecutionProtocol] = None,
    ):
        for name, backend in routes.items():
            if not callable(backend):
                raise TypeError(
                    f"backend for '{name}' must be callable, {backend!r} given"
                )
        if default is not None and not callable(default):
            raise TypeError(f"default backend must be callable, {default!r} given")
        self._routes = dict(routes)
        self._default = default

    def unrouted(self, registry: CapabilityRegistry) -> List[str]:
        """Registry components with no route; empty when a default is set."""
        if self._default is not None:
            return []
        return [
            metadata.name
            for metadata in registry.list_all()
            if metadata.name not in self._routes
        ]

    def __call__(self, metadata: ComponentMetadata, node_input: Any) -> ProbeResult:
        backend = self._routes.get(metadata.name, self._default)
        if backend is None:
            raise BackendConfigurationError(
                f"no backend routed for component '{metadata.name}' and no default set"
            )
        return backend(metadata, node_input)
