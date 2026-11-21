"""Keep every answer a pipeline step gives, and reuse it on the next run.

:class:`AnswerLog` is a file with one JSON record per call: what was
asked, of what, when, and what came back. :class:`LoggedBackend` wraps
any backend so that each step is written to the log, and a step that was
already answered is read back from it instead of being called again::

    log = AnswerLog("answers.jsonl")
    backend = LoggedBackend(OpenAIBackend(model="..."), log)
    evaluate_pipelines(pipelines, AgentExecutor(backend=backend), eval_set)

Run the same comparison again and no model is called; add one pipeline
and only its new steps are.

What makes two calls "the same" is the step's *identity* and its input.
The identity says what actually ran: for a model call the endpoint, the
model name and the request options; for your own function its name and a
version you choose. Change the model and the old answers are no longer
used. A Python function cannot be fingerprinted reliably, so when you
change what a function does, change its version too::

    LoggedBackend(backend, log, versions={"retriever": "bm25-v2"})

Only successful answers are reused. A failed call is written to the log,
because a rate limit or an outage is worth knowing about, but the next
run tries it again.

The log is append-only: nothing is ever rewritten, so it doubles as a
record of where each number in a results table came from. To sample the
same step more than once, on different days for instance, give each
sample its own ``label``.
"""

import hashlib
import json
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from ..execution.errors import classify_error
from ..execution.executor import BackendConfigurationError, ExecutionProtocol, ProbeResult
from ..registry.metadata import ComponentMetadata

FORMAT = 1

Identity = Dict[str, Any]


class NotLoggable(TypeError):
    """A value cannot be written to the log as JSON."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise NotLoggable(str(exc)) from exc


def fingerprint(identity: Identity, node_input: Any) -> str:
    """A short, stable name for "this step, asked this".

    Raises:
        NotLoggable: If the identity or the input is not JSON data.
    """
    text = _canonical({"identity": identity, "input": node_input})
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def identify(
    backend: Any,
    metadata: ComponentMetadata,
    versions: Optional[Mapping[str, Any]] = None,
) -> Identity:
    """What would actually run for ``metadata`` on ``backend``.

    Always holds the component name. A backend adds what it knows by
    defining ``identity(metadata) -> dict``; ``versions`` adds your own
    marker for a component, typically for a Python function.
    """
    identity: Identity = {"component": metadata.name}
    describe = getattr(backend, "identity", None)
    if callable(describe):
        extra = describe(metadata)
        if extra:
            identity.update(extra)
    if versions and metadata.name in versions:
        identity["version"] = versions[metadata.name]
    return identity


class AnswerLog:
    """An append-only file of answers, one JSON record per line.

    Args:
        path: Where the log lives. It is created on the first write;
            an existing log is read and extended.

    Raises:
        ValueError: If an existing file holds a line that is not a
            record of this log.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._answers: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._count = 0
        if self.path.exists():
            for record in self._read():
                self._remember(record)

    def _read(self) -> Iterator[Dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}, line {number} is not valid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict) or "key" not in record:
                    raise ValueError(
                        f"{self.path}, line {number} is not an answer record"
                    )
                yield record

    def _remember(self, record: Dict[str, Any]) -> None:
        self._count += 1
        if record.get("success"):
            self._answers[(record["key"], str(record.get("label", "")))] = record

    def __len__(self) -> int:
        """Number of records, failed calls included."""
        return self._count

    def records(self) -> List[Dict[str, Any]]:
        """Every record in the order it was written."""
        if not self.path.exists():
            return []
        return list(self._read())

    def find(self, key: str, label: str = "") -> Optional[Dict[str, Any]]:
        """The latest successful answer for ``key`` under ``label``, if any."""
        return self._answers.get((key, str(label)))

    def append(self, record: Dict[str, Any]) -> None:
        """Write one record.

        Raises:
            NotLoggable: If the record is not JSON data.
        """
        try:
            line = json.dumps(record, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise NotLoggable(str(exc)) from exc
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._remember(record)


class LoggedBackend:
    """Write every step of ``backend`` to a log, and reuse logged answers.

    Args:
        backend: The backend that does the work.
        log: The :class:`AnswerLog` to write to, or a path for one.
        versions: ``{component name: marker}``. Change a marker when you
            change what that component does, so its old answers are not
            reused.
        label: Names this sample. Answers are reused only within the
            same label, so a new label (today's date, "run-2") asks
            every step again and keeps both sets of answers.
        reuse: With ``False`` every step is called and logged, and
            nothing is read back.
        clock: Returns the current time; replace it in tests.

    A step whose input or output is not JSON data (text, numbers, lists,
    dicts) cannot be logged. It still runs, uncached, and a warning says
    so once per component. Tuples are written as lists, so a reused
    answer holds lists where the first run held tuples.

    Attributes:
        calls: Steps passed to the wrapped backend since creation.
        reused: Steps answered from the log since creation.
    """

    def __init__(
        self,
        backend: ExecutionProtocol,
        log: Union[AnswerLog, str, Path],
        versions: Optional[Mapping[str, Any]] = None,
        label: str = "",
        reuse: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        if not callable(backend):
            raise TypeError(f"backend must be callable, {backend!r} given")
        self.backend = backend
        self.log = log if isinstance(log, AnswerLog) else AnswerLog(log)
        self.versions = dict(versions or {})
        self.label = str(label)
        self.reuse = reuse
        self.calls = 0
        self.reused = 0
        self._clock = clock
        self._warned: set = set()

    def identity(self, metadata: ComponentMetadata) -> Identity:
        """What would run for ``metadata``: see :func:`identify`."""
        return identify(self.backend, metadata, self.versions)

    def _warn(self, metadata: ComponentMetadata, what: str, reason: Exception) -> None:
        if metadata.name in self._warned:
            return
        self._warned.add(metadata.name)
        warnings.warn(
            f"component '{metadata.name}' is not logged or reused: its {what} "
            f"is not JSON data ({reason})",
            stacklevel=3,
        )

    def __call__(self, metadata: ComponentMetadata, node_input: Any) -> ProbeResult:
        identity = self.identity(metadata)
        try:
            key = fingerprint(identity, node_input)
        except NotLoggable as exc:
            self._warn(metadata, "input", exc)
            self.calls += 1
            return self.backend(metadata, node_input)

        if self.reuse:
            known = self.log.find(key, self.label)
            if known is not None:
                self.reused += 1
                return ProbeResult(
                    success=True,
                    latency_ms=known["latency_ms"],
                    cost=known["cost"],
                    output=known["output"],
                )

        self.calls += 1
        started = time.perf_counter()
        try:
            probe = self.backend(metadata, node_input)
        except BackendConfigurationError:
            raise  # a setup mistake, not an answer
        except Exception as exc:  # a failed call is still worth logging
            probe = ProbeResult(
                success=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                cost=0.0,
                output=f"backend raised {type(exc).__name__}: {exc}",
                error=classify_error(exc),
            )
        record = {
            "format": FORMAT,
            "key": key,
            "label": self.label,
            "time": self._clock().isoformat(timespec="seconds"),
            "component": metadata.name,
            "identity": identity,
            "input": node_input,
            "success": bool(probe.success),
            "error": getattr(probe, "error", None),
            "latency_ms": probe.latency_ms,
            "cost": probe.cost,
            "output": probe.output,
        }
        try:
            self.log.append(record)
        except NotLoggable as exc:
            self._warn(metadata, "output", exc)
        return probe
