"""Name what went wrong when a step fails.

A step can fail because the answer never arrived (a rate limit, an
outage, a model that no longer exists) or because the step itself said
so. Those are different problems: the first says something about the
service, the second about the pipeline. Every failed step is recorded
with one of the kinds below, so a comparison can tell them apart.

:func:`classify_error` maps an exception to a kind. It looks at the HTTP
status carried by the exception when there is one, then at the exception
type and its name, and needs no vendor SDK to be installed.
"""

import socket
from typing import Optional

RATE_LIMITED = "rate_limited"
"""The service refused the call because too many were made (HTTP 429)."""

TIMEOUT = "timeout"
"""The call did not finish in time."""

UNAVAILABLE = "unavailable"
"""The service could not be reached or answered with a server error."""

NOT_FOUND = "not_found"
"""The model or address does not exist: retired, renamed or misspelt."""

AUTH = "auth"
"""The key was missing, wrong or not allowed to use the model."""

EMPTY_OUTPUT = "empty_output"
"""The call succeeded but returned nothing."""

STEP_FAILED = "step_failed"
"""The step ran and reported failure itself, without raising."""

ERROR = "error"
"""Any other exception."""

SERVICE_ERRORS = frozenset({RATE_LIMITED, TIMEOUT, UNAVAILABLE, NOT_FOUND, AUTH})
"""Kinds that describe the service, not the quality of an answer."""


def _status_code(exc: BaseException) -> Optional[int]:
    for holder in (exc, getattr(exc, "response", None)):
        for attribute in ("status_code", "status", "code"):
            value = getattr(holder, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def classify_error(exc: BaseException) -> str:
    """The kind of failure an exception stands for.

    Returns one of the constants of this module. An exception that
    carries no HTTP status and matches no known type is ``"error"``.
    """
    status = _status_code(exc)
    if status == 429:
        return RATE_LIMITED
    if status in (408, 504):
        return TIMEOUT
    if status in (401, 403):
        return AUTH
    if status in (404, 410):
        return NOT_FOUND
    if status is not None and status >= 500:
        return UNAVAILABLE

    name = type(exc).__name__.lower()
    # Timeouts first: SDKs often derive them from their connection error.
    if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in name:
        return TIMEOUT
    if "ratelimit" in name:
        return RATE_LIMITED
    if "authentication" in name or "permissiondenied" in name:
        return AUTH
    if "notfound" in name:
        return NOT_FOUND
    if (
        isinstance(exc, ConnectionError)
        or "connection" in name
        or "unavailable" in name
        or "internalserver" in name
    ):
        return UNAVAILABLE
    return ERROR
