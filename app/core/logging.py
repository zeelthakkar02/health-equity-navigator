"""Logging setup.

Two formats: readable text for a terminal, and one JSON object per line for a
log pipeline. JSON mode carries the per-request metrics dictionary as structured
fields rather than embedding them in a message string.

The JSON shape follows Google Cloud Logging's structured-log convention, so a
container on Cloud Run gets correct log levels for free: it reads ``severity``
(not ``level``), and Python's level names are already the values it expects.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s"

# Attributes LogRecord always carries; anything else was attached by the caller.
_RESERVED = frozenset(
    """
    args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName
    relativeCreated stack_info thread threadName taskName
    """.split()
)


class RequestIdFilter(logging.Filter):
    """Guarantee ``request_id`` exists so the format string never explodes."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with caller-supplied extras merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            # Cloud Logging keys off "severity"; DEBUG/INFO/WARNING/ERROR/CRITICAL
            # are exactly the values it recognises.
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", log_format: str = "text") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if log_format == "json" else logging.Formatter(_TEXT_FORMAT)
    )
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn installs its own handlers; let records bubble up to ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
