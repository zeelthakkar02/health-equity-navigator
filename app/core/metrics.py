"""Per-request metrics for the Navigator.

Emitted as one structured record per request so retrieval time, model time, and
the decision the service reached are all queryable in production without
reconstructing them from prose log lines.

What is deliberately absent: the community member's message. A query here can
describe someone's health, immigration status, or housing situation. The record
carries its length and the needs detected from it, never its text, unless
``LOG_QUERY_TEXT`` is explicitly turned on for local debugging.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("app.metrics")


@dataclass
class RequestMetrics:
    """Timings and outcome for one Navigator request."""

    request_id: str
    answer_source: str = ""
    escalation_reason: str | None = None
    needs_escalation: bool = False
    retrieval_ms: int = 0
    llm_ms: int = 0
    grounding_ms: int = 0
    total_ms: int = 0
    resources_returned: int = 0
    stated_needs: list[str] = field(default_factory=list)
    query_chars: int = 0
    has_location: bool = False
    language: str = "en"
    provider: str = ""
    model: str = ""
    grounding_issues: list[str] = field(default_factory=list)
    query_text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        if record["query_text"] is None:
            record.pop("query_text")
        return record


MetricsSink = Callable[[RequestMetrics], None]


def log_metrics(metrics: RequestMetrics) -> None:
    """Default sink: one structured log record per request."""
    logger.info(
        "navigator_request source=%s escalation=%s total_ms=%d",
        metrics.answer_source,
        metrics.escalation_reason or "-",
        metrics.total_ms,
        extra={"request_id": metrics.request_id, "metrics": metrics.as_dict()},
    )


class CollectingSink:
    """Keeps metrics in memory. Used by the evaluation harness and tests."""

    def __init__(self) -> None:
        self.records: list[RequestMetrics] = []

    def __call__(self, metrics: RequestMetrics) -> None:
        self.records.append(metrics)

    def clear(self) -> None:
        self.records.clear()
