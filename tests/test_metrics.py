"""Per-request metrics and structured logging."""

from __future__ import annotations

import json
import logging

import pytest

from app.core.config import Settings
from app.core.logging import JsonFormatter, configure_logging
from app.core.metrics import CollectingSink, RequestMetrics, log_metrics
from app.schemas.navigator import NavigatorQueryRequest
from app.services.navigator_service import NavigatorService
from tests.test_navigator_service import FOOD, FakeLLM, FakeRetriever


def _service(settings: Settings, sink: CollectingSink) -> NavigatorService:
    return NavigatorService(
        FakeLLM(text="Oakland Community Food Pantry [1] can help."),
        settings,
        retriever=FakeRetriever(resources=[FOOD]),  # type: ignore[arg-type]
        metrics_sink=sink,
    )


async def test_every_request_emits_one_metrics_record(settings: Settings) -> None:
    sink = CollectingSink()

    await _service(settings, sink).answer(NavigatorQueryRequest(query="I need food help."))

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.answer_source == "generated"
    assert record.resources_returned == 1
    assert record.total_ms >= 0
    assert record.stated_needs == ["food_assistance"]


async def test_timings_are_broken_out_by_stage(settings: Settings) -> None:
    sink = CollectingSink()

    await _service(settings, sink).answer(NavigatorQueryRequest(query="I need food help."))

    record = sink.records[0]
    assert record.retrieval_ms >= 0
    assert record.llm_ms >= 0
    assert record.grounding_ms >= 0
    assert record.total_ms >= record.retrieval_ms


async def test_the_query_text_is_not_recorded_by_default(settings: Settings) -> None:
    """A message can describe someone's health or immigration status."""
    sink = CollectingSink()
    query = "I need food help for my family."

    await _service(settings, sink).answer(NavigatorQueryRequest(query=query))

    record = sink.records[0]
    assert record.query_text is None
    assert "query_text" not in record.as_dict()
    assert query not in json.dumps(record.as_dict())
    assert record.query_chars == len(query)


async def test_query_text_can_be_opted_into_for_debugging(settings: Settings) -> None:
    sink = CollectingSink()
    verbose = settings.model_copy(update={"log_query_text": True})

    await _service(verbose, sink).answer(NavigatorQueryRequest(query="I need food help."))

    assert sink.records[0].query_text == "I need food help."


async def test_escalation_is_recorded(settings: Settings) -> None:
    sink = CollectingSink()
    service = NavigatorService(
        FakeLLM(),
        settings,
        retriever=FakeRetriever(resources=[]),  # type: ignore[arg-type]
        metrics_sink=sink,
    )

    await service.answer(NavigatorQueryRequest(query="Something unrelated entirely."))

    record = sink.records[0]
    assert record.needs_escalation is True
    assert record.escalation_reason == "no_verified_resources"


async def test_a_crisis_message_is_recorded_without_its_text(settings: Settings) -> None:
    sink = CollectingSink()

    await _service(settings, sink).answer(NavigatorQueryRequest(query="I want to kill myself."))

    record = sink.records[0]
    assert record.answer_source == "safety_notice"
    assert record.escalation_reason == "possible_crisis"
    assert "kill myself" not in json.dumps(record.as_dict())


# --- structured logging ----------------------------------------------------


def test_json_logs_carry_extras_as_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello", None, None)
    record.request_id = "req-1"
    record.metrics = {"total_ms": 42}

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "hello"
    assert payload["request_id"] == "req-1"
    assert payload["metrics"] == {"total_ms": 42}
    assert payload["level"] == "INFO"


def test_a_record_without_a_request_id_still_formats() -> None:
    configure_logging("INFO", "text")
    logging.getLogger("app.test").info("no request id here")


def test_the_default_sink_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.metrics"):
        log_metrics(RequestMetrics(request_id="req-2", answer_source="generated", total_ms=5))

    assert "navigator_request" in caplog.text
