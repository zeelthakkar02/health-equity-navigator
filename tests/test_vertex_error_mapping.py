"""How the Vertex provider translates SDK failures.

Credential problems surface on the first request rather than at client
construction, so they must be recognised in ``generate`` and reported as
configuration errors (HTTP 503), not upstream errors (HTTP 502).
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services.llm.base import LLMRequest
from app.services.llm.errors import LLMConfigurationError, LLMUpstreamError
from app.services.llm.vertex_provider import VertexGeminiService

pytest.importorskip("google.genai", reason="the Vertex SDK is optional for offline runs")


class _FakeModels:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def generate_content(self, **_: Any) -> Any:
        raise self._error


class _FakeAio:
    def __init__(self, error: Exception) -> None:
        self.models = _FakeModels(error)


class _FakeClient:
    """Stands in for genai.Client so no network call or credential is needed."""

    def __init__(self, error: Exception) -> None:
        self.aio = _FakeAio(error)


def _service(monkeypatch: pytest.MonkeyPatch, error: Exception) -> VertexGeminiService:
    from google import genai

    monkeypatch.setattr(genai, "Client", lambda **_: _FakeClient(error))
    return VertexGeminiService(
        project="example-gcp-project", location="global", model="gemini-3.8-flash"
    )


async def test_missing_credentials_map_to_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.auth.exceptions import DefaultCredentialsError

    service = _service(monkeypatch, DefaultCredentialsError("no ADC"))

    with pytest.raises(LLMConfigurationError, match="application-default login"):
        await service.generate(LLMRequest(prompt="ping"))


async def test_expired_credentials_map_to_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.auth.exceptions import RefreshError

    service = _service(monkeypatch, RefreshError("token refresh failed"))

    with pytest.raises(LLMConfigurationError, match="RefreshError"):
        await service.generate(LLMRequest(prompt="ping"))


async def test_other_sdk_failures_stay_upstream_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(monkeypatch, RuntimeError("backend unavailable"))

    with pytest.raises(LLMUpstreamError, match="RuntimeError"):
        await service.generate(LLMRequest(prompt="ping"))


async def test_sdk_error_details_are_not_leaked_to_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the exception type is surfaced; raw SDK text stays in the cause."""
    service = _service(monkeypatch, RuntimeError("internal trace with request payload"))

    with pytest.raises(LLMUpstreamError) as caught:
        await service.generate(LLMRequest(prompt="ping"))

    assert "internal trace" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
