"""Provider selection via LLM_PROVIDER."""

from __future__ import annotations

import pytest

from app.core.config import LLMProvider, Settings
from app.services.llm.errors import LLMConfigurationError
from app.services.llm.factory import build_llm_service
from app.services.llm.stub_provider import StubLLMService


def test_stub_is_the_default(settings: Settings) -> None:
    service = build_llm_service(settings)

    assert isinstance(service, StubLLMService)
    assert service.describe() == {"provider": "stub", "model": "stub-navigator-v1"}


def test_vertex_without_a_project_raises_a_configuration_error(settings: Settings) -> None:
    # model_copy bypasses the Settings validator so the factory's own guard is
    # what gets exercised here.
    broken = settings.model_copy(
        update={"llm_provider": LLMProvider.VERTEX, "google_cloud_project": None}
    )

    with pytest.raises(LLMConfigurationError, match="GOOGLE_CLOUD_PROJECT"):
        build_llm_service(broken)


async def test_stub_service_closes_cleanly(settings: Settings) -> None:
    service = build_llm_service(settings)
    assert await service.aclose() is None
