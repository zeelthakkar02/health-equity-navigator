"""Provider selection.

``LLM_PROVIDER`` is the only switch. Swapping providers must never require a
code change anywhere else in the service.
"""

from __future__ import annotations

import logging

from app.core.config import LLMProvider, Settings
from app.services.llm.base import LLMService
from app.services.llm.errors import LLMConfigurationError
from app.services.llm.stub_provider import StubLLMService
from app.services.llm.vertex_provider import VertexGeminiService

logger = logging.getLogger(__name__)


def build_llm_service(settings: Settings) -> LLMService:
    """Construct the configured :class:`LLMService`.

    Raises:
        LLMConfigurationError: the provider is unknown or cannot be constructed.
    """
    match settings.llm_provider:
        case LLMProvider.STUB:
            logger.info("LLM provider: stub (offline, no cloud calls)")
            return StubLLMService()
        case LLMProvider.VERTEX:
            return VertexGeminiService(
                project=settings.google_cloud_project or "",
                location=settings.google_cloud_location,
                model=settings.gemini_model,
                timeout_seconds=settings.llm_timeout_seconds,
                thinking_level=settings.gemini_thinking_level,
            )
        case _:  # pragma: no cover - guarded by the Settings enum
            raise LLMConfigurationError(f"Unsupported LLM provider: {settings.llm_provider!r}")
