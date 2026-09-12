"""Google Gemini via Vertex AI.

Isolation rules for this module:

* The ``google-genai`` SDK is imported lazily inside ``__init__``. Importing this
  module — which the factory does at import time — pulls in no cloud SDK, so the
  service starts and the tests run without Vertex installed or configured.
* Authentication is Application Default Credentials only. No key, token, or
  service-account path is read, stored, or logged here.
* Every SDK exception is translated into an :mod:`app.services.llm.errors` type.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services.llm.base import LLMRequest, LLMResponse, LLMService
from app.services.llm.errors import LLMConfigurationError, LLMTimeoutError, LLMUpstreamError

logger = logging.getLogger(__name__)


class VertexGeminiService(LLMService):
    """Calls Gemini on Vertex AI through the google-genai SDK."""

    provider_name = "vertex"

    def __init__(
        self,
        *,
        project: str,
        location: str,
        model: str,
        timeout_seconds: float = 30.0,
        thinking_level: str | None = None,
    ) -> None:
        if not project:
            raise LLMConfigurationError(
                "GOOGLE_CLOUD_PROJECT must be set to use the Vertex AI provider."
            )

        try:
            from google import genai
            from google.auth import exceptions as google_auth_exceptions
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise LLMConfigurationError(
                "The 'google-genai' package is required for LLM_PROVIDER=vertex. "
                "Install it with: pip install google-genai"
            ) from exc

        self._types = types
        self.model_name = model
        self._timeout_seconds = timeout_seconds
        self._thinking_level = thinking_level

        # The SDK resolves credentials lazily on the first request rather than at
        # client construction, so generate() has to recognise auth failures too and
        # report them as configuration problems (503) instead of upstream ones (502).
        self._auth_error_types: tuple[type[Exception], ...] = (
            google_auth_exceptions.DefaultCredentialsError,
            google_auth_exceptions.RefreshError,
        )

        try:
            # vertexai=True routes through Vertex AI and uses ADC for auth.
            self._client = genai.Client(vertexai=True, project=project, location=location)
        except Exception as exc:  # SDK raises a wide range of auth/transport errors
            raise LLMConfigurationError(
                "Could not initialise the Vertex AI client. Confirm the project and "
                "location are correct and that Application Default Credentials are "
                "available (gcloud auth application-default login)."
            ) from exc

        logger.info(
            "Vertex AI provider ready (project=%s, location=%s, model=%s)",
            project,
            location,
            model,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Only the system instruction and the output cap are sent. The legacy
        # sampling knobs (temperature, top_p, top_k) are deliberately omitted for
        # Gemini 3.x, and thinking_level is left at its default. LLMRequest still
        # carries `temperature` because other providers honour it.
        config = self._types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            max_output_tokens=request.max_output_tokens,
            thinking_config=self._thinking_config(),
        )

        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._client.aio.models.generate_content(
                    model=self.model_name,
                    contents=request.prompt,
                    config=config,
                )
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"Vertex AI did not respond within {self._timeout_seconds:g}s."
            ) from exc
        except self._auth_error_types as exc:
            raise LLMConfigurationError(
                "Vertex AI credentials are unavailable or could not be refreshed "
                f"({type(exc).__name__}). Configure Application Default Credentials "
                "with: gcloud auth application-default login"
            ) from exc
        except Exception as exc:  # never leak SDK exception types past this layer
            # The detail stays server-side: callers get a generic message, but an
            # operator needs the real error to diagnose a quota or safety block.
            logger.error("Vertex AI generate_content failed: %s: %s", type(exc).__name__, exc)
            raise LLMUpstreamError(f"Vertex AI request failed: {type(exc).__name__}") from exc

        text = getattr(response, "text", None)
        if not text:
            # Typically a safety block or an empty candidate list.
            raise LLMUpstreamError("Vertex AI returned no usable text for this request.")

        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            finish_reason=_finish_reason(response),
            usage=_usage(response),
        )

    def _thinking_config(self) -> Any:
        """Only sent when configured; otherwise the model's own default applies."""
        if not self._thinking_level:
            return None
        return self._types.ThinkingConfig(thinking_level=self._thinking_level)


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return str(reason) if reason is not None else None


def _usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return {}
    fields = {
        "prompt_tokens": "prompt_token_count",
        "completion_tokens": "candidates_token_count",
        "total_tokens": "total_token_count",
    }
    usage: dict[str, int] = {}
    for key, attribute in fields.items():
        value = getattr(metadata, attribute, None)
        if isinstance(value, int):
            usage[key] = value
    return usage
