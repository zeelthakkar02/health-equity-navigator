"""Google embedding models on Vertex AI.

Same isolation rules as the generation provider: the ``google-genai`` SDK is
imported lazily inside ``__init__``, authentication is Application Default
Credentials only, and every SDK exception is translated into an
:mod:`app.services.embeddings.errors` type.

Documents and queries are embedded with different task types
(``RETRIEVAL_DOCUMENT`` / ``RETRIEVAL_QUERY``), which is what makes an asymmetric
retrieval model work properly.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from app.services.embeddings.base import Embedding, EmbeddingService
from app.services.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingTimeoutError,
    EmbeddingUpstreamError,
)

logger = logging.getLogger(__name__)

DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"


class VertexEmbeddingService(EmbeddingService):
    """Embeds text with a Google embedding model served through Vertex AI."""

    provider_name = "vertex"

    def __init__(
        self,
        *,
        project: str,
        location: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 30.0,
        batch_size: int = 1,
        max_concurrency: int = 4,
    ) -> None:
        if not project:
            raise EmbeddingConfigurationError(
                "GOOGLE_CLOUD_PROJECT must be set to use the Vertex AI embedding provider."
            )

        try:
            from google import genai
            from google.auth import exceptions as google_auth_exceptions
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise EmbeddingConfigurationError(
                "The 'google-genai' package is required for EMBEDDING_PROVIDER=vertex. "
                "Install it with: pip install google-genai"
            ) from exc

        self._types = types
        self.model_name = model
        self.dimensions = dimensions
        self._timeout_seconds = timeout_seconds
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # Credentials resolve on first use, not at construction, so the request
        # path has to recognise auth failures as configuration problems too.
        self._auth_error_types: tuple[type[Exception], ...] = (
            google_auth_exceptions.DefaultCredentialsError,
            google_auth_exceptions.RefreshError,
        )

        try:
            self._client = genai.Client(vertexai=True, project=project, location=location)
        except Exception as exc:  # SDK raises a wide range of auth/transport errors
            raise EmbeddingConfigurationError(
                "Could not initialise the Vertex AI client for embeddings. Confirm the "
                "project and location are correct and that Application Default "
                "Credentials are available (gcloud auth application-default login)."
            ) from exc

        logger.info(
            "Vertex embedding provider ready (project=%s, location=%s, model=%s, dims=%d)",
            project,
            location,
            model,
            dimensions,
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        if not texts:
            return []
        batches = [
            list(texts[start : start + self._batch_size])
            for start in range(0, len(texts), self._batch_size)
        ]
        results = await asyncio.gather(
            *(self._embed_batch(batch, DOCUMENT_TASK_TYPE) for batch in batches)
        )
        return [vector for batch_result in results for vector in batch_result]

    async def embed_query(self, text: str) -> Embedding:
        vectors = await self._embed_batch([text], QUERY_TASK_TYPE)
        return vectors[0]

    async def _embed_batch(self, texts: list[str], task_type: str) -> list[Embedding]:
        config = self._types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimensions,
        )

        async with self._semaphore:
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    response = await self._client.aio.models.embed_content(
                        model=self.model_name,
                        contents=texts,
                        config=config,
                    )
            except TimeoutError as exc:
                raise EmbeddingTimeoutError(
                    f"Vertex AI embeddings did not respond within {self._timeout_seconds:g}s."
                ) from exc
            except self._auth_error_types as exc:
                raise EmbeddingConfigurationError(
                    "Vertex AI credentials are unavailable or could not be refreshed "
                    f"({type(exc).__name__}). Configure Application Default Credentials "
                    "with: gcloud auth application-default login"
                ) from exc
            except Exception as exc:  # never leak SDK exception types past this layer
                logger.error("Vertex AI embed_content failed: %s: %s", type(exc).__name__, exc)
                raise EmbeddingUpstreamError(
                    f"Vertex AI embedding request failed: {type(exc).__name__}"
                ) from exc

        return [_normalise(_values(item), self.dimensions) for item in _embeddings(response)]


def _embeddings(response: Any) -> list[Any]:
    items = getattr(response, "embeddings", None)
    if not items:
        raise EmbeddingUpstreamError("Vertex AI returned no embeddings for this request.")
    return list(items)


def _values(item: Any) -> list[float]:
    values = getattr(item, "values", None)
    if not values:
        raise EmbeddingUpstreamError("Vertex AI returned an empty embedding vector.")
    return [float(value) for value in values]


def _normalise(values: list[float], expected_dimensions: int) -> Embedding:
    """Unit-normalise, which Matryoshka-truncated outputs require."""
    if len(values) != expected_dimensions:
        raise EmbeddingUpstreamError(
            f"Vertex AI returned a {len(values)}-dimension vector, expected {expected_dimensions}."
        )
    norm = sum(value * value for value in values) ** 0.5
    if norm == 0.0:
        raise EmbeddingUpstreamError("Vertex AI returned a zero-length embedding vector.")
    return [value / norm for value in values]
