"""Embedding provider selection.

``EMBEDDING_PROVIDER`` is the only switch, and it is independent of
``LLM_PROVIDER``: retrieval can run offline while generation uses Vertex, or the
other way round.
"""

from __future__ import annotations

import logging

from app.core.config import EmbeddingProvider, Settings
from app.services.embeddings.base import EmbeddingService
from app.services.embeddings.errors import EmbeddingConfigurationError
from app.services.embeddings.hashing_provider import HashingEmbeddingService
from app.services.embeddings.vertex_provider import VertexEmbeddingService

logger = logging.getLogger(__name__)


def build_embedding_service(settings: Settings) -> EmbeddingService:
    """Construct the configured :class:`EmbeddingService`.

    Raises:
        EmbeddingConfigurationError: the provider is unknown or unusable.
    """
    match settings.embedding_provider:
        case EmbeddingProvider.HASHING:
            logger.info(
                "Embedding provider: hashing (offline, lexical, %d dims)",
                settings.embedding_dimensions,
            )
            return HashingEmbeddingService(dimensions=settings.embedding_dimensions)
        case EmbeddingProvider.VERTEX:
            return VertexEmbeddingService(
                project=settings.google_cloud_project or "",
                location=settings.google_cloud_location,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                timeout_seconds=settings.llm_timeout_seconds,
                batch_size=settings.embedding_batch_size,
                max_concurrency=settings.embedding_max_concurrency,
            )
        case _:  # pragma: no cover - guarded by the Settings enum
            raise EmbeddingConfigurationError(
                f"Unsupported embedding provider: {settings.embedding_provider!r}"
            )
