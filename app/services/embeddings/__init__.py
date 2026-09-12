"""Embedding abstraction layer.

Importing this package must never import a cloud SDK — the Vertex provider
imports ``google-genai`` lazily so retrieval runs offline by default.
"""

from app.services.embeddings.base import Embedding, EmbeddingService
from app.services.embeddings.errors import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingTimeoutError,
    EmbeddingUpstreamError,
)
from app.services.embeddings.factory import build_embedding_service

__all__ = [
    "Embedding",
    "EmbeddingConfigurationError",
    "EmbeddingError",
    "EmbeddingService",
    "EmbeddingTimeoutError",
    "EmbeddingUpstreamError",
    "build_embedding_service",
]
