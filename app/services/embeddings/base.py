"""The EmbeddingService abstraction every embedding provider implements.

Documents and queries are embedded through separate methods on purpose. Retrieval
models are asymmetric: the same text embedded as a document and as a query should
produce different vectors, because a question and the passage answering it do not
look alike. Vertex expresses this through task types; the offline provider keeps
the same shape so swapping providers changes nothing upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

Embedding = list[float]


class EmbeddingService(ABC):
    """Interface for turning text into vectors."""

    provider_name: str = "unknown"
    model_name: str = "unknown"
    dimensions: int = 0

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed resource documents for storage. Order matches ``texts``."""

    @abstractmethod
    async def embed_query(self, text: str) -> Embedding:
        """Embed a user's need for search."""

    async def aclose(self) -> None:
        """Release provider resources. Overridden when a client needs teardown."""
        return None

    def describe(self) -> dict[str, str | int]:
        """Non-sensitive provider description, safe for logs and CLI output."""
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "dimensions": self.dimensions,
        }
