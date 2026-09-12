"""The VectorStore abstraction.

The store is deliberately domain-agnostic: it knows about ids, vectors, flat
filterable metadata, and an opaque payload it hands back untouched. It does not
know what a Resource is. That is what lets the in-memory implementation be
swapped for Vertex AI Vector Search, pgvector, or Firestore without the
retrieval layer changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class VectorStoreError(Exception):
    """Base class for vector store failures."""

    code = "vector_store_error"


class DimensionMismatchError(VectorStoreError):
    """A vector's width does not match what the store already holds."""

    code = "vector_dimension_mismatch"


@dataclass(frozen=True)
class VectorRecord:
    """One indexed item."""

    id: str
    embedding: Sequence[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchFilters:
    """Metadata constraints applied before scoring.

    ``any_of`` maps a metadata field to the values that satisfy it. A record
    matches when its value for that field — a scalar or a list — overlaps the
    allowed set. All named fields must match.
    """

    any_of: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.any_of


@dataclass(frozen=True)
class SearchHit:
    """A scored record returned by :meth:`VectorStore.search`."""

    id: str
    score: float
    metadata: Mapping[str, Any]
    payload: Mapping[str, Any]


class VectorStore(ABC):
    """Stores vectors and returns the nearest ones by cosine similarity."""

    backend_name: str = "unknown"

    @abstractmethod
    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace records by id. Returns how many were written."""

    @abstractmethod
    async def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        """Return up to ``top_k`` hits, most similar first."""

    @abstractmethod
    async def count(self) -> int:
        """How many records are indexed."""

    @abstractmethod
    async def clear(self) -> None:
        """Drop every record."""

    async def aclose(self) -> None:
        """Release backend resources. Overridden when a client needs teardown."""
        return None

    def describe(self) -> dict[str, str]:
        return {"backend": self.backend_name}
