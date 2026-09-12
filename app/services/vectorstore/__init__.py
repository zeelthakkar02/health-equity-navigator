"""Vector storage abstraction and the in-memory implementation."""

from app.services.vectorstore.base import (
    DimensionMismatchError,
    SearchFilters,
    SearchHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)
from app.services.vectorstore.memory_store import InMemoryVectorStore

__all__ = [
    "DimensionMismatchError",
    "InMemoryVectorStore",
    "SearchFilters",
    "SearchHit",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
]
