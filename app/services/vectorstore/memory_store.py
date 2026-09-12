"""In-memory brute-force vector store.

Exact cosine similarity over a NumPy matrix. For a verified resource directory —
tens of records now, realistically thousands later — an exhaustive scan is
sub-millisecond and, unlike an approximate index, never silently drops a relevant
result. Vectors are unit-normalised on write, so cosine similarity is a dot
product.

State lives in the process and disappears on restart; ingestion runs at startup
or from the CLI. A persistent backend slots in behind the same ABC.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from app.services.vectorstore.base import (
    DimensionMismatchError,
    SearchFilters,
    SearchHit,
    VectorRecord,
    VectorStore,
)

logger = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStore):
    """Exact nearest-neighbour search over an in-process matrix."""

    backend_name = "in-memory"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._index_by_id: dict[str, int] = {}
        self._metadata: list[Mapping[str, Any]] = []
        self._payloads: list[Mapping[str, Any]] = []
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int | None:
        """Vector width the store is holding, or None while empty."""
        return self._dimensions

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        for record in records:
            vector = self._prepare(record)
            existing = self._index_by_id.get(record.id)
            if existing is None:
                self._index_by_id[record.id] = len(self._ids)
                self._ids.append(record.id)
                self._metadata.append(dict(record.metadata))
                self._payloads.append(dict(record.payload))
                self._vectors.append(vector)
            else:
                self._metadata[existing] = dict(record.metadata)
                self._payloads[existing] = dict(record.payload)
                self._vectors[existing] = vector

        if records:
            self._matrix = None  # rebuilt lazily on the next search
        return len(records)

    async def search(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._ids:
            return []

        query = np.asarray(embedding, dtype=np.float64)
        if self._dimensions is not None and query.shape[0] != self._dimensions:
            raise DimensionMismatchError(
                f"Query vector has {query.shape[0]} dimensions, store holds {self._dimensions}."
            )

        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm

        candidates = self._candidate_indices(filters)
        if not candidates.size:
            return []

        matrix = self._ensure_matrix()
        scores = matrix[candidates] @ query

        # argpartition avoids a full sort when only the top few are wanted.
        limit = min(top_k, candidates.size)
        top_positions = np.argpartition(-scores, limit - 1)[:limit]
        top_positions = top_positions[np.argsort(-scores[top_positions])]

        return [
            SearchHit(
                id=self._ids[candidates[position]],
                score=float(scores[position]),
                metadata=self._metadata[candidates[position]],
                payload=self._payloads[candidates[position]],
            )
            for position in top_positions
        ]

    async def count(self) -> int:
        return len(self._ids)

    async def clear(self) -> None:
        self._ids.clear()
        self._index_by_id.clear()
        self._metadata.clear()
        self._payloads.clear()
        self._vectors.clear()
        self._matrix = None
        self._dimensions = None

    def _prepare(self, record: VectorRecord) -> np.ndarray:
        vector = np.asarray(record.embedding, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] == 0:
            raise DimensionMismatchError(f"Record '{record.id}' has no usable vector.")
        if self._dimensions is None:
            self._dimensions = int(vector.shape[0])
        elif vector.shape[0] != self._dimensions:
            raise DimensionMismatchError(
                f"Record '{record.id}' has {vector.shape[0]} dimensions, "
                f"store holds {self._dimensions}."
            )

        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.vstack(self._vectors)
        return self._matrix

    def _candidate_indices(self, filters: SearchFilters | None) -> np.ndarray:
        if filters is None or filters.is_empty():
            return np.arange(len(self._ids))
        matching = [
            index for index, metadata in enumerate(self._metadata) if _matches(metadata, filters)
        ]
        return np.asarray(matching, dtype=np.int64)


def _matches(metadata: Mapping[str, Any], filters: SearchFilters) -> bool:
    for field_name, allowed in filters.any_of.items():
        value = metadata.get(field_name)
        if value is None:
            return False
        if isinstance(value, list | tuple | set):
            values = {str(item) for item in value}
        else:
            values = {str(value)}
        if values.isdisjoint(allowed):
            return False
    return True
