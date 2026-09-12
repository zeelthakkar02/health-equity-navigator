"""The ingestion pipeline: load, embed, index.

Deliberately policy-free. Everything that loads is indexed, including records
that are unverified or overdue for re-verification; deciding what a community
member may actually be shown belongs to retrieval, where the request context
lives.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.domain.resource import Resource
from app.services.embeddings.base import EmbeddingService
from app.services.ingestion.documents import (
    build_embedding_text,
    build_metadata,
    build_payload,
)
from app.services.ingestion.loader import LoadIssue, load_resources_from_file
from app.services.vectorstore.base import VectorRecord, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionReport:
    """What one ingestion run did."""

    loaded: int = 0
    indexed: int = 0
    rejected: int = 0
    issues: list[LoadIssue] = field(default_factory=list)
    embedding_provider: str = ""
    embedding_model: str = ""
    dimensions: int = 0
    duration_ms: int = 0
    source: str = "embedded"
    cache_note: str = ""

    def summary(self) -> str:
        origin = f"from {self.source}" if self.source != "embedded" else "embedded"
        note = f"; {self.cache_note}" if self.cache_note else ""
        return (
            f"{self.indexed} resources indexed, {self.rejected} rejected "
            f"({self.embedding_provider}/{self.embedding_model}, "
            f"{self.dimensions} dims, {origin}, {self.duration_ms} ms{note})"
        )


class ResourceIngestionPipeline:
    """Embeds resources and writes them to a vector store."""

    def __init__(self, embeddings: EmbeddingService, store: VectorStore) -> None:
        self._embeddings = embeddings
        self._store = store

    async def build_records(self, resources: Sequence[Resource]) -> list[VectorRecord]:
        """Embed resources into indexable records without writing them.

        Separated from :meth:`ingest` so a caller that also persists the index
        can hold on to the records it just paid to embed.
        """
        if not resources:
            return []

        texts = [build_embedding_text(resource) for resource in resources]
        vectors = await self._embeddings.embed_documents(texts)

        if len(vectors) != len(resources):
            raise RuntimeError(
                f"Embedding provider returned {len(vectors)} vectors "
                f"for {len(resources)} resources."
            )

        return [
            VectorRecord(
                id=resource.resource_id,
                embedding=vector,
                metadata=build_metadata(resource),
                payload=build_payload(resource),
            )
            for resource, vector in zip(resources, vectors, strict=True)
        ]

    async def ingest(self, resources: Sequence[Resource]) -> IngestionReport:
        started = time.perf_counter()

        if not resources:
            return IngestionReport(
                embedding_provider=self._embeddings.provider_name,
                embedding_model=self._embeddings.model_name,
                dimensions=self._embeddings.dimensions,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        records = await self.build_records(resources)
        indexed = await self._store.upsert(records)

        report = IngestionReport(
            loaded=len(resources),
            indexed=indexed,
            embedding_provider=self._embeddings.provider_name,
            embedding_model=self._embeddings.model_name,
            dimensions=self._embeddings.dimensions,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        logger.info("Ingestion complete: %s", report.summary())
        return report

    async def ingest_from_file(self, path: Path | str) -> IngestionReport:
        """Load a resource file and index everything that validates."""
        result = load_resources_from_file(path)
        for issue in result.issues:
            logger.warning(
                "Skipped resource %s (index %d): %s",
                issue.resource_id or "<no id>",
                issue.index,
                issue.reason,
            )

        report = await self.ingest(result.resources)
        return IngestionReport(
            loaded=result.loaded_count,
            indexed=report.indexed,
            rejected=result.rejected_count,
            issues=result.issues,
            embedding_provider=report.embedding_provider,
            embedding_model=report.embedding_model,
            dimensions=report.dimensions,
            duration_ms=report.duration_ms,
            source=report.source,
        )
