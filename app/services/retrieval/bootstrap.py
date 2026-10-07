"""Wiring for the retrieval stack.

One place that assembles embeddings + vector store + ingestion + retriever from
settings, so the CLI, the evaluation harness, the tests, and (in a later phase)
application startup all build the same thing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.config import EmbeddingProvider, Settings
from app.domain.resource import VerificationStatus
from app.services.embeddings.base import EmbeddingService
from app.services.embeddings.factory import build_embedding_service
from app.services.ingestion.loader import load_resources_from_file
from app.services.ingestion.pipeline import IngestionReport, ResourceIngestionPipeline
from app.services.retrieval.retriever import ResourceRetriever
from app.services.vectorstore.base import VectorStore
from app.services.vectorstore.memory_store import InMemoryVectorStore
from app.services.vectorstore.persistence import (
    IndexManifest,
    ResourceIndexCache,
    fingerprint_file,
)

logger = logging.getLogger(__name__)

# Similarity scores are not comparable across providers: the lexical provider
# runs roughly 0.03-0.30 while Vertex embeddings sit around 0.46-0.79, because
# dense embeddings place everything in a much narrower cone. A single default
# would either admit everything or reject everything, so each provider carries
# its own value: the midpoint of the best-scoring plateau measured by
# scripts/eval_retrieval.py against app/data/eval_queries.json. Re-run that
# script after changing the resource data, the embedding model, or the ranking.
DEFAULT_MIN_SCORE: dict[EmbeddingProvider, float] = {
    EmbeddingProvider.HASHING: 0.10,
    EmbeddingProvider.VERTEX: 0.57,
}


def resolve_servable_statuses(settings: Settings) -> frozenset[VerificationStatus]:
    """Parse the configured statuses, ignoring any name we do not recognise."""
    statuses = set()
    for raw in settings.retrieval_servable_statuses:
        try:
            statuses.add(VerificationStatus(raw.strip().lower()))
        except ValueError:
            logger.warning("Ignoring unknown verification status in config: %r", raw)
    return frozenset(statuses)


def resolve_min_score(settings: Settings) -> float:
    """The configured gate, or the value tuned for the configured provider."""
    if settings.retrieval_min_score is not None:
        return settings.retrieval_min_score
    return DEFAULT_MIN_SCORE.get(settings.embedding_provider, 0.10)


@dataclass
class RetrievalStack:
    """The assembled retrieval components."""

    embeddings: EmbeddingService
    store: VectorStore
    pipeline: ResourceIngestionPipeline
    retriever: ResourceRetriever

    async def aclose(self) -> None:
        await self.embeddings.aclose()
        await self.store.aclose()


def build_retrieval_stack(settings: Settings, store: VectorStore | None = None) -> RetrievalStack:
    """Assemble the retrieval stack without loading any data."""
    embeddings = build_embedding_service(settings)
    vector_store = store or InMemoryVectorStore()

    return RetrievalStack(
        embeddings=embeddings,
        store=vector_store,
        pipeline=ResourceIngestionPipeline(embeddings, vector_store),
        retriever=ResourceRetriever(
            embeddings,
            vector_store,
            top_k=settings.retrieval_top_k,
            min_score=resolve_min_score(settings),
            location_boost=settings.retrieval_location_boost,
            category_boost=settings.retrieval_category_boost,
            verified_boost=settings.retrieval_verified_boost,
            partially_verified_boost=settings.retrieval_partially_verified_boost,
            servable_statuses=resolve_servable_statuses(settings),
            candidate_multiplier=settings.retrieval_candidate_multiplier,
            max_resource_age_days=settings.retrieval_max_resource_age_days,
            require_verified=settings.retrieval_require_verified,
        ),
    )


def build_manifest(settings: Settings, embeddings: EmbeddingService) -> IndexManifest:
    """Describe the index the current configuration would produce."""
    return IndexManifest(
        embedding_provider=embeddings.provider_name,
        embedding_model=embeddings.model_name,
        dimensions=embeddings.dimensions,
        resource_fingerprint=fingerprint_file(settings.resource_data_path),
        resource_count=0,
    )


async def bootstrap_retrieval(
    settings: Settings, store: VectorStore | None = None
) -> tuple[RetrievalStack, IngestionReport]:
    """Assemble the stack and load the resource index.

    Uses the persisted index when it was produced by the same provider, model,
    dimensionality, and resource data; otherwise re-embeds and rewrites it.
    """
    stack = build_retrieval_stack(settings, store)

    if not settings.resource_index_cache_enabled:
        report = await stack.pipeline.ingest_from_file(settings.resource_data_path)
        return stack, report

    started = time.perf_counter()
    cache = ResourceIndexCache(settings.resource_index_cache_path)
    expected = build_manifest(settings, stack.embeddings)

    loaded = cache.load(expected)
    if loaded.is_hit and loaded.records is not None:
        indexed = await stack.store.upsert(loaded.records)
        return stack, IngestionReport(
            loaded=indexed,
            indexed=indexed,
            embedding_provider=stack.embeddings.provider_name,
            embedding_model=stack.embeddings.model_name,
            dimensions=stack.embeddings.dimensions,
            duration_ms=int((time.perf_counter() - started) * 1000),
            source="cache",
        )

    logger.info("Rebuilding resource index (%s: %s)", loaded.status, loaded.reason)

    result = load_resources_from_file(settings.resource_data_path)
    for issue in result.issues:
        logger.warning(
            "Skipped resource %s (index %d): %s",
            issue.resource_id or "<no id>",
            issue.index,
            issue.reason,
        )

    records = await stack.pipeline.build_records(result.resources)
    indexed = await stack.store.upsert(records)
    cache.save(expected, records)

    return stack, IngestionReport(
        loaded=result.loaded_count,
        indexed=indexed,
        rejected=result.rejected_count,
        issues=result.issues,
        embedding_provider=stack.embeddings.provider_name,
        embedding_model=stack.embeddings.model_name,
        dimensions=stack.embeddings.dimensions,
        duration_ms=int((time.perf_counter() - started) * 1000),
        source="embedded",
        cache_note=f"cache {loaded.status}: {loaded.reason}",
    )
