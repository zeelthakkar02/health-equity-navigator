"""Embedding provider selection."""

from __future__ import annotations

import pytest

from app.core.config import EmbeddingProvider, Settings
from app.services.embeddings.errors import EmbeddingConfigurationError
from app.services.embeddings.factory import build_embedding_service
from app.services.embeddings.hashing_provider import HashingEmbeddingService


def test_hashing_is_the_default(settings: Settings) -> None:
    service = build_embedding_service(settings)

    assert isinstance(service, HashingEmbeddingService)
    assert service.describe() == {
        "provider": "hashing",
        "model": "hashing-lexical-v1",
        "dimensions": 768,
    }


def test_configured_dimensions_are_honoured(settings: Settings) -> None:
    service = build_embedding_service(settings.model_copy(update={"embedding_dimensions": 256}))
    assert service.dimensions == 256


def test_vertex_without_a_project_raises_a_configuration_error(settings: Settings) -> None:
    broken = settings.model_copy(
        update={
            "embedding_provider": EmbeddingProvider.VERTEX,
            "google_cloud_project": None,
        }
    )

    with pytest.raises(EmbeddingConfigurationError, match="GOOGLE_CLOUD_PROJECT"):
        build_embedding_service(broken)


def test_vertex_embeddings_require_google_settings() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="GOOGLE_CLOUD_PROJECT"):
        Settings(
            _env_file=None,
            embedding_provider=EmbeddingProvider.VERTEX,
            google_cloud_project=None,
        )


def test_embedding_provider_is_independent_of_the_llm_provider(settings: Settings) -> None:
    """Retrieval can stay offline while generation uses Vertex, and vice versa."""
    assert settings.llm_provider.value == "stub"
    assert settings.embedding_provider is EmbeddingProvider.HASHING
    assert build_embedding_service(settings).provider_name == "hashing"


def test_each_provider_carries_its_own_tuned_relevance_gate(settings: Settings) -> None:
    """Scores are an order of magnitude apart, so one default cannot serve both."""
    from app.services.retrieval.bootstrap import DEFAULT_MIN_SCORE, resolve_min_score

    hashing = resolve_min_score(settings)
    vertex = resolve_min_score(
        settings.model_copy(
            update={
                "embedding_provider": EmbeddingProvider.VERTEX,
                "google_cloud_project": "example-gcp-project",
            }
        )
    )

    assert hashing == DEFAULT_MIN_SCORE[EmbeddingProvider.HASHING]
    assert vertex == DEFAULT_MIN_SCORE[EmbeddingProvider.VERTEX]
    assert vertex > hashing


def test_an_explicit_min_score_overrides_the_provider_default(settings: Settings) -> None:
    from app.services.retrieval.bootstrap import resolve_min_score

    assert resolve_min_score(settings.model_copy(update={"retrieval_min_score": 0.33})) == 0.33
