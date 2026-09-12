"""The offline hashing embedding provider."""

from __future__ import annotations

import math

import pytest

from app.services.embeddings.hashing_provider import HashingEmbeddingService
from app.services.text_normalization import tokenize


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_vectors_have_the_configured_width() -> None:
    service = HashingEmbeddingService(dimensions=256)
    assert len(service.embed("free groceries")) == 256
    assert service.dimensions == 256


def test_vectors_are_unit_length() -> None:
    vector = HashingEmbeddingService().embed("wheelchair accessible transportation")
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_embeddings_are_deterministic_across_instances() -> None:
    text = "rides to dialysis appointments"
    assert HashingEmbeddingService().embed(text) == HashingEmbeddingService().embed(text)


def test_related_text_scores_higher_than_unrelated_text() -> None:
    service = HashingEmbeddingService()
    query = service.embed("I need food assistance")
    food = service.embed("Food pantry providing free groceries and emergency food boxes")
    housing = service.embed("Emergency rental assistance and eviction defense referrals")

    assert _cosine(query, food) > _cosine(query, housing)


def test_text_without_usable_tokens_yields_a_zero_vector() -> None:
    vector = HashingEmbeddingService().embed("the and of")
    assert not any(vector)


async def test_document_order_is_preserved() -> None:
    service = HashingEmbeddingService()
    texts = ["food pantry", "wheelchair ramps", "medical interpretation"]

    vectors = await service.embed_documents(texts)

    assert len(vectors) == 3
    assert vectors[1] == service.embed("wheelchair ramps")


async def test_empty_document_list_is_handled() -> None:
    assert await HashingEmbeddingService().embed_documents([]) == []


async def test_query_and_document_embedding_agree_for_this_provider() -> None:
    service = HashingEmbeddingService()
    assert await service.embed_query("food") == (await service.embed_documents(["food"]))[0]


def test_tokenizer_drops_stopwords_and_stems() -> None:
    tokens = tokenize("I need rides to the appointments")
    assert "the" not in tokens
    assert "need" not in tokens
    assert "ride" in tokens


def test_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        HashingEmbeddingService(dimensions=0)
