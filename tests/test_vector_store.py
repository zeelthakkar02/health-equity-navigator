"""The in-memory vector store."""

from __future__ import annotations

import pytest

from app.services.vectorstore.base import (
    DimensionMismatchError,
    SearchFilters,
    VectorRecord,
)
from app.services.vectorstore.memory_store import InMemoryVectorStore


def _record(record_id: str, vector: list[float], **metadata: object) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        embedding=vector,
        metadata=metadata,
        payload={"resource_id": record_id, "name": record_id.upper()},
    )


async def test_upsert_and_count() -> None:
    store = InMemoryVectorStore()

    written = await store.upsert([_record("a", [1.0, 0.0]), _record("b", [0.0, 1.0])])

    assert written == 2
    assert await store.count() == 2
    assert store.dimensions == 2


async def test_search_returns_hits_ordered_by_similarity() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _record("east", [1.0, 0.0]),
            _record("north", [0.0, 1.0]),
            _record("northeast", [0.7, 0.7]),
        ]
    )

    hits = await store.search([1.0, 0.0], top_k=3)

    assert [hit.id for hit in hits] == ["east", "northeast", "north"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score >= hits[1].score >= hits[2].score


async def test_search_respects_top_k() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record(str(index), [1.0, index / 10]) for index in range(10)])

    assert len(await store.search([1.0, 0.0], top_k=3)) == 3


async def test_payload_is_returned_verbatim() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0])])

    hit = (await store.search([1.0, 0.0], top_k=1))[0]

    assert hit.payload == {"resource_id": "a", "name": "A"}


async def test_vectors_need_not_be_pre_normalised() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("long", [10.0, 0.0])])

    hit = (await store.search([3.0, 0.0], top_k=1))[0]

    assert hit.score == pytest.approx(1.0)


async def test_upsert_replaces_a_record_with_the_same_id() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0], city="oakland")])
    await store.upsert([_record("a", [0.0, 1.0], city="fresno")])

    assert await store.count() == 1
    hits = await store.search([0.0, 1.0], top_k=1)
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].metadata["city"] == "fresno"


async def test_filters_restrict_candidates() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _record("food", [1.0, 0.0], categories=["food_assistance"]),
            _record("housing", [1.0, 0.0], categories=["housing_support"]),
        ]
    )

    hits = await store.search(
        [1.0, 0.0],
        top_k=5,
        filters=SearchFilters(any_of={"categories": frozenset({"housing_support"})}),
    )

    assert [hit.id for hit in hits] == ["housing"]


async def test_filters_match_scalar_metadata_too() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0], state="ca"), _record("b", [1.0, 0.0], state="ny")])

    hits = await store.search(
        [1.0, 0.0], top_k=5, filters=SearchFilters(any_of={"state": frozenset({"ca"})})
    )

    assert [hit.id for hit in hits] == ["a"]


async def test_a_filter_with_no_matches_returns_nothing() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0], categories=["food_assistance"])])

    hits = await store.search(
        [1.0, 0.0], top_k=5, filters=SearchFilters(any_of={"categories": frozenset({"nope"})})
    )

    assert hits == []


async def test_searching_an_empty_store_returns_nothing() -> None:
    assert await InMemoryVectorStore().search([1.0, 0.0], top_k=5) == []


async def test_a_zero_query_vector_returns_nothing() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0])])

    assert await store.search([0.0, 0.0], top_k=5) == []


async def test_mismatched_record_width_is_rejected() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0])])

    with pytest.raises(DimensionMismatchError, match="3 dimensions"):
        await store.upsert([_record("b", [1.0, 0.0, 0.0])])


async def test_mismatched_query_width_is_rejected() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0])])

    with pytest.raises(DimensionMismatchError, match="Query vector"):
        await store.search([1.0, 0.0, 0.0], top_k=1)


async def test_clear_empties_the_store() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0])])

    await store.clear()

    assert await store.count() == 0
    assert store.dimensions is None


async def test_top_k_must_be_positive() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record("a", [1.0, 0.0])])

    with pytest.raises(ValueError, match="top_k"):
        await store.search([1.0, 0.0], top_k=0)
