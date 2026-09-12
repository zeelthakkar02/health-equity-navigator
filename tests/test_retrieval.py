"""Semantic retrieval over the verified resource index.

These run on the offline hashing provider, so they assert on ranking behaviour
and policy — not on semantic understanding the provider does not have.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.core.config import Settings
from app.domain.resource import Resource, ServiceCategory
from app.services.embeddings.hashing_provider import HashingEmbeddingService
from app.services.ingestion.pipeline import ResourceIngestionPipeline
from app.services.retrieval.bootstrap import bootstrap_retrieval
from app.services.retrieval.retriever import ResourceRetriever
from app.services.vectorstore.memory_store import InMemoryVectorStore

TODAY = date(2026, 9, 11)


@pytest.fixture
async def stack(settings: Settings):
    """The full retrieval stack, loaded with the synthetic sample data."""
    built, _ = await bootstrap_retrieval(settings)
    yield built
    await built.aclose()


@pytest.fixture
async def permissive_stack(settings: Settings):
    """Same stack with the relevance gate open, for testing ranking itself.

    The offline lexical provider scores some perfectly reasonable queries below
    the tuned gate; those queries return nothing, which is correct behaviour but
    useless for asserting on rank order.
    """
    built, _ = await bootstrap_retrieval(settings.model_copy(update={"retrieval_min_score": 0.0}))
    yield built
    await built.aclose()


def categories_of(result) -> set[ServiceCategory]:
    return set(result.resource.service_categories)


# --- the example queries ---------------------------------------------------


async def test_transportation_need_surfaces_transportation_resources(stack) -> None:
    results = await stack.retriever.retrieve(
        "I need transportation to my doctor appointment.", as_of=TODAY
    )

    assert results
    assert ServiceCategory.TRANSPORTATION in categories_of(results[0])
    assert results[0].rank == 1


async def test_food_need_near_oakland_surfaces_local_food_resources(stack) -> None:
    results = await stack.retriever.retrieve(
        "I need food assistance near Oakland.", location="Oakland", as_of=TODAY
    )

    assert results
    top = results[0]
    assert ServiceCategory.FOOD_ASSISTANCE in categories_of(top)
    assert "Oakland" in top.resource.service_area.cities
    assert top.matched_location


async def test_wheelchair_need_surfaces_accessibility_resources(stack) -> None:
    results = await stack.retriever.retrieve(
        "My mother needs wheelchair-accessible support.", as_of=TODAY
    )

    assert results
    assert ServiceCategory.ACCESSIBILITY_SUPPORT in categories_of(results[0])


# --- ranking behaviour -----------------------------------------------------


async def test_results_are_ranked_and_numbered(stack) -> None:
    results = await stack.retriever.retrieve("food pantry groceries", as_of=TODAY)

    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


async def test_top_k_limits_the_result_count(stack) -> None:
    results = await stack.retriever.retrieve("food", top_k=2, as_of=TODAY)
    assert len(results) <= 2


async def test_location_boosts_but_does_not_filter(stack) -> None:
    """A statewide service must still be reachable from a specific city."""
    results = await stack.retriever.retrieve(
        "help paying medical bills", location="Oakland", as_of=TODAY
    )

    names = {result.resource.organization_name for result in results}
    assert "Medical Debt Relief Project" in names  # statewide, not Oakland-specific


async def test_a_matching_location_raises_a_resource_above_an_equal_one(stack) -> None:
    without = await stack.retriever.retrieve("food assistance", top_k=10, as_of=TODAY)
    with_location = await stack.retriever.retrieve(
        "food assistance", location="Fresno", top_k=10, as_of=TODAY
    )

    fresno_rank_before = _rank_of(without, "Fresno Mobile Meals Project")
    fresno_rank_after = _rank_of(with_location, "Fresno Mobile Meals Project")

    assert fresno_rank_after < fresno_rank_before


async def test_location_boost_is_recorded_on_the_result(stack) -> None:
    results = await stack.retriever.retrieve(
        "food assistance", location="94601", top_k=10, as_of=TODAY
    )
    pantry = _find(results, "Oakland Community Food Pantry")

    assert pantry.location_match == 1.0
    assert pantry.score > pantry.semantic_score


async def test_an_unrelated_query_scores_far_below_a_relevant_one(stack) -> None:
    """Feature hashing has a collision noise floor, so this is a comparison.

    An off-topic query still produces small non-zero similarities because
    unrelated terms occasionally land in the same bucket. What must hold is that
    a genuine match scores clearly higher. Real semantic embeddings separate
    these much more sharply.
    """
    relevant = await stack.retriever.retrieve("food pantry groceries", as_of=TODAY)
    unrelated = await stack.retriever.retrieve(
        "quantum chromodynamics research grants", location="Oakland", as_of=TODAY
    )

    assert relevant
    top_unrelated = unrelated[0].semantic_score if unrelated else 0.0
    assert relevant[0].semantic_score > top_unrelated * 1.5


async def test_location_alone_cannot_promote_an_off_topic_resource(stack) -> None:
    """The boost re-ranks what already cleared the relevance gate; it never adds."""
    results = await stack.retriever.retrieve(
        "quantum chromodynamics research grants",
        location="Oakland",
        top_k=20,
        as_of=TODAY,
    )

    assert all(result.semantic_score >= 0.05 for result in results)


# --- filters ---------------------------------------------------------------


async def test_category_filter_restricts_results(stack) -> None:
    results = await stack.retriever.retrieve(
        "help with my appointment",
        categories=[ServiceCategory.TRANSPORTATION],
        top_k=10,
        as_of=TODAY,
    )

    assert results
    assert all(ServiceCategory.TRANSPORTATION in categories_of(r) for r in results)


async def test_language_filter_restricts_results(stack) -> None:
    results = await stack.retriever.retrieve(
        "interpreter for my appointment", languages=["Hmong"], top_k=10, as_of=TODAY
    )

    assert results
    assert all(
        any(language.lower() == "hmong" for language in r.resource.languages) for r in results
    )


async def test_an_empty_query_returns_nothing(stack) -> None:
    assert await stack.retriever.retrieve("   ", as_of=TODAY) == []


# --- verification and freshness policy -------------------------------------


def _resource(resource_id: str, **overrides: object) -> Resource:
    base: dict[str, object] = {
        "resource_id": resource_id,
        "organization_name": f"Example Food Support {resource_id}",
        "description": "Free groceries and emergency food boxes for local families.",
        "service_categories": ["food_assistance"],
        "source": "test-fixture",
        "last_verified": TODAY - timedelta(days=30),
    }
    return Resource.model_validate(base | overrides)


async def _retriever_over(resources: list[Resource], **kwargs: object) -> ResourceRetriever:
    embeddings = HashingEmbeddingService()
    store = InMemoryVectorStore()
    await ResourceIngestionPipeline(embeddings, store).ingest(resources)
    return ResourceRetriever(embeddings, store, **kwargs)  # type: ignore[arg-type]


async def test_unverified_resources_are_withheld() -> None:
    retriever = await _retriever_over(
        [
            _resource("ok-1"),
            _resource("pending-1", verification_status="needs_review"),
            _resource("unverified-1", verification_status="unverified"),
        ]
    )

    results = await retriever.retrieve("food groceries", top_k=10, as_of=TODAY)

    assert {r.resource.resource_id for r in results} == {"ok-1"}


async def test_stale_resources_are_withheld() -> None:
    retriever = await _retriever_over(
        [
            _resource("fresh-1"),
            _resource("stale-1", last_verified=TODAY - timedelta(days=900)),
        ],
        max_resource_age_days=548,
    )

    results = await retriever.retrieve("food groceries", top_k=10, as_of=TODAY)

    assert {r.resource.resource_id for r in results} == {"fresh-1"}


async def test_the_verification_requirement_can_be_relaxed() -> None:
    retriever = await _retriever_over(
        [_resource("unverified-1", verification_status="unverified")],
        require_verified=False,
    )

    results = await retriever.retrieve("food groceries", top_k=10, as_of=TODAY)

    assert len(results) == 1


async def test_the_relevance_gate_uses_semantic_score_only() -> None:
    """A high min_score must not be rescued by a location match."""
    retriever = await _retriever_over([_resource("ok-1")], min_score=0.99)

    results = await retriever.retrieve("food groceries", location="Oakland", as_of=TODAY)

    assert results == []


def _find(results, name: str):
    for result in results:
        if result.resource.organization_name == name:
            return result
    raise AssertionError(f"{name!r} not in results")


def _rank_of(results, name: str) -> int:
    return _find(results, name).rank


# --- stated-need boosting --------------------------------------------------


async def test_a_stated_need_outranks_incidental_context(stack) -> None:
    """The flagship case: 'my mother' is context, 'wheelchair' is the need."""
    results = await stack.retriever.retrieve(
        "My mother needs wheelchair-accessible support.", top_k=3, as_of=TODAY
    )

    assert results
    assert all(ServiceCategory.ACCESSIBILITY_SUPPORT in categories_of(result) for result in results)
    assert results[0].matched_stated_need


async def test_caregiver_resources_still_win_a_caregiving_query(permissive_stack) -> None:
    results = await permissive_stack.retriever.retrieve(
        "I'm caring for my husband and need respite.", top_k=3, as_of=TODAY
    )

    assert results
    assert ServiceCategory.CAREGIVER_SUPPORT in categories_of(results[0])
    assert results[0].matched_stated_need


async def test_the_category_boost_is_recorded_on_the_result(stack) -> None:
    results = await stack.retriever.retrieve("I need a wheelchair ramp", top_k=5, as_of=TODAY)
    boosted = [result for result in results if result.matched_stated_need]

    assert boosted
    assert all(result.score > result.semantic_score for result in boosted)


async def test_a_query_stating_no_need_is_ranked_by_similarity_alone(stack) -> None:
    results = await stack.retriever.retrieve("help for my family", top_k=5, as_of=TODAY)

    assert all(result.category_match == 0.0 for result in results)
    assert all(result.score == result.semantic_score for result in results)


async def test_the_boost_reorders_but_never_admits(stack) -> None:
    """Boosting must not pull in anything the relevance gate rejected."""
    retriever = await _retriever_over([_resource("ok-1")], min_score=0.99, category_boost=0.5)

    assert await retriever.retrieve("food groceries pantry", as_of=TODAY) == []
