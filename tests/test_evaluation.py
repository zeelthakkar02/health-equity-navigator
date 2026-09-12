"""The retrieval evaluation harness.

Metrics are computed from synthetic outcomes so the arithmetic is checked
independently of any embedding provider.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.domain.resource import Resource, ServiceCategory
from app.services.retrieval.evaluation import (
    EvalQuery,
    QueryOutcome,
    candidate_thresholds,
    load_eval_queries,
    recommend_threshold,
    score_at_threshold,
    sweep,
)
from app.services.retrieval.retriever import RetrievedResource


def _resource(resource_id: str, *categories: ServiceCategory) -> Resource:
    return Resource.model_validate(
        {
            "resource_id": resource_id,
            "organization_name": f"Org {resource_id}",
            "description": "A synthetic organization used only in tests.",
            "service_categories": [category.value for category in categories],
            "source": "test-fixture",
            "last_verified": "2026-06-01",
        }
    )


def _result(rank: int, score: float, resource: Resource) -> RetrievedResource:
    return RetrievedResource(
        resource=resource,
        score=score,
        semantic_score=score,
        location_match=0.0,
        rank=rank,
    )


def _outcome(
    query_id: str,
    expected: list[ServiceCategory],
    ranked: list[tuple[float, list[ServiceCategory]]],
    *,
    off_topic: bool = False,
    expected_ids: list[str] | None = None,
) -> QueryOutcome:
    return QueryOutcome(
        query=EvalQuery(
            query_id=query_id,
            query=f"query {query_id}",
            expected_categories=frozenset(expected),
            expected_resource_ids=frozenset(expected_ids or []),
            off_topic=off_topic,
        ),
        results=[
            _result(index + 1, score, _resource(f"{query_id}-{index}", *categories))
            for index, (score, categories) in enumerate(ranked)
        ],
    )


# --- dataset ---------------------------------------------------------------


def test_the_shipped_eval_set_loads(settings: Settings) -> None:
    queries = load_eval_queries(settings.eval_queries_path)

    assert len(queries) >= 30
    assert sum(1 for query in queries if query.off_topic) >= 5
    assert all(query.expected_categories or query.off_topic for query in queries), (
        "every answerable query needs expected categories"
    )
    assert all(not (query.off_topic and query.expected_categories) for query in queries)


def test_the_eval_set_covers_every_service_category(settings: Settings) -> None:
    queries = load_eval_queries(settings.eval_queries_path)
    covered = {category for query in queries for category in query.expected_categories}

    assert covered == set(ServiceCategory)


def test_query_ids_are_unique(settings: Settings) -> None:
    queries = load_eval_queries(settings.eval_queries_path)
    assert len({query.query_id for query in queries}) == len(queries)


# --- metrics ---------------------------------------------------------------


def test_a_perfect_run_scores_one() -> None:
    outcomes = [
        _outcome(
            "a", [ServiceCategory.FOOD_ASSISTANCE], [(0.9, [ServiceCategory.FOOD_ASSISTANCE])]
        ),
        _outcome("off", [], [(0.2, [ServiceCategory.FOOD_ASSISTANCE])], off_topic=True),
    ]

    metrics = score_at_threshold(outcomes, 0.5)

    assert metrics.top1_category_accuracy == 1.0
    assert metrics.category_recall_at_k == 1.0
    assert metrics.rejection_rate == 1.0
    assert metrics.combined == 1.0


def test_top1_is_wrong_but_recall_at_3_still_counts() -> None:
    outcomes = [
        _outcome(
            "a",
            [ServiceCategory.FOOD_ASSISTANCE],
            [
                (0.9, [ServiceCategory.HOUSING_SUPPORT]),
                (0.8, [ServiceCategory.FOOD_ASSISTANCE]),
            ],
        )
    ]

    metrics = score_at_threshold(outcomes, 0.0)

    assert metrics.top1_category_accuracy == 0.0
    assert metrics.category_recall_at_k == 1.0


def test_a_hit_below_rank_three_does_not_count_for_recall() -> None:
    outcomes = [
        _outcome(
            "a",
            [ServiceCategory.FOOD_ASSISTANCE],
            [
                (0.9, [ServiceCategory.HOUSING_SUPPORT]),
                (0.8, [ServiceCategory.HOUSING_SUPPORT]),
                (0.7, [ServiceCategory.HOUSING_SUPPORT]),
                (0.6, [ServiceCategory.FOOD_ASSISTANCE]),
            ],
        )
    ]

    assert score_at_threshold(outcomes, 0.0).category_recall_at_k == 0.0


def test_the_threshold_filters_results() -> None:
    outcomes = [
        _outcome("a", [ServiceCategory.FOOD_ASSISTANCE], [(0.4, [ServiceCategory.FOOD_ASSISTANCE])])
    ]

    assert score_at_threshold(outcomes, 0.5).answerable_empty == 1
    assert score_at_threshold(outcomes, 0.5).top1_category_accuracy == 0.0
    assert score_at_threshold(outcomes, 0.3).top1_category_accuracy == 1.0


def test_off_topic_rejection_counts_empty_result_sets() -> None:
    outcomes = [
        _outcome("off1", [], [(0.6, [ServiceCategory.FOOD_ASSISTANCE])], off_topic=True),
        _outcome("off2", [], [(0.2, [ServiceCategory.FOOD_ASSISTANCE])], off_topic=True),
    ]

    assert score_at_threshold(outcomes, 0.5).rejection_rate == 0.5


def test_resource_recall_only_counts_labelled_queries() -> None:
    outcomes = [
        _outcome(
            "a",
            [ServiceCategory.FOOD_ASSISTANCE],
            [(0.9, [ServiceCategory.FOOD_ASSISTANCE])],
            expected_ids=["a-0"],
        ),
        _outcome(
            "b", [ServiceCategory.FOOD_ASSISTANCE], [(0.9, [ServiceCategory.FOOD_ASSISTANCE])]
        ),
    ]

    metrics = score_at_threshold(outcomes, 0.0)

    assert metrics.resource_labelled == 1
    assert metrics.resource_recall_at_k == 1.0


def test_metrics_with_no_queries_do_not_divide_by_zero() -> None:
    metrics = score_at_threshold([], 0.5)

    assert metrics.top1_category_accuracy == 0.0
    assert metrics.rejection_rate == 0.0


# --- threshold selection ---------------------------------------------------


def test_candidate_thresholds_span_the_observed_scores() -> None:
    outcomes = [
        _outcome(
            "a",
            [ServiceCategory.FOOD_ASSISTANCE],
            [(0.9, [ServiceCategory.FOOD_ASSISTANCE]), (0.3, [ServiceCategory.HOUSING_SUPPORT])],
        )
    ]

    grid = candidate_thresholds(outcomes, steps=4)

    assert grid[0] == pytest.approx(0.3)
    assert grid[-1] == pytest.approx(0.9)


def test_the_recommendation_sits_in_the_middle_of_the_best_plateau() -> None:
    """An edge of the plateau is one data change away from a cliff."""
    outcomes = [
        _outcome(
            "a", [ServiceCategory.FOOD_ASSISTANCE], [(0.9, [ServiceCategory.FOOD_ASSISTANCE])]
        ),
        _outcome("off", [], [(0.5, [ServiceCategory.HOUSING_SUPPORT])], off_topic=True),
    ]
    # Perfect between 0.51 and 0.90; below admits the off-topic hit, above loses
    # the real one.
    grid = [0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95]

    best = recommend_threshold(sweep(outcomes, grid))

    assert 0.55 <= best.threshold <= 0.80
    assert best.combined == 1.0


def test_recommending_from_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no metrics"):
        recommend_threshold([])
