"""Retrieval quality evaluation.

Measures the retriever against a hand-written query set and sweeps the relevance
threshold so ``RETRIEVAL_MIN_SCORE`` is chosen from data rather than guessed.

Queries are collected **once** with the gate wide open; every threshold is then
evaluated by filtering those cached results. Sweeping this way costs one
embedding pass instead of one per candidate threshold.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.domain.resource import ServiceCategory
from app.services.retrieval.retriever import ResourceRetriever, RetrievedResource

RECALL_AT = 3


@dataclass(frozen=True)
class EvalQuery:
    """One labelled need."""

    query_id: str
    query: str
    location: str | None = None
    expected_categories: frozenset[ServiceCategory] = frozenset()
    expected_resource_ids: frozenset[str] = frozenset()
    off_topic: bool = False
    note: str = ""


@dataclass(frozen=True)
class QueryOutcome:
    """Everything the retriever returned for one query, ungated."""

    query: EvalQuery
    results: list[RetrievedResource] = field(default_factory=list)

    def above(self, threshold: float) -> list[RetrievedResource]:
        """Results that would survive a given relevance gate."""
        return [result for result in self.results if result.semantic_score >= threshold]


@dataclass(frozen=True)
class Metrics:
    """Retrieval quality at one threshold."""

    threshold: float
    answerable: int
    off_topic: int
    top1_category_accuracy: float
    category_recall_at_k: float
    resource_recall_at_k: float
    resource_labelled: int
    rejection_rate: float
    answerable_empty: int
    mean_results: float

    @property
    def combined(self) -> float:
        """Balance of the three headline measures, used to pick a threshold."""
        return (self.top1_category_accuracy + self.category_recall_at_k + self.rejection_rate) / 3.0

    def as_row(self) -> str:
        return (
            f"{self.threshold:>6.2f}  {self.top1_category_accuracy:>9.1%}  "
            f"{self.category_recall_at_k:>9.1%}  {self.resource_recall_at_k:>9.1%}  "
            f"{self.rejection_rate:>9.1%}  {self.answerable_empty:>7d}  "
            f"{self.mean_results:>6.2f}  {self.combined:>8.3f}"
        )


HEADER = (
    f"{'min':>6}  {'top-1 cat':>9}  {'recall@3':>9}  {'res@3':>9}  "
    f"{'reject':>9}  {'lost':>7}  {'avg n':>6}  {'combined':>8}"
)


def load_eval_queries(path: Path | str) -> list[EvalQuery]:
    """Read the evaluation set. Accepts a list or a ``queries`` object."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    raw: Sequence[dict[str, Any]] = document["queries"] if isinstance(document, dict) else document

    queries = []
    for entry in raw:
        queries.append(
            EvalQuery(
                query_id=entry["query_id"],
                query=entry["query"],
                location=entry.get("location"),
                expected_categories=frozenset(
                    ServiceCategory(value) for value in entry.get("expected_categories", [])
                ),
                expected_resource_ids=frozenset(entry.get("expected_resource_ids", [])),
                off_topic=bool(entry.get("off_topic", False)),
                note=entry.get("note", ""),
            )
        )
    return queries


async def collect_outcomes(
    retriever: ResourceRetriever,
    queries: Sequence[EvalQuery],
    *,
    top_k: int = 10,
) -> list[QueryOutcome]:
    """Run every query once with the relevance gate wide open.

    The retriever passed in must have ``min_score=0`` for the sweep to be able
    to explore thresholds above it.
    """
    outcomes = []
    for query in queries:
        results = await retriever.retrieve(query.query, location=query.location, top_k=top_k)
        outcomes.append(QueryOutcome(query=query, results=results))
    return outcomes


def score_at_threshold(outcomes: Sequence[QueryOutcome], threshold: float) -> Metrics:
    """Compute the headline measures at one relevance gate."""
    answerable = [outcome for outcome in outcomes if not outcome.query.off_topic]
    off_topic = [outcome for outcome in outcomes if outcome.query.off_topic]

    top1_hits = 0
    recall_hits = 0
    resource_hits = 0
    resource_labelled = 0
    empty = 0
    total_results = 0

    for outcome in answerable:
        kept = outcome.above(threshold)
        total_results += len(kept)
        if not kept:
            empty += 1
            if outcome.query.expected_resource_ids:
                resource_labelled += 1
            continue

        expected = outcome.query.expected_categories
        if expected & set(kept[0].resource.service_categories):
            top1_hits += 1
        if any(expected & set(result.resource.service_categories) for result in kept[:RECALL_AT]):
            recall_hits += 1

        if outcome.query.expected_resource_ids:
            resource_labelled += 1
            found = {result.resource.resource_id for result in kept[:RECALL_AT]}
            if found & outcome.query.expected_resource_ids:
                resource_hits += 1

    rejected = sum(1 for outcome in off_topic if not outcome.above(threshold))

    return Metrics(
        threshold=threshold,
        answerable=len(answerable),
        off_topic=len(off_topic),
        top1_category_accuracy=_ratio(top1_hits, len(answerable)),
        category_recall_at_k=_ratio(recall_hits, len(answerable)),
        resource_recall_at_k=_ratio(resource_hits, resource_labelled),
        resource_labelled=resource_labelled,
        rejection_rate=_ratio(rejected, len(off_topic)),
        answerable_empty=empty,
        mean_results=_ratio(total_results, len(answerable)),
    )


def sweep(
    outcomes: Sequence[QueryOutcome],
    thresholds: Sequence[float],
) -> list[Metrics]:
    return [score_at_threshold(outcomes, threshold) for threshold in thresholds]


def recommend_threshold(metrics: Sequence[Metrics], *, tolerance: float = 1e-6) -> Metrics:
    """Pick the middle of the widest run of best-scoring thresholds.

    The best score is usually a plateau rather than a point, and both of its
    edges are cliffs: a step below and off-topic queries start getting answered,
    a step above and real needs start returning nothing. Taking an edge would
    leave the configuration one small data change away from a cliff, so the
    midpoint of the longest optimal run is chosen instead.
    """
    if not metrics:
        raise ValueError("no metrics to choose from")

    best_score = max(item.combined for item in metrics)
    runs: list[list[Metrics]] = []
    current: list[Metrics] = []

    for item in sorted(metrics, key=lambda entry: entry.threshold):
        if abs(item.combined - best_score) <= tolerance:
            current.append(item)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    widest = max(runs, key=len)
    return widest[len(widest) // 2]


def candidate_thresholds(outcomes: Sequence[QueryOutcome], *, steps: int = 40) -> list[float]:
    """Threshold grid spanning the observed score range."""
    scores = [result.semantic_score for outcome in outcomes for result in outcome.results]
    if not scores:
        return [0.0]
    low, high = min(scores), max(scores)
    if high <= low:
        return [round(low, 4)]
    step = (high - low) / steps
    return [round(low + step * index, 4) for index in range(steps + 1)]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
