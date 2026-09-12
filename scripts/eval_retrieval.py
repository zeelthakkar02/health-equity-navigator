#!/usr/bin/env python
"""Evaluate retrieval quality and tune the relevance threshold.

Runs the labelled query set through the retriever with the gate wide open, then
sweeps candidate thresholds over those cached results — one embedding pass, not
one per threshold.

    python scripts/eval_retrieval.py                        # offline, hashing
    EMBEDDING_PROVIDER=vertex python scripts/eval_retrieval.py

Options:

    --failures    show every query the current threshold gets wrong
    --sweep       print the full threshold sweep table

Retrieval only — nothing is sent to a generation model.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running directly from a checkout where the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings
from app.services.retrieval.bootstrap import (
    DEFAULT_MIN_SCORE,
    bootstrap_retrieval,
    resolve_min_score,
)
from app.services.retrieval.evaluation import (
    HEADER,
    Metrics,
    QueryOutcome,
    candidate_thresholds,
    collect_outcomes,
    load_eval_queries,
    recommend_threshold,
    score_at_threshold,
    sweep,
)
from app.services.retrieval.need_lexicon import detect_categories

RULE = "─" * 92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--failures", action="store_true", help="List incorrect queries.")
    parser.add_argument("--sweep", action="store_true", help="Print the full sweep table.")
    parser.add_argument(
        "--category-boost",
        type=float,
        default=None,
        help="Override RETRIEVAL_CATEGORY_BOOST (use 0 to measure its effect).",
    )
    return parser.parse_args()


def print_metrics(label: str, metrics: Metrics) -> None:
    print(f"\n{label}")
    print(f"  relevance gate          : {metrics.threshold:.3f}")
    print(f"  top-1 category accuracy : {metrics.top1_category_accuracy:.1%}")
    print(f"  category recall@3       : {metrics.category_recall_at_k:.1%}")
    print(
        f"  resource recall@3       : {metrics.resource_recall_at_k:.1%}  "
        f"({metrics.resource_labelled} queries name a specific resource)"
    )
    print(
        f"  off-topic rejection     : {metrics.rejection_rate:.1%}  "
        f"({metrics.off_topic} off-topic queries)"
    )
    print(f"  answerable left empty   : {metrics.answerable_empty}/{metrics.answerable}")
    print(f"  mean results returned   : {metrics.mean_results:.2f}")


def print_failures(outcomes: list[QueryOutcome], threshold: float) -> None:
    print(f"\n{RULE}")
    print(f"FAILURES AT GATE {threshold:.3f}")
    print(RULE)
    any_failure = False

    for outcome in outcomes:
        query = outcome.query
        kept = outcome.above(threshold)

        if query.off_topic:
            if kept:
                any_failure = True
                top = kept[0]
                print(f"\n  [{query.query_id}] off-topic query was NOT rejected")
                print(f"    {query.query!r}")
                print(
                    f"    top hit: {top.resource.organization_name} "
                    f"(semantic {top.semantic_score:.3f})"
                )
            continue

        expected = query.expected_categories
        if not kept:
            any_failure = True
            print(f"\n  [{query.query_id}] returned nothing")
            print(f"    {query.query!r}")
            continue

        top_categories = set(kept[0].resource.service_categories)
        if expected & top_categories:
            continue

        any_failure = True
        stated = sorted(category.value for category in detect_categories(query.query))
        print(f"\n  [{query.query_id}] wrong top-1 category")
        print(f"    {query.query!r}")
        print(f"    expected : {sorted(c.value for c in expected)}")
        print(f"    got      : {kept[0].resource.organization_name}")
        print(f"               {sorted(c.value for c in top_categories)}")
        print(f"    stated needs detected: {stated or '(none)'}")

    if not any_failure:
        print("\n  none")


async def main() -> int:
    args = parse_args()
    settings = Settings()

    print(RULE)
    print("RETRIEVAL EVALUATION")
    print(f"  embedding provider : {settings.embedding_provider.value}")
    print(f"  resources          : {settings.resource_data_path.name}")
    print(f"  eval queries       : {settings.eval_queries_path.name}")
    print(RULE)

    queries = load_eval_queries(settings.eval_queries_path)
    answerable = sum(1 for query in queries if not query.off_topic)
    print(
        f"  {len(queries)} queries ({answerable} answerable, {len(queries) - answerable} off-topic)"
    )

    # Gate wide open so the sweep can explore every threshold above it.
    overrides: dict[str, object] = {"retrieval_min_score": 0.0}
    if args.category_boost is not None:
        overrides["retrieval_category_boost"] = args.category_boost
        print(f"  category boost     : {args.category_boost} (overridden)")
    open_gate = settings.model_copy(update=overrides)
    stack, report = await bootstrap_retrieval(open_gate)
    print(f"  ingestion: {report.summary()}")

    try:
        outcomes = await collect_outcomes(stack.retriever, queries, top_k=10)
    finally:
        await stack.aclose()

    configured = resolve_min_score(settings)
    print_metrics(
        f"CURRENT DEFAULT for {settings.embedding_provider.value}",
        score_at_threshold(outcomes, configured),
    )

    grid = candidate_thresholds(outcomes)
    results = sweep(outcomes, grid)
    best = recommend_threshold(results)

    if args.sweep:
        print(f"\n{RULE}")
        print("THRESHOLD SWEEP")
        print(RULE)
        print(HEADER)
        for metrics in results:
            marker = "  <- best" if metrics.threshold == best.threshold else ""
            print(metrics.as_row() + marker)

    print_metrics("RECOMMENDED", best)
    print(
        f"\n  recommended RETRIEVAL_MIN_SCORE for "
        f"{settings.embedding_provider.value}: {best.threshold:.2f}"
    )
    print(f"  currently compiled in : {DEFAULT_MIN_SCORE[settings.embedding_provider]:.2f}")

    if args.failures:
        print_failures(outcomes, configured)

    print(f"\n{RULE}")
    print("Retrieval only. Nothing was sent to a generation model.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
