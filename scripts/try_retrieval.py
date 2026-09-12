#!/usr/bin/env python
"""Try retrieval queries against the resource index.

Ingests the configured resource file, then runs one or more needs through the
retriever and prints the ranked results. Retrieval only — nothing is sent to a
generation model.

Run the built-in example queries:

    python scripts/try_retrieval.py

Ask something specific:

    python scripts/try_retrieval.py "I need help paying rent" --location Oakland
    python scripts/try_retrieval.py "wheelchair ramp" --top-k 3 --category accessibility_support

Configuration comes from the environment (or ``.env``). By default
``EMBEDDING_PROVIDER=hashing`` keeps everything offline; set
``EMBEDDING_PROVIDER=vertex`` to use real semantic embeddings via ADC.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap
from pathlib import Path

# Allow running directly from a checkout where the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings
from app.domain.resource import ServiceCategory
from app.services.retrieval.bootstrap import bootstrap_retrieval
from app.services.retrieval.retriever import RetrievedResource

EXAMPLE_QUERIES: list[tuple[str, str | None]] = [
    ("I need transportation to my doctor appointment.", None),
    ("I need food assistance near Oakland.", "Oakland"),
    ("My mother needs wheelchair-accessible support.", None),
]

_RULE = "─" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semantic retrieval against the verified resource index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="The need to search for. Omit to run the built-in example queries.",
    )
    parser.add_argument("--location", help="ZIP code, city, or county.")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        choices=[category.value for category in ServiceCategory],
        help="Restrict to a service category. Repeatable.",
    )
    parser.add_argument("--language", action="append", dest="languages", help="Repeatable.")
    parser.add_argument("--top-k", type=int, default=None, help="How many results to show.")
    return parser.parse_args()


def print_result(result: RetrievedResource) -> None:
    resource = result.resource
    location_note = (
        f"  location match {result.location_match:.2f}" if result.matched_location else ""
    )
    print(
        f"  {result.rank}. {resource.organization_name}  "
        f"[score {result.score:.3f}  semantic {result.semantic_score:.3f}{location_note}]"
    )
    print(f"     categories : {', '.join(c.label for c in resource.service_categories)}")
    print(f"     serves     : {resource.service_area.describe()}")
    if resource.languages:
        print(f"     languages  : {', '.join(resource.languages)}")
    if resource.accessibility:
        print(f"     access     : {', '.join(resource.accessibility)}")
    contact = " | ".join(part for part in (resource.phone, resource.website) if part)
    if contact:
        print(f"     contact    : {contact}")
    print(f"     verified   : {resource.last_verified.isoformat()} ({resource.source})")
    summary = textwrap.shorten(resource.description, width=150, placeholder="…")
    print(f"     {summary}")
    print()


async def main() -> int:
    args = parse_args()
    settings = Settings()

    print(_RULE)
    print("Health Equity Navigator — retrieval only (no generation)")
    print(f"  embedding provider : {settings.embedding_provider.value}")
    print(f"  resource file      : {settings.resource_data_path}")
    print(_RULE)

    stack, report = await bootstrap_retrieval(settings)
    print(f"Ingestion: {report.summary()}")
    for issue in report.issues:
        print(f"  rejected [{issue.index}] {issue.resource_id or '<no id>'}: {issue.reason}")
    print(_RULE)

    if args.query:
        queries = [(args.query, args.location)]
    else:
        queries = EXAMPLE_QUERIES

    categories = [ServiceCategory(value) for value in args.categories] if args.categories else None

    try:
        for query, location in queries:
            suffix = f"   (location: {location})" if location else ""
            print(f'\nQUERY: "{query}"{suffix}')
            print()

            results = await stack.retriever.retrieve(
                query,
                location=location,
                categories=categories,
                languages=args.languages,
                top_k=args.top_k,
            )

            if not results:
                print("  No verified resources matched this need.\n")
                continue
            for result in results:
                print_result(result)
    finally:
        await stack.aclose()

    print(_RULE)
    print("Retrieval only. Results are NOT sent to Gemini in this phase.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
