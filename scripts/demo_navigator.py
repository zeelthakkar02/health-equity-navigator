#!/usr/bin/env python
"""End-to-end Navigator demo: retrieval → Gemini → grounded, cited answer.

Runs the full pipeline in process — the same NavigatorService the API uses — and
prints the answer, its citations, and how it was produced.

    python scripts/demo_navigator.py                       # offline: hashing + stub
    EMBEDDING_PROVIDER=vertex LLM_PROVIDER=vertex \\
        python scripts/demo_navigator.py                   # live Vertex + Gemini

    python scripts/demo_navigator.py "I need help paying rent" --location Oakland

Synthetic resources only (app/data/sample_resources.json). No real organization,
and no private data of any kind.
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
from app.schemas.navigator import (
    AnswerSource,
    NavigatorQueryRequest,
    NavigatorQueryResponse,
)
from app.services.llm.factory import build_llm_service
from app.services.navigator_service import NavigatorService
from app.services.retrieval.bootstrap import bootstrap_retrieval

RULE = "═" * 90
THIN = "─" * 90

# Chosen to exercise every branch: a plain need, a location-scoped need, the
# accessibility case, a multi-resource need, an off-topic question, and a
# message that must never reach a model.
DEMO_QUERIES: list[tuple[str, str | None]] = [
    ("I need transportation to my dialysis appointment.", None),
    ("I need food assistance near Oakland.", "Oakland"),
    ("My mother needs wheelchair-accessible support.", None),
    ("I got a huge hospital bill and I can't pay it.", None),
    ("I am Deaf and need an ASL interpreter at my appointment.", None),
    ("Who won the football game last night?", None),
    ("I want to kill myself.", None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Navigator end to end.")
    parser.add_argument("query", nargs="?", help="Ask one thing instead of the demo set.")
    parser.add_argument("--location", help="ZIP code, city, or county.")
    parser.add_argument("--max-resources", type=int, default=4)
    return parser.parse_args()


def print_response(response: NavigatorQueryResponse) -> None:
    badge = {
        AnswerSource.GENERATED: "GENERATED (grounded + validated)",
        AnswerSource.VERIFIED_LISTING: "VERIFIED LISTING (no model text used)",
        AnswerSource.NO_MATCH: "NO VERIFIED MATCH (model never called)",
        AnswerSource.SAFETY_NOTICE: "SAFETY NOTICE (model never called)",
    }[response.answer_source]

    print(f"  source        : {badge}")
    print(f"  provider/model: {response.provider} / {response.model}")
    print(f"  escalation    : {response.needs_escalation}", end="")
    print(f"  ({response.escalation_reason.value})" if response.escalation_reason else "")
    print(f"  latency       : {response.latency_ms} ms")
    print()

    for line in response.answer.splitlines():
        print(textwrap.fill(line, width=86, initial_indent="  ", subsequent_indent="  ") or "")
    print()

    if response.resources:
        print(f"  CITED RESOURCES ({len(response.resources)}) — from stored records only")
        for index, citation in enumerate(response.resources, start=1):
            print(f"    [{index}] {citation.title}  ({', '.join(citation.categories)})")
            contact = " | ".join(part for part in (citation.phone, citation.url) if part)
            if contact:
                print(f"        {contact}")
            print(f"        serves {citation.service_area} · verified {citation.last_verified}")
    else:
        print("  CITED RESOURCES: none")


async def main() -> int:
    args = parse_args()
    settings = Settings()

    print(RULE)
    print("HEALTH EQUITY NAVIGATOR — end-to-end demo (synthetic resources only)")
    print(f"  embeddings : {settings.embedding_provider.value}")
    print(f"  generation : {settings.llm_provider.value} / {settings.gemini_model}")
    print(RULE)

    stack, report = await bootstrap_retrieval(settings)
    print(f"Retrieval ready: {report.summary()}")

    llm = build_llm_service(settings)
    navigator = NavigatorService(llm, settings, retriever=stack.retriever)

    queries = [(args.query, args.location)] if args.query else DEMO_QUERIES

    try:
        for query, location in queries:
            print(f"\n{THIN}")
            suffix = f"   (location: {location})" if location else ""
            print(f'QUERY: "{query}"{suffix}')
            print(THIN)

            response = await navigator.answer(
                NavigatorQueryRequest(
                    query=query, location=location, max_resources=args.max_resources
                )
            )
            print_response(response)
    finally:
        await llm.aclose()
        await stack.aclose()

    print(f"\n{RULE}")
    print("Every cited organization comes from a stored record. Generated text that")
    print("names anything outside its context is discarded, never returned.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
