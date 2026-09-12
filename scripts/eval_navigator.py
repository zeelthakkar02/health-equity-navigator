#!/usr/bin/env python
"""Run the end-to-end Navigator evaluation.

Builds an index from the synthetic resources **plus** the decoy records that are
stale or unverified, then runs every labelled case through the full service and
reports retrieval accuracy, grounded-answer rate, leak rate, no-match accuracy,
escalation accuracy, and latency.

    python scripts/eval_navigator.py                          # offline
    EMBEDDING_PROVIDER=vertex LLM_PROVIDER=vertex \\
        python scripts/eval_navigator.py --failures           # live

    GEMINI_THINKING_LEVEL=low python scripts/eval_navigator.py --failures

Synthetic data only. Nothing here is a real person, organization, or message.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running directly from a checkout where the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings
from app.core.metrics import RequestMetrics
from app.domain.resource import Resource
from app.services.ingestion.loader import load_resources_from_file
from app.services.llm.factory import build_llm_service
from app.services.navigator_evaluation import (
    EvalReport,
    load_eval_cases,
    run_evaluation,
)
from app.services.navigator_service import NavigatorService
from app.services.retrieval.bootstrap import bootstrap_retrieval

RULE = "═" * 92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Navigator end to end.")
    parser.add_argument("--failures", action="store_true", help="Show every failing case.")
    parser.add_argument("--kind", help="Run only one kind of case (e.g. adversarial).")
    parser.add_argument("--json", dest="json_out", help="Write the summary to this path.")
    parser.add_argument("--progress", action="store_true", help="Print each case as it runs.")
    return parser.parse_args()


def merged_resource_file(settings: Settings) -> tuple[Path, dict[str, Resource], frozenset[str]]:
    """Write sample + decoy resources to one file the normal bootstrap can load.

    Decoys are stale or unverified and must be filtered out by retrieval policy.
    Putting them in the same index is the point: it proves the filter holds in
    the real pipeline rather than only in a unit test.
    """
    sample = load_resources_from_file(settings.resource_data_path)
    decoys = load_resources_from_file(settings.eval_decoy_resources_path)

    merged = sample.resources + decoys.resources
    destination = Path(".cache/eval_resources.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "_meta": {"generated_by": "scripts/eval_navigator.py", "synthetic": True},
                "resources": [resource.model_dump(mode="json") for resource in merged],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    by_id = {resource.resource_id: resource for resource in merged}
    forbidden = frozenset(resource.resource_id for resource in decoys.resources)
    return destination, by_id, forbidden


def print_report(report: EvalReport, settings: Settings) -> None:
    print(f"\n{RULE}")
    print("HEADLINE MEASURES")
    print(RULE)
    print(f"  retrieval accuracy (top-1 category) : {report.retrieval_accuracy:.1%}")
    print(f"  grounded-answer rate                : {report.grounded_answer_rate:.1%}")
    print(f"  hallucination rejection rate        : {report.hallucination_rejection_rate:.1%}")
    print(f"  no-match accuracy                   : {report.no_match_accuracy:.1%}")
    print(f"  escalation accuracy                 : {report.escalation_accuracy:.1%}")
    print(f"  required-content rate               : {report.required_content_rate:.1%}")
    print(f"  LEAK RATE (must be 0.0%)            : {report.leak_rate:.1%}")
    decoys = report.decoy_citations
    print(
        f"  withheld records cited (must be 0)  : {len(decoys)}" + (f"  {decoys}" if decoys else "")
    )

    print(f"\n{RULE}")
    print("LATENCY (ms)")
    print(RULE)
    print(f"  {'stage':<14}{'count':>7}{'mean':>10}{'p50':>10}{'p95':>10}{'max':>10}")
    for label, attribute in (
        ("total", "total_ms"),
        ("retrieval", "retrieval_ms"),
        ("generation", "llm_ms"),
        ("grounding", "grounding_ms"),
    ):
        summary = report.latency_summary(attribute)
        print(
            f"  {label:<14}{summary['count']:>7.0f}{summary['mean']:>10.0f}"
            f"{summary['p50']:>10.0f}{summary['p95']:>10.0f}{summary['max']:>10.0f}"
        )

    print(f"\n{RULE}")
    print("BY CASE KIND")
    print(RULE)
    print(
        f"  {'kind':<20}{'cases':>7}{'answered':>10}{'generated':>11}{'leaks':>8}{'failures':>10}"
    )
    for kind, outcomes in sorted(report.by_kind().items()):
        answered = sum(1 for outcome in outcomes if outcome.answered)
        generated = sum(1 for outcome in outcomes if outcome.generated)
        leaks = sum(1 for outcome in outcomes if outcome.leaks)
        failures = sum(1 for outcome in outcomes if outcome in report.failures)
        print(
            f"  {kind:<20}{len(outcomes):>7}{answered:>10}{generated:>11}{leaks:>8}{failures:>10}"
        )

    print(f"\n  model: {settings.gemini_model}", end="")
    print(f"  thinking_level: {settings.gemini_thinking_level or '(model default)'}")


def print_failures(report: EvalReport) -> None:
    print(f"\n{RULE}")
    print("FAILING CASES")
    print(RULE)
    if not report.failures:
        print("  none")
        return

    for outcome in report.failures:
        case = outcome.case
        print(f"\n  [{case.case_id}] {case.kind}")
        print(f"    query   : {case.query[:100]!r}")
        print(
            f"    got     : {outcome.response.answer_source.value}"
            f"  escalation={outcome.response.needs_escalation}"
            f"  resources={len(outcome.response.resources)}"
        )
        if outcome.leaks:
            print(f"    LEAKED  : {outcome.leaks}")
        if outcome.missing:
            print(f"    MISSING : {outcome.missing}")
        if outcome.retrieval_correct is False:
            expected = sorted(c.value for c in case.expected_categories)
            actual = sorted(c.value for c in outcome.top_categories)
            print(f"    top-1   : expected one of {expected}, got {actual}")
        if outcome.answerability_correct is False:
            print(f"    answerable: expected {case.expect_answerable}, got {outcome.answered}")
        if outcome.escalation_correct is False:
            print(f"    escalation: expected {case.expect_escalation}")


async def main() -> int:
    args = parse_args()
    settings = Settings()

    print(RULE)
    print("NAVIGATOR END-TO-END EVALUATION (synthetic data only)")
    print(f"  embeddings : {settings.embedding_provider.value}")
    print(f"  generation : {settings.llm_provider.value} / {settings.gemini_model}")
    print(f"  thinking   : {settings.gemini_thinking_level or '(model default)'}")
    print(RULE)

    resource_file, resources_by_id, forbidden_ids = merged_resource_file(settings)
    eval_settings = settings.model_copy(
        update={
            "resource_data_path": resource_file,
            "resource_index_cache_path": Path(".cache/eval_index.npz"),
        }
    )

    stack, report_in = await bootstrap_retrieval(eval_settings)
    print(f"Index: {report_in.summary()}")
    print(f"  withheld decoys in the index: {len(forbidden_ids)}")

    cases = load_eval_cases(settings.eval_navigator_path)
    if args.kind:
        cases = [case for case in cases if case.kind == args.kind]
    print(f"  cases: {len(cases)}")

    collected: dict[str, RequestMetrics] = {}
    llm = build_llm_service(eval_settings)
    navigator = NavigatorService(
        llm,
        eval_settings,
        retriever=stack.retriever,
        metrics_sink=lambda record: collected.setdefault(record.request_id, record),
    )

    try:
        report = await run_evaluation(
            navigator,
            cases,
            resources_by_id,
            forbidden_ids=forbidden_ids,
            metrics_by_request=collected,
            progress=args.progress,
        )
    finally:
        await llm.aclose()
        await stack.aclose()

    print_report(report, settings)
    if args.failures:
        print_failures(report)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "model": settings.gemini_model,
                    "thinking_level": settings.gemini_thinking_level,
                    "embedding_provider": settings.embedding_provider.value,
                    "llm_provider": settings.llm_provider.value,
                    "cases": len(report.outcomes),
                    "retrieval_accuracy": report.retrieval_accuracy,
                    "grounded_answer_rate": report.grounded_answer_rate,
                    "hallucination_rejection_rate": report.hallucination_rejection_rate,
                    "no_match_accuracy": report.no_match_accuracy,
                    "escalation_accuracy": report.escalation_accuracy,
                    "leak_rate": report.leak_rate,
                    "decoy_citations": report.decoy_citations,
                    "latency_total_ms": report.latency_summary("total_ms"),
                    "latency_llm_ms": report.latency_summary("llm_ms"),
                    "latency_retrieval_ms": report.latency_summary("retrieval_ms"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  summary written to {args.json_out}")

    print(f"\n{RULE}")
    failed = report.leak_rate > 0 or report.decoy_citations
    print("RESULT: LEAKS DETECTED" if failed else "RESULT: no leaks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
