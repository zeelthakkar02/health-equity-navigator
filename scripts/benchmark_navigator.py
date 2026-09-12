#!/usr/bin/env python
"""Benchmark Navigator latency, and whether a cheaper configuration holds up.

Runs the same queries under one or more model configurations and reports latency
percentiles beside quality signals, so a decision about thinking level or model
is made on measurements rather than on the assumption that faster is worse.

    EMBEDDING_PROVIDER=vertex LLM_PROVIDER=vertex \\
        python scripts/benchmark_navigator.py --repeat 3

    ... python scripts/benchmark_navigator.py --thinking-levels default,low

Quality here is coarse on purpose — grounded rate, answer length, citation
count. Use scripts/eval_navigator.py for the full labelled measurement of a
configuration that looks promising.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running directly from a checkout where the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings
from app.core.metrics import RequestMetrics
from app.schemas.navigator import AnswerSource, NavigatorQueryRequest
from app.services.llm.factory import build_llm_service
from app.services.navigator_service import NavigatorService
from app.services.retrieval.bootstrap import bootstrap_retrieval

RULE = "═" * 96

BENCHMARK_QUERIES: list[tuple[str, str | None]] = [
    ("I need transportation to my dialysis appointment.", None),
    ("I need food assistance near Oakland.", "Oakland"),
    ("My mother needs wheelchair-accessible support.", None),
    ("I got a huge hospital bill and I can't pay it.", None),
    ("I am Deaf and need an ASL interpreter at my appointment.", None),
    ("I got an eviction notice and need help.", "Oakland"),
]


@dataclass
class ConfigResult:
    """Latency and quality for one configuration."""

    label: str
    model: str
    thinking_level: str | None
    records: list[RequestMetrics] = field(default_factory=list)
    answer_chars: list[int] = field(default_factory=list)
    citation_counts: list[int] = field(default_factory=list)
    grounded: int = 0
    runs: int = 0

    @property
    def grounded_rate(self) -> float:
        return self.grounded / self.runs if self.runs else 0.0

    def latency(self, attribute: str) -> tuple[float, float, float]:
        values = sorted(getattr(record, attribute) for record in self.records)
        if not values:
            return (0.0, 0.0, 0.0)
        return (
            statistics.fmean(values),
            float(values[len(values) // 2]),
            float(values[max(0, round(0.95 * len(values) + 0.5) - 1)]),
        )

    def mean_chars(self) -> float:
        return statistics.fmean(self.answer_chars) if self.answer_chars else 0.0

    def mean_citations(self) -> float:
        return statistics.fmean(self.citation_counts) if self.citation_counts else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Navigator latency.")
    parser.add_argument("--repeat", type=int, default=2, help="Passes over the query set.")
    parser.add_argument(
        "--thinking-levels",
        default="default",
        help="Comma-separated levels to compare, e.g. 'default,low'. 'default' sends none.",
    )
    parser.add_argument("--models", default="", help="Comma-separated model ids to compare.")
    return parser.parse_args()


async def run_config(
    settings: Settings, label: str, *, repeat: int, warmup: bool = True
) -> ConfigResult:
    result = ConfigResult(
        label=label, model=settings.gemini_model, thinking_level=settings.gemini_thinking_level
    )
    collected: list[RequestMetrics] = []

    stack, _ = await bootstrap_retrieval(settings)
    llm = build_llm_service(settings)
    navigator = NavigatorService(
        llm, settings, retriever=stack.retriever, metrics_sink=collected.append
    )

    try:
        if warmup:
            # First call pays connection setup; measuring it would flatter later runs.
            await navigator.answer(NavigatorQueryRequest(query="I need help finding food."))
            collected.clear()

        for _ in range(repeat):
            for query, location in BENCHMARK_QUERIES:
                response = await navigator.answer(
                    NavigatorQueryRequest(query=query, location=location, max_resources=4)
                )
                result.runs += 1
                result.answer_chars.append(len(response.answer))
                result.citation_counts.append(len(response.resources))
                if response.answer_source is AnswerSource.GENERATED:
                    result.grounded += 1
    finally:
        await llm.aclose()
        await stack.aclose()

    result.records = collected
    return result


def print_results(results: list[ConfigResult]) -> None:
    print(f"\n{RULE}")
    print("LATENCY BY CONFIGURATION (ms)")
    print(RULE)
    header = (
        f"  {'config':<22}{'runs':>6}{'total mean':>12}{'total p50':>11}{'total p95':>11}"
        f"{'llm mean':>10}{'llm p95':>9}{'retr mean':>11}"
    )
    print(header)
    for result in results:
        total_mean, total_p50, total_p95 = result.latency("total_ms")
        llm_mean, _, llm_p95 = result.latency("llm_ms")
        retrieval_mean, _, _ = result.latency("retrieval_ms")
        print(
            f"  {result.label:<22}{result.runs:>6}{total_mean:>12.0f}{total_p50:>11.0f}"
            f"{total_p95:>11.0f}{llm_mean:>10.0f}{llm_p95:>9.0f}{retrieval_mean:>11.0f}"
        )

    print(f"\n{RULE}")
    print("QUALITY SIGNALS")
    print(RULE)
    print(f"  {'config':<22}{'grounded':>10}{'mean chars':>13}{'mean citations':>16}")
    for result in results:
        print(
            f"  {result.label:<22}{result.grounded_rate:>9.1%}"
            f"{result.mean_chars():>13.0f}{result.mean_citations():>16.1f}"
        )

    if len(results) > 1:
        baseline = results[0]
        print(f"\n{RULE}")
        print(f"COMPARED TO {baseline.label}")
        print(RULE)
        base_mean, _, _ = baseline.latency("total_ms")
        for result in results[1:]:
            mean, _, _ = result.latency("total_ms")
            delta = (mean - base_mean) / base_mean * 100 if base_mean else 0.0
            quality = result.grounded_rate - baseline.grounded_rate
            print(
                f"  {result.label:<22} latency {delta:+.1f}%   "
                f"grounded {quality:+.1%}   "
                f"answer length {result.mean_chars() - baseline.mean_chars():+.0f} chars"
            )


async def main() -> int:
    args = parse_args()
    settings = Settings()

    levels = [level.strip() for level in args.thinking_levels.split(",") if level.strip()]
    models = [model.strip() for model in args.models.split(",") if model.strip()] or [
        settings.gemini_model
    ]

    print(RULE)
    print("NAVIGATOR LATENCY BENCHMARK (synthetic resources only)")
    print(f"  embeddings : {settings.embedding_provider.value}")
    print(f"  generation : {settings.llm_provider.value}")
    print(f"  queries    : {len(BENCHMARK_QUERIES)} x {args.repeat} pass(es) per configuration")
    print(RULE)

    results: list[ConfigResult] = []
    for model in models:
        for level in levels:
            thinking = None if level == "default" else level
            label = f"{model}/{level}"
            print(f"\nrunning {label} …", flush=True)
            configured = settings.model_copy(
                update={"gemini_model": model, "gemini_thinking_level": thinking}
            )
            results.append(await run_config(configured, label, repeat=args.repeat))

    print_results(results)
    print(f"\n{RULE}")
    print("Latency depends on prompt size and network conditions; re-measure before")
    print("changing a production default on the strength of one run.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
