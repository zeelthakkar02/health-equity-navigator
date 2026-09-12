"""End-to-end evaluation of the Navigator.

Phase 2's harness measured retrieval alone. This one runs the whole service —
crisis screen, retrieval, generation, grounding validation, escalation — against
deliberately messy input, and reports what a reviewer actually needs to know
before putting this in front of the public.

The strictest measure here is **leak rate**. Every generated answer that is
returned is re-validated against the resources the request actually retrieved,
and any case-specific forbidden string (an injected organization, a fabricated
phone number, a withheld decoy) is searched for in the final text. Leak rate is
the one number that must be zero; the rest are quality measures that trade off.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.metrics import RequestMetrics
from app.domain.resource import Resource, ServiceCategory
from app.schemas.navigator import AnswerSource, NavigatorQueryRequest, NavigatorQueryResponse
from app.services.grounding import GroundingValidator
from app.services.navigator_service import NavigatorService

# Cases whose answer legitimately comes from a fixed script rather than a model.
_SCRIPTED_SOURCES = frozenset({AnswerSource.NO_MATCH, AnswerSource.SAFETY_NOTICE})


@dataclass(frozen=True)
class EvalCase:
    """One labelled end-to-end case."""

    case_id: str
    kind: str
    query: str
    location: str | None = None
    language: str = "en"
    # None means "either outcome is defensible" — a vague query may reasonably
    # return something or nothing, and pinning it would measure taste, not quality.
    expect_answerable: bool | None = None
    expected_categories: frozenset[ServiceCategory] = frozenset()
    expect_escalation: bool | None = None
    must_not_appear: tuple[str, ...] = ()
    must_appear: tuple[str, ...] = ()


@dataclass
class CaseOutcome:
    """What happened for one case."""

    case: EvalCase
    response: NavigatorQueryResponse
    metrics: RequestMetrics | None = None
    leaks: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return bool(self.response.resources)

    @property
    def generated(self) -> bool:
        return self.response.answer_source is AnswerSource.GENERATED

    @property
    def cited_ids(self) -> set[str]:
        return {citation.resource_id for citation in self.response.resources}

    @property
    def top_categories(self) -> set[ServiceCategory]:
        if not self.response.resources:
            return set()
        return {ServiceCategory(value) for value in self.response.resources[0].categories}

    @property
    def retrieval_correct(self) -> bool | None:
        """None when the case does not label expected categories."""
        if not self.case.expected_categories:
            return None
        return bool(self.case.expected_categories & self.top_categories)

    @property
    def escalation_correct(self) -> bool | None:
        if self.case.expect_escalation is None:
            return None
        return self.response.needs_escalation == self.case.expect_escalation

    @property
    def answerability_correct(self) -> bool | None:
        if self.case.expect_answerable is None:
            return None
        return self.answered == self.case.expect_answerable


@dataclass
class EvalReport:
    """Aggregate results across every case."""

    outcomes: list[CaseOutcome]
    forbidden_ids: frozenset[str] = frozenset()

    # --- headline measures ---

    @property
    def retrieval_accuracy(self) -> float:
        judged = [o.retrieval_correct for o in self.outcomes if o.retrieval_correct is not None]
        return _ratio(sum(judged), len(judged))

    @property
    def grounded_answer_rate(self) -> float:
        """Of cases that reached the model, how many produced a usable answer."""
        reached = [o for o in self.outcomes if o.response.answer_source not in _SCRIPTED_SOURCES]
        return _ratio(sum(1 for o in reached if o.generated), len(reached))

    @property
    def leak_rate(self) -> float:
        """Answers that carried content they should not have. Must be zero."""
        return _ratio(sum(1 for o in self.outcomes if o.leaks), len(self.outcomes))

    @property
    def hallucination_rejection_rate(self) -> float:
        """Share of adversarial and decoy cases that kept forbidden content out."""
        probes = [o for o in self.outcomes if o.case.must_not_appear]
        return _ratio(sum(1 for o in probes if not o.leaks), len(probes))

    @property
    def no_match_accuracy(self) -> float:
        judged = [
            o.answerability_correct for o in self.outcomes if o.answerability_correct is not None
        ]
        return _ratio(sum(judged), len(judged))

    @property
    def escalation_accuracy(self) -> float:
        judged = [o.escalation_correct for o in self.outcomes if o.escalation_correct is not None]
        return _ratio(sum(judged), len(judged))

    @property
    def required_content_rate(self) -> float:
        probes = [o for o in self.outcomes if o.case.must_appear]
        return _ratio(sum(1 for o in probes if not o.missing), len(probes))

    @property
    def decoy_citations(self) -> list[str]:
        """Withheld records that reached a response. Must be empty."""
        return sorted(
            {
                resource_id
                for outcome in self.outcomes
                for resource_id in outcome.cited_ids & self.forbidden_ids
            }
        )

    # --- latency ---

    def latencies(self, attribute: str = "total_ms") -> list[int]:
        return [
            getattr(outcome.metrics, attribute)
            for outcome in self.outcomes
            if outcome.metrics is not None
        ]

    def latency_summary(self, attribute: str = "total_ms") -> dict[str, float]:
        values = sorted(self.latencies(attribute))
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "max": float(values[-1]),
        }

    def by_kind(self) -> dict[str, list[CaseOutcome]]:
        grouped: dict[str, list[CaseOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.case.kind, []).append(outcome)
        return grouped

    @property
    def failures(self) -> list[CaseOutcome]:
        """Cases that got something demonstrably wrong."""
        return [
            outcome
            for outcome in self.outcomes
            if outcome.leaks
            or outcome.missing
            or outcome.retrieval_correct is False
            or outcome.escalation_correct is False
            or outcome.answerability_correct is False
        ]


def load_eval_cases(path: Path | str) -> list[EvalCase]:
    """Read the end-to-end case set."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    raw: Sequence[dict[str, Any]] = document["cases"] if isinstance(document, dict) else document

    return [
        EvalCase(
            case_id=entry["case_id"],
            kind=entry.get("kind", "unspecified"),
            query=entry["query"],
            location=entry.get("location"),
            language=entry.get("language", "en"),
            expect_answerable=entry.get("expect_answerable"),
            expected_categories=frozenset(
                ServiceCategory(value) for value in entry.get("expected_categories", [])
            ),
            expect_escalation=entry.get("expect_escalation"),
            must_not_appear=tuple(entry.get("must_not_appear", [])),
            must_appear=tuple(entry.get("must_appear", [])),
        )
        for entry in raw
    ]


def check_case(
    case: EvalCase,
    response: NavigatorQueryResponse,
    resources_by_id: dict[str, Resource],
) -> tuple[list[str], list[str]]:
    """Find content that leaked into the answer, and required content that did not.

    Two independent checks: the case's own forbidden strings, and an
    re-validation of the returned text against the resources actually cited.
    Re-validating catches anything the service should have rejected but didn't.
    """
    answer = response.answer
    lowered = answer.lower()

    leaks = [phrase for phrase in case.must_not_appear if phrase.lower() in lowered]
    missing = [phrase for phrase in case.must_appear if phrase.lower() not in lowered]

    if response.answer_source is AnswerSource.GENERATED:
        cited = [
            resources_by_id[citation.resource_id]
            for citation in response.resources
            if citation.resource_id in resources_by_id
        ]
        result = GroundingValidator.for_resources(cited).validate(answer)
        leaks.extend(f"ungrounded {issue.kind}: {issue.detail}" for issue in result.issues)

    return leaks, missing


async def run_evaluation(
    navigator: NavigatorService,
    cases: Sequence[EvalCase],
    resources_by_id: dict[str, Resource],
    *,
    forbidden_ids: frozenset[str] = frozenset(),
    metrics_by_request: dict[str, RequestMetrics] | None = None,
    max_resources: int = 4,
    progress: bool = False,
) -> EvalReport:
    """Run every case through the Navigator and score the results."""
    outcomes: list[CaseOutcome] = []

    for index, case in enumerate(cases, start=1):
        if progress:
            print(f"  [{index}/{len(cases)}] {case.case_id}", flush=True)

        started = time.perf_counter()
        response = await navigator.answer(
            NavigatorQueryRequest(
                query=case.query,
                location=case.location,
                language=case.language,
                max_resources=max_resources,
            )
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        metrics = (metrics_by_request or {}).get(response.request_id)
        if metrics is None:
            metrics = RequestMetrics(request_id=response.request_id, total_ms=elapsed_ms)

        leaks, missing = check_case(case, response, resources_by_id)
        outcomes.append(
            CaseOutcome(case=case, response=response, metrics=metrics, leaks=leaks, missing=missing)
        )

    return EvalReport(outcomes=outcomes, forbidden_ids=forbidden_ids)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(sorted_values: list[int], fraction: float) -> float:
    """Nearest-rank percentile; exact and obvious on small samples."""
    if not sorted_values:
        return 0.0
    rank = max(1, min(len(sorted_values), round(fraction * len(sorted_values) + 0.5)))
    return float(sorted_values[rank - 1])
