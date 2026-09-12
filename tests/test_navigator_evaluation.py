"""The end-to-end evaluation harness.

Scored against synthetic outcomes so the arithmetic and the leak detection are
verified independently of any model.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.metrics import RequestMetrics
from app.domain.resource import ServiceCategory
from app.schemas.navigator import (
    AnswerSource,
    EscalationReason,
    NavigatorQueryResponse,
    ResourceCitation,
)
from app.services.navigator_evaluation import (
    CaseOutcome,
    EvalCase,
    EvalReport,
    check_case,
    load_eval_cases,
)
from app.services.navigator_service import to_citation
from tests.test_navigator_service import FOOD

CITATION = to_citation(FOOD)


def _response(
    answer: str,
    *,
    source: AnswerSource = AnswerSource.GENERATED,
    citations: list[ResourceCitation] | None = None,
    escalation: EscalationReason | None = None,
) -> NavigatorQueryResponse:
    return NavigatorQueryResponse(
        request_id="req-1",
        answer=answer,
        provider="fake",
        model="fake-model",
        resources=citations if citations is not None else [CITATION],
        retrieval_performed=True,
        needs_escalation=escalation is not None,
        escalation_reason=escalation,
        answer_source=source,
        disclaimer="d",
        latency_ms=1,
    )


def _case(**overrides: object) -> EvalCase:
    base: dict[str, object] = {"case_id": "c1", "kind": "typos", "query": "food please"}
    return EvalCase(**(base | overrides))  # type: ignore[arg-type]


def _outcome(case: EvalCase, response: NavigatorQueryResponse) -> CaseOutcome:
    leaks, missing = check_case(case, response, {FOOD.resource_id: FOOD})
    return CaseOutcome(
        case=case,
        response=response,
        metrics=RequestMetrics(request_id="req-1", total_ms=100),
        leaks=leaks,
        missing=missing,
    )


# --- dataset ---------------------------------------------------------------


def test_the_shipped_case_set_loads(settings: Settings) -> None:
    cases = load_eval_cases(settings.eval_navigator_path)

    assert len(cases) >= 40
    kinds = {case.kind for case in cases}
    for required in (
        "typos",
        "vague",
        "multi_need",
        "simple_language",
        "multilingual",
        "adversarial",
        "off_topic",
        "unsupported_claim",
        "decoy_resource",
        "crisis",
    ):
        assert required in kinds, f"missing case kind: {required}"


def test_case_ids_are_unique(settings: Settings) -> None:
    cases = load_eval_cases(settings.eval_navigator_path)
    assert len({case.case_id for case in cases}) == len(cases)


def test_adversarial_cases_all_declare_forbidden_content(settings: Settings) -> None:
    """An injection case that asserts nothing proves nothing."""
    adversarial = [
        c for c in load_eval_cases(settings.eval_navigator_path) if c.kind == "adversarial"
    ]

    assert adversarial
    assert all(case.must_not_appear for case in adversarial)


# --- leak detection --------------------------------------------------------


def test_a_clean_answer_leaks_nothing() -> None:
    case = _case(must_not_appear=("Acme Health Clinic",))
    outcome = _outcome(case, _response("Oakland Community Food Pantry [1] can help."))

    assert outcome.leaks == []


def test_a_forbidden_phrase_is_caught() -> None:
    case = _case(must_not_appear=("Acme Health Clinic",))
    outcome = _outcome(case, _response("Try Acme Health Clinic instead."))

    assert outcome.leaks


def test_forbidden_phrases_are_matched_case_insensitively() -> None:
    case = _case(must_not_appear=("BANANA",))
    outcome = _outcome(case, _response("banana"))

    assert outcome.leaks


def test_a_returned_answer_is_revalidated_against_its_own_citations() -> None:
    """Independent of the service: catches anything it should have rejected."""
    outcome = _outcome(_case(), _response("Call the Riverside Eviction Help Center at 555-0902."))

    assert any("ungrounded" in leak for leak in outcome.leaks)


def test_scripted_answers_are_not_revalidated() -> None:
    """A no-match answer legitimately mentions 211, which is in no resource."""
    outcome = _outcome(
        _case(must_appear=("211",)),
        _response("Call 211 for referrals.", source=AnswerSource.NO_MATCH, citations=[]),
    )

    assert outcome.leaks == []
    assert outcome.missing == []


def test_required_content_is_checked() -> None:
    outcome = _outcome(
        _case(must_appear=("988",)),
        _response("Call 211.", source=AnswerSource.NO_MATCH, citations=[]),
    )

    assert outcome.missing == ["988"]


# --- aggregate measures ----------------------------------------------------


def test_retrieval_accuracy_uses_the_top_citation() -> None:
    right = _outcome(
        _case(expected_categories=frozenset({ServiceCategory.FOOD_ASSISTANCE})),
        _response("ok"),
    )
    wrong = _outcome(
        _case(expected_categories=frozenset({ServiceCategory.HOUSING_SUPPORT})),
        _response("ok"),
    )

    assert EvalReport([right, wrong]).retrieval_accuracy == 0.5


def test_unlabelled_cases_are_excluded_from_accuracy() -> None:
    """A vague query has no single right answer, so it must not be scored."""
    report = EvalReport([_outcome(_case(), _response("ok"))])

    assert report.retrieval_accuracy == 0.0  # nothing judged
    assert report.no_match_accuracy == 0.0


def test_grounded_answer_rate_ignores_scripted_answers() -> None:
    generated = _outcome(_case(), _response("Oakland Community Food Pantry [1]."))
    scripted = _outcome(_case(), _response("No match.", source=AnswerSource.NO_MATCH, citations=[]))

    assert EvalReport([generated, scripted]).grounded_answer_rate == 1.0


def test_leak_rate_counts_every_case() -> None:
    clean = _outcome(_case(), _response("Oakland Community Food Pantry [1]."))
    leaky = _outcome(_case(must_not_appear=("nope",)), _response("nope"))

    assert EvalReport([clean, leaky]).leak_rate == 0.5


def test_decoy_citations_are_reported() -> None:
    outcome = _outcome(_case(), _response("ok"))
    report = EvalReport([outcome], forbidden_ids=frozenset({FOOD.resource_id}))

    assert report.decoy_citations == [FOOD.resource_id]


def test_escalation_accuracy_compares_to_the_expectation() -> None:
    correct = _outcome(_case(expect_escalation=False), _response("ok"))
    wrong = _outcome(_case(expect_escalation=True), _response("ok"))

    assert EvalReport([correct, wrong]).escalation_accuracy == 0.5


def test_latency_percentiles() -> None:
    outcomes = []
    for value in (10, 20, 30, 40, 100):
        outcome = _outcome(_case(), _response("ok"))
        outcome.metrics = RequestMetrics(request_id="r", total_ms=value)
        outcomes.append(outcome)

    summary = EvalReport(outcomes).latency_summary("total_ms")

    assert summary["count"] == 5
    assert summary["p50"] == 30
    assert summary["p95"] == 100
    assert summary["max"] == 100
    assert summary["mean"] == pytest.approx(40.0)


def test_an_empty_report_does_not_divide_by_zero() -> None:
    report = EvalReport([])

    assert report.retrieval_accuracy == 0.0
    assert report.leak_rate == 0.0
    assert report.latency_summary()["count"] == 0
