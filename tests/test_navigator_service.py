"""Grounded RAG orchestration.

Every scenario runs against a fake LLM and a fake retriever, so these assert on
the Navigator's own decisions — what it sends to the model, what it does with
what comes back, and what it refuses to return — rather than on any provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.core.config import Settings
from app.domain.resource import Resource, ServiceCategory
from app.schemas.navigator import AnswerSource, EscalationReason, NavigatorQueryRequest
from app.services.llm.base import LLMRequest, LLMResponse, LLMService
from app.services.llm.errors import LLMTimeoutError, LLMUpstreamError
from app.services.navigator_service import NavigatorService
from app.services.retrieval.retriever import RetrievedResource

# --- test doubles ----------------------------------------------------------


@dataclass
class FakeLLM(LLMService):
    """Returns canned text and records exactly what it was asked."""

    text: str = "Grounded answer."
    error: Exception | None = None
    provider_name: str = "fake"
    model_name: str = "fake-model"
    requests: list[LLMRequest] = field(default_factory=list)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(
            text=self.text,
            model=self.model_name,
            provider=self.provider_name,
            finish_reason="stop",
        )

    @property
    def last_prompt(self) -> str:
        return self.requests[-1].prompt


@dataclass
class FakeRetriever:
    """Stands in for ResourceRetriever without embeddings or a store."""

    resources: list[Resource] = field(default_factory=list)
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    async def retrieve(self, query: str, **kwargs: object) -> list[RetrievedResource]:
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        return [
            RetrievedResource(
                resource=resource,
                score=0.9 - index * 0.01,
                semantic_score=0.9 - index * 0.01,
                location_match=0.0,
                rank=index + 1,
            )
            for index, resource in enumerate(self.resources)
        ]


def make_resource(
    resource_id: str,
    name: str,
    *categories: ServiceCategory,
    phone: str = "555-0100",
    website: str = "https://example.org/org",
    cities: list[str] | None = None,
) -> Resource:
    return Resource.model_validate(
        {
            "resource_id": resource_id,
            "organization_name": name,
            "description": f"{name} provides synthetic support used only in tests.",
            "service_categories": [category.value for category in categories]
            or ["community_services"],
            "service_area": {"scope": "local", "cities": cities or ["Oakland"], "state": "CA"},
            "phone": phone,
            "website": website,
            "eligibility": "Open to everyone.",
            "languages": ["English", "Spanish"],
            "accessibility": ["wheelchair accessible"],
            "source": "test-fixture",
            "last_verified": "2026-08-01",
        }
    )


FOOD = make_resource(
    "syn-food-001", "Oakland Community Food Pantry", ServiceCategory.FOOD_ASSISTANCE
)
NUTRITION = make_resource(
    "syn-food-002",
    "Bay Area Nutrition Access Coalition",
    ServiceCategory.FOOD_ASSISTANCE,
    phone="555-0177",
    website="https://example.org/nutrition",
)
ACCESS = make_resource(
    "syn-access-001",
    "Independent Living Access Network",
    ServiceCategory.ACCESSIBILITY_SUPPORT,
    phone="555-0138",
    website="https://example.org/access",
    cities=["Berkeley"],
)


def build(
    settings: Settings,
    *,
    llm: FakeLLM | None = None,
    resources: list[Resource] | None = None,
    retriever_error: Exception | None = None,
    retriever: FakeRetriever | None = None,
) -> tuple[NavigatorService, FakeLLM, FakeRetriever]:
    fake_llm = llm or FakeLLM()
    fake_retriever = retriever or FakeRetriever(
        resources=resources if resources is not None else [FOOD], error=retriever_error
    )
    service = NavigatorService(fake_llm, settings, retriever=fake_retriever)  # type: ignore[arg-type]
    return service, fake_llm, fake_retriever


def ask(**kwargs: object) -> NavigatorQueryRequest:
    payload: dict[str, object] = {"query": "I need help finding food."}
    return NavigatorQueryRequest.model_validate(payload | kwargs)


# --- 1. successful grounded recommendation ---------------------------------


async def test_a_grounded_answer_is_returned_with_citations(settings: Settings) -> None:
    llm = FakeLLM(text="Oakland Community Food Pantry [1] can help. Call 555-0100.")
    service, _, _ = build(settings, llm=llm)

    response = await service.answer(ask())

    assert response.answer_source is AnswerSource.GENERATED
    assert response.answer == llm.text
    assert response.needs_escalation is False
    assert response.escalation_reason is None
    assert response.retrieval_performed is True
    assert [citation.resource_id for citation in response.resources] == ["syn-food-001"]
    assert response.resources[0].title == "Oakland Community Food Pantry"
    assert response.resources[0].phone == "555-0100"


async def test_citations_come_from_records_not_from_the_model(settings: Settings) -> None:
    """Whatever the model writes, it cannot put an organization in `resources`."""
    llm = FakeLLM(text="Oakland Community Food Pantry [1] can help.")
    service, _, _ = build(settings, llm=llm)

    response = await service.answer(ask())

    citation = response.resources[0]
    assert citation.last_verified == FOOD.last_verified
    assert citation.eligibility == FOOD.eligibility
    assert citation.languages == FOOD.languages
    assert citation.service_area == "Oakland, CA"


async def test_the_prompt_carries_the_context_the_model_needs(settings: Settings) -> None:
    service, llm, _ = build(settings)

    await service.answer(ask(query="I need food assistance near Oakland.", location="Oakland"))

    prompt = llm.last_prompt
    assert "Oakland Community Food Pantry" in prompt
    assert "[1]" in prompt
    assert "location: Oakland" in prompt
    assert "food assistance" in prompt  # detected stated need
    assert "555-0100" in prompt
    assert "grounding" in llm.requests[-1].system_instruction.lower()


async def test_no_internal_score_reaches_the_prompt(settings: Settings) -> None:
    service, llm, _ = build(settings)

    await service.answer(ask())

    assert "0.9" not in llm.last_prompt
    assert "score" not in llm.last_prompt.lower()


# --- 2. multiple relevant resources ----------------------------------------


async def test_multiple_resources_are_all_offered_and_cited(settings: Settings) -> None:
    llm = FakeLLM(
        text=(
            "Oakland Community Food Pantry [1] gives out groceries. "
            "Bay Area Nutrition Access Coalition [2] can help you apply for benefits."
        )
    )
    service, _, _ = build(settings, llm=llm, resources=[FOOD, NUTRITION])

    response = await service.answer(ask())

    assert response.answer_source is AnswerSource.GENERATED
    assert [citation.resource_id for citation in response.resources] == [
        "syn-food-001",
        "syn-food-002",
    ]
    assert "[2]" in response.answer


# --- 3. no matching resources ----------------------------------------------


async def test_no_resources_skips_the_model_entirely(settings: Settings) -> None:
    service, llm, _ = build(settings, resources=[])

    response = await service.answer(ask(query="I need a helicopter to the moon."))

    assert llm.requests == [], "the model must never be asked to invent an answer"
    assert response.answer_source is AnswerSource.NO_MATCH
    assert response.resources == []
    assert response.needs_escalation is True
    assert response.escalation_reason is EscalationReason.NO_VERIFIED_RESOURCES
    assert "211" in response.answer
    assert response.retrieval_performed is True


# --- 4. off-topic query ----------------------------------------------------


async def test_an_off_topic_query_gets_the_no_match_response(settings: Settings) -> None:
    """The retriever's gate rejects it, so nothing reaches the model."""
    service, llm, _ = build(settings, resources=[])

    response = await service.answer(ask(query="Who won the football game last night?"))

    assert llm.requests == []
    assert response.answer_source is AnswerSource.NO_MATCH
    assert response.needs_escalation is True


# --- 5. location-aware retrieval -------------------------------------------


async def test_the_location_is_passed_through_to_retrieval(settings: Settings) -> None:
    service, _, retriever = build(settings)

    await service.answer(ask(query="I need groceries.", location="94601"))

    assert retriever.calls[0]["location"] == "94601"


async def test_request_filters_reach_the_retriever(settings: Settings) -> None:
    service, _, retriever = build(settings)

    await service.answer(ask(categories=["food_assistance"], max_resources=3))

    call = retriever.calls[0]
    assert call["categories"] == [ServiceCategory.FOOD_ASSISTANCE]
    assert call["top_k"] == 3


# --- 6. accessibility request ----------------------------------------------


async def test_an_accessibility_request_is_answered_from_accessibility_records(
    settings: Settings,
) -> None:
    llm = FakeLLM(text="Independent Living Access Network [1] lends mobility equipment.")
    service, fake_llm, _ = build(settings, llm=llm, resources=[ACCESS])

    response = await service.answer(ask(query="My mother needs wheelchair-accessible support."))

    assert response.answer_source is AnswerSource.GENERATED
    assert response.resources[0].resource_id == "syn-access-001"
    assert "wheelchair accessible" in response.resources[0].accessibility
    assert "accessibility support" in fake_llm.last_prompt


# --- 7. the model hallucinates an organization -----------------------------


async def test_a_hallucinated_organization_is_never_returned(settings: Settings) -> None:
    llm = FakeLLM(text="Try the Bayview Family Resource Center [1], they hand out food.")
    service, _, _ = build(settings, llm=llm)

    response = await service.answer(ask())

    assert "Bayview Family Resource Center" not in response.answer
    assert response.answer_source is AnswerSource.VERIFIED_LISTING
    assert response.escalation_reason is EscalationReason.GROUNDING_FAILED
    assert response.needs_escalation is True
    # The verified records are still offered, straight from storage.
    assert response.resources[0].resource_id == "syn-food-001"
    assert "Oakland Community Food Pantry" in response.answer


async def test_a_hallucinated_phone_number_is_never_returned(settings: Settings) -> None:
    llm = FakeLLM(text="Call Oakland Community Food Pantry [1] at 555-9999 today.")
    service, _, _ = build(settings, llm=llm)

    response = await service.answer(ask())

    assert "555-9999" not in response.answer
    assert response.escalation_reason is EscalationReason.GROUNDING_FAILED


async def test_a_hallucinated_link_is_never_returned(settings: Settings) -> None:
    llm = FakeLLM(text="See https://bayareafoodhelp.example.com for details.")
    service, _, _ = build(settings, llm=llm)

    response = await service.answer(ask())

    assert "bayareafoodhelp" not in response.answer
    assert response.escalation_reason is EscalationReason.GROUNDING_FAILED


async def test_a_citation_out_of_range_is_never_returned(settings: Settings) -> None:
    llm = FakeLLM(text="Oakland Community Food Pantry [4] can help you.")
    service, _, _ = build(settings, llm=llm)

    response = await service.answer(ask())

    assert response.escalation_reason is EscalationReason.GROUNDING_FAILED


# --- truncated generation --------------------------------------------------


async def test_an_answer_cut_off_by_the_output_cap_is_discarded(settings: Settings) -> None:
    """Half an answer ending in "call them at" is worse than no answer."""

    class TruncatingLLM(FakeLLM):
        async def generate(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            return LLMResponse(
                text="Oakland Community Food Pantry [1] is open. You can call them at",
                model=self.model_name,
                provider=self.provider_name,
                finish_reason="FinishReason.MAX_TOKENS",
            )

    service, _, _ = build(settings, llm=TruncatingLLM())

    response = await service.answer(ask())

    assert "call them at" not in response.answer
    assert response.answer_source is AnswerSource.VERIFIED_LISTING
    assert response.escalation_reason is EscalationReason.ANSWER_TRUNCATED
    assert response.needs_escalation is True


async def test_a_normal_stop_is_not_treated_as_truncation(settings: Settings) -> None:
    llm = FakeLLM(text="Oakland Community Food Pantry [1] can help.")
    service, _, _ = build(settings, llm=llm)

    response = await service.answer(ask())

    assert response.answer_source is AnswerSource.GENERATED


# --- 8. LLM failure --------------------------------------------------------


async def test_generation_failure_degrades_to_the_verified_listing(settings: Settings) -> None:
    service, _, _ = build(settings, llm=FakeLLM(error=LLMUpstreamError("Vertex is down")))

    response = await service.answer(ask())

    assert response.answer_source is AnswerSource.VERIFIED_LISTING
    assert response.escalation_reason is EscalationReason.GENERATION_UNAVAILABLE
    assert response.needs_escalation is True
    assert "Oakland Community Food Pantry" in response.answer
    assert response.resources[0].resource_id == "syn-food-001"


async def test_a_generation_timeout_also_degrades(settings: Settings) -> None:
    service, _, _ = build(settings, llm=FakeLLM(error=LLMTimeoutError("too slow")))

    response = await service.answer(ask())

    assert response.answer_source is AnswerSource.VERIFIED_LISTING


async def test_generation_failure_with_nothing_to_show_is_raised(settings: Settings) -> None:
    """With no resources the model is never called, so there is nothing to degrade to."""
    service, llm, _ = build(settings, llm=FakeLLM(error=LLMUpstreamError("down")), resources=[])

    response = await service.answer(ask())

    assert llm.requests == []
    assert response.answer_source is AnswerSource.NO_MATCH


# --- 9. retrieval failure --------------------------------------------------


async def test_retrieval_failure_never_produces_an_ungrounded_answer(
    settings: Settings,
) -> None:
    service, llm, _ = build(settings, retriever_error=RuntimeError("index unavailable"))

    response = await service.answer(ask())

    assert llm.requests == [], "a broken index must not lead to a guessed answer"
    assert response.answer_source is AnswerSource.NO_MATCH
    assert response.escalation_reason is EscalationReason.RETRIEVAL_UNAVAILABLE
    assert response.retrieval_performed is False
    assert response.resources == []


async def test_a_bad_category_from_the_client_is_a_bad_request(settings: Settings) -> None:
    service, _, _ = build(settings)

    with pytest.raises(ValueError):
        await service.answer(ask(categories=["not_a_category"]))


# --- crisis screening ------------------------------------------------------


async def test_a_crisis_message_short_circuits_before_any_model_call(
    settings: Settings,
) -> None:
    service, llm, retriever = build(settings)

    response = await service.answer(ask(query="I want to kill myself."))

    assert llm.requests == []
    assert retriever.calls == []
    assert response.answer_source is AnswerSource.SAFETY_NOTICE
    assert response.escalation_reason is EscalationReason.POSSIBLE_CRISIS
    assert response.needs_escalation is True
    assert "988" in response.answer
    assert "911" in response.answer


# --- no retriever configured -----------------------------------------------


async def test_without_a_retriever_the_navigator_still_answers_safely(
    settings: Settings,
) -> None:
    service = NavigatorService(FakeLLM(), settings, retriever=None)

    response = await service.answer(ask())

    assert response.answer_source is AnswerSource.NO_MATCH
    assert response.resources == []
    assert response.retrieval_performed is True


# --- response shape --------------------------------------------------------


async def test_the_response_never_carries_a_score(settings: Settings) -> None:
    service, _, _ = build(settings)

    response = await service.answer(ask())

    body = response.model_dump()
    assert not any("score" in key.lower() for key in body)
    assert all(
        not any("score" in key.lower() for key in citation) for citation in body["resources"]
    )


async def test_the_disclaimer_is_always_present(settings: Settings) -> None:
    service, _, _ = build(settings, resources=[])

    response = await service.answer(ask())

    assert response.disclaimer.startswith("This is general information")
