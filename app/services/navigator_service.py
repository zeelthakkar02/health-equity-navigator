"""Navigator orchestration: retrieval-grounded generation.

    message
      -> crisis screen           (deterministic; no model call)
      -> detect stated needs     (keyword lexicon)
      -> retrieve                (verified resources only, above the tuned gate)
      -> generate                (Gemini sees ONLY the retrieved resources)
      -> validate grounding      (discard the answer if it left its context)
      -> respond                 (citations built only from stored records)

Design commitments this module keeps:

* **It never sees a provider.** It depends on ``LLMService`` and
  ``ResourceRetriever``. No Vertex, google-genai, or embedding symbol appears
  here, so swapping either side changes nothing in this file.
* **Citations come from records, not from text.** ``resources`` is built from
  the retrieved :class:`Resource` objects. Whatever the model writes, it cannot
  put an organization into that array.
* **Degrade, do not invent.** Every failure path — no matches, a failed
  grounding check, an unavailable model — produces a useful answer assembled
  from verified records, with ``needs_escalation`` set, rather than a guess.
"""

from __future__ import annotations

import logging
import time
import uuid

from app.core.config import Settings
from app.core.metrics import MetricsSink, RequestMetrics, log_metrics
from app.domain.resource import Resource, ServiceCategory
from app.schemas.navigator import (
    AnswerSource,
    EscalationReason,
    NavigatorQueryRequest,
    NavigatorQueryResponse,
    ResourceCitation,
)
from app.services import crisis
from app.services.grounding import GroundingValidator
from app.services.llm.base import LLMRequest, LLMService
from app.services.llm.errors import LLMError
from app.services.llm.prompts import (
    DISCLAIMER,
    NO_RESOURCES_ANSWER,
    SYSTEM_INSTRUCTION,
    build_navigator_prompt,
)
from app.services.retrieval.need_lexicon import detect_categories
from app.services.retrieval.retriever import ResourceRetriever

logger = logging.getLogger(__name__)

SNIPPET_LENGTH = 320


class NavigatorService:
    """Turns a community member's need into a grounded, cited answer."""

    def __init__(
        self,
        llm: LLMService,
        settings: Settings,
        retriever: ResourceRetriever | None = None,
        metrics_sink: MetricsSink | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._retriever = retriever
        self._metrics_sink = metrics_sink or log_metrics

    async def answer(
        self,
        request: NavigatorQueryRequest,
        *,
        request_id: str | None = None,
    ) -> NavigatorQueryResponse:
        request_id = request_id or str(uuid.uuid4())
        started = time.perf_counter()
        metrics = RequestMetrics(
            request_id=request_id,
            query_chars=len(request.query),
            has_location=request.location is not None,
            language=request.language,
            provider=self._llm.provider_name,
            model=self._llm.model_name,
            query_text=request.query if self._settings.log_query_text else None,
        )

        signal = crisis.screen(request.query)
        if signal.triggered:
            logger.warning(
                "crisis signal (%s) — returning safety notice without a model call",
                signal.kind,
                extra={"request_id": request_id},
            )
            return self._respond(
                request_id=request_id,
                started=started,
                answer=crisis.SAFETY_NOTICE,
                citations=[],
                retrieval_performed=False,
                source=AnswerSource.SAFETY_NOTICE,
                escalation=EscalationReason.POSSIBLE_CRISIS,
                metrics=metrics,
            )

        stated_needs = detect_categories(request.query)
        metrics.stated_needs = sorted(category.value for category in stated_needs)

        retrieval_started = time.perf_counter()
        resources, retrieval_failed = await self._retrieve(request)
        metrics.retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
        metrics.resources_returned = len(resources)
        citations = [to_citation(resource) for resource in resources]

        if not resources:
            reason = (
                EscalationReason.RETRIEVAL_UNAVAILABLE
                if retrieval_failed
                else EscalationReason.NO_VERIFIED_RESOURCES
            )
            logger.info(
                "no verified resources for query (%s)",
                reason.value,
                extra={"request_id": request_id},
            )
            return self._respond(
                request_id=request_id,
                started=started,
                answer=NO_RESOURCES_ANSWER,
                citations=[],
                retrieval_performed=not retrieval_failed,
                source=AnswerSource.NO_MATCH,
                escalation=reason,
                metrics=metrics,
            )

        llm_started = time.perf_counter()
        try:
            generated = await self._generate(
                request, citations, stated_needs=stated_needs, request_id=request_id
            )
        except LLMError as exc:
            metrics.llm_ms = int((time.perf_counter() - llm_started) * 1000)
            # We already hold verified resources; showing them beats a 502. With
            # nothing to show, the error is the honest answer.
            logger.error(
                "generation failed (%s), falling back to a verified listing: %s",
                exc.code,
                exc,
                extra={"request_id": request_id},
            )
            return self._respond(
                request_id=request_id,
                started=started,
                answer=build_listing_answer(resources),
                citations=citations,
                retrieval_performed=True,
                source=AnswerSource.VERIFIED_LISTING,
                escalation=EscalationReason.GENERATION_UNAVAILABLE,
                metrics=metrics,
            )

        metrics.llm_ms = int((time.perf_counter() - llm_started) * 1000)

        if _is_truncated(generated.finish_reason):
            # An answer cut off at "call them at" is worse than no answer.
            logger.error(
                "generated answer hit the output cap, discarding it (finish_reason=%s)",
                generated.finish_reason,
                extra={"request_id": request_id},
            )
            return self._respond(
                request_id=request_id,
                started=started,
                answer=build_listing_answer(resources),
                citations=citations,
                retrieval_performed=True,
                source=AnswerSource.VERIFIED_LISTING,
                escalation=EscalationReason.ANSWER_TRUNCATED,
                provider=generated.provider,
                model=generated.model,
                metrics=metrics,
            )

        grounding_started = time.perf_counter()
        grounding = GroundingValidator.for_resources(resources).validate(generated.text)
        metrics.grounding_ms = int((time.perf_counter() - grounding_started) * 1000)
        metrics.grounding_issues = [issue.kind for issue in grounding.issues]

        if not grounding.passed:
            logger.error(
                "grounding validation failed, discarding generated answer: %s",
                grounding.summary(),
                extra={"request_id": request_id},
            )
            return self._respond(
                request_id=request_id,
                started=started,
                answer=build_listing_answer(resources),
                citations=citations,
                retrieval_performed=True,
                source=AnswerSource.VERIFIED_LISTING,
                escalation=EscalationReason.GROUNDING_FAILED,
                provider=generated.provider,
                model=generated.model,
                metrics=metrics,
            )

        logger.info(
            "navigator query answered provider=%s model=%s resources=%d grounded=True",
            generated.provider,
            generated.model,
            len(resources),
            extra={"request_id": request_id},
        )
        return self._respond(
            request_id=request_id,
            started=started,
            answer=generated.text,
            citations=citations,
            retrieval_performed=True,
            source=AnswerSource.GENERATED,
            provider=generated.provider,
            model=generated.model,
            metrics=metrics,
        )

    async def _retrieve(self, request: NavigatorQueryRequest) -> tuple[list[Resource], bool]:
        """Retrieve verified resources. Returns (resources, retrieval_failed).

        A broken index must not produce an ungrounded answer, so a failure here
        is reported as "no resources" and escalated rather than raised.
        """
        if self._retriever is None:
            return [], False

        categories = (
            [ServiceCategory(value) for value in request.categories] if request.categories else None
        )
        try:
            results = await self._retriever.retrieve(
                request.query,
                location=request.location,
                categories=categories,
                top_k=request.max_resources,
            )
        except ValueError:
            # An unknown category from the client is a bad request, not an outage.
            raise
        except Exception:
            logger.exception("retrieval failed")
            return [], True

        return [result.resource for result in results], False

    async def _generate(
        self,
        request: NavigatorQueryRequest,
        citations: list[ResourceCitation],
        *,
        stated_needs: set[ServiceCategory],
        request_id: str,
    ):
        prompt = build_navigator_prompt(
            query=request.query,
            resources=citations,
            location=request.location,
            language=request.language,
            stated_needs=stated_needs,
        )
        return await self._llm.generate(
            LLMRequest(
                prompt=prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self._settings.llm_temperature,
                max_output_tokens=self._settings.llm_max_output_tokens,
                metadata={"request_id": request_id},
            )
        )

    def _respond(
        self,
        *,
        request_id: str,
        started: float,
        answer: str,
        citations: list[ResourceCitation],
        retrieval_performed: bool,
        source: AnswerSource,
        escalation: EscalationReason | None = None,
        provider: str | None = None,
        model: str | None = None,
        metrics: RequestMetrics | None = None,
    ) -> NavigatorQueryResponse:
        total_ms = int((time.perf_counter() - started) * 1000)

        if metrics is not None:
            metrics.answer_source = source.value
            metrics.escalation_reason = escalation.value if escalation else None
            metrics.needs_escalation = escalation is not None
            metrics.total_ms = total_ms
            metrics.provider = provider or self._llm.provider_name
            metrics.model = model or self._llm.model_name
            self._metrics_sink(metrics)

        return NavigatorQueryResponse(
            request_id=request_id,
            answer=answer,
            provider=provider or self._llm.provider_name,
            model=model or self._llm.model_name,
            resources=citations,
            retrieval_performed=retrieval_performed,
            needs_escalation=escalation is not None,
            escalation_reason=escalation,
            answer_source=source,
            disclaimer=DISCLAIMER,
            latency_ms=total_ms,
        )


def _is_truncated(finish_reason: str | None) -> bool:
    """True when the model stopped because it ran out of output budget."""
    return "MAX_TOKENS" in (finish_reason or "").upper()


def to_citation(resource: Resource) -> ResourceCitation:
    """Project a stored resource into its public citation form."""
    description = resource.description
    snippet = (
        description
        if len(description) <= SNIPPET_LENGTH
        else description[:SNIPPET_LENGTH].rsplit(" ", 1)[0] + "…"
    )
    return ResourceCitation(
        resource_id=resource.resource_id,
        title=resource.organization_name,
        organization=resource.organization_name,
        category=resource.service_categories[0].value if resource.service_categories else None,
        categories=[category.value for category in resource.service_categories],
        url=resource.website,
        phone=resource.phone,
        service_area=resource.service_area.describe(),
        eligibility=resource.eligibility,
        languages=list(resource.languages),
        accessibility=list(resource.accessibility),
        cost=resource.cost,
        last_verified=resource.last_verified,
        snippet=snippet,
    )


def build_listing_answer(resources: list[Resource]) -> str:
    """Assemble an answer from records alone, with no model involved.

    Used when generation is unavailable or its output failed grounding. Plainer
    than a written answer, but every word of it is verified.
    """
    lines = [
        "Here are the verified resources that match what you described. I am listing "
        "them directly from our records rather than summarising them.",
        "",
    ]
    for index, resource in enumerate(resources, start=1):
        lines.append(f"{index}. {resource.organization_name}")
        lines.append(f"   {resource.description}")
        contact = [part for part in (resource.phone, resource.website) if part]
        if contact:
            lines.append(f"   Contact: {' | '.join(contact)}")
        lines.append(f"   Serves: {resource.service_area.describe()}")
        if resource.eligibility:
            lines.append(f"   Who qualifies: {resource.eligibility}")
        lines.append("")

    lines.append(
        "Please confirm details directly with the organization before you go. For more "
        "local referrals you can also call 211."
    )
    return "\n".join(lines)
