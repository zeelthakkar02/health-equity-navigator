"""Request/response models for the Navigator endpoint.

Every field here is part of the public contract a client integrates against.

Two things are deliberately absent: any relevance or similarity score, and any
field the model could populate directly. ``resources`` is built exclusively from
retrieved :class:`~app.domain.resource.Resource` objects, never from generated
text, so a citation can never describe an organization that does not exist.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUERY_LENGTH = 2000


class ResourceCitation(BaseModel):
    """A verified community resource cited in an answer.

    Built only from a stored Resource. The model never fills this in.
    """

    resource_id: str
    title: str = Field(description="The organization's name.")
    organization: str | None = None
    category: str | None = Field(
        default=None, description="Primary service category, for simple clients."
    )
    categories: list[str] = Field(
        default_factory=list, description="All service categories this resource covers."
    )
    url: str | None = None
    phone: str | None = None
    service_area: str | None = Field(
        default=None, description="Where the organization serves people, in plain words."
    )
    eligibility: str | None = None
    languages: list[str] = Field(default_factory=list)
    accessibility: list[str] = Field(default_factory=list)
    cost: str | None = None
    last_verified: date | None = Field(
        default=None, description="When a human last confirmed these details, if known."
    )
    verification_status: str | None = Field(
        default=None, description="'verified' or 'partially_verified'."
    )
    confirmation_notes: str | None = Field(
        default=None,
        description=(
            "Details a human flagged as still needing confirmation. Show this to the member."
        ),
    )
    snippet: str | None = Field(
        default=None, description="Short description the answer was grounded on."
    )
    # No relevance/similarity score is exposed here by design. Ranking scores are
    # internal signals on RetrievedResource: they are not comparable across
    # embedding providers, they shift whenever ranking is tuned, and a number
    # next to a community resource reads as a quality judgement about the
    # organization, which it is not.


class NavigatorQueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "I need a clinic that takes uninsured patients near downtown.",
                "location": "43210",
                "language": "en",
                "categories": ["primary_care"],
                "max_resources": 5,
            }
        }
    )

    query: str = Field(
        min_length=3,
        max_length=MAX_QUERY_LENGTH,
        description="The community member's need, in their own words. Do not send PHI.",
    )
    location: str | None = Field(
        default=None, max_length=120, description="ZIP code, city, or neighbourhood."
    )
    language: str = Field(default="en", min_length=2, max_length=10)
    categories: list[str] | None = Field(
        default=None, max_length=10, description="Optional resource-category filters."
    )
    max_resources: int = Field(default=5, ge=1, le=20)
    session_id: str | None = Field(
        default=None, max_length=64, description="Opaque client-supplied correlation id."
    )

    @field_validator("query", "location", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class AnswerSource(StrEnum):
    """How the answer text was produced."""

    GENERATED = "generated"
    VERIFIED_LISTING = "verified_listing"
    NO_MATCH = "no_match"
    SAFETY_NOTICE = "safety_notice"


class EscalationReason(StrEnum):
    """Why a human should pick this up."""

    NO_VERIFIED_RESOURCES = "no_verified_resources"
    GROUNDING_FAILED = "grounding_failed"
    ANSWER_TRUNCATED = "answer_truncated"
    POSSIBLE_CRISIS = "possible_crisis"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    GENERATION_UNAVAILABLE = "generation_unavailable"


class NavigatorQueryResponse(BaseModel):
    request_id: str
    answer: str
    provider: str = Field(description="LLM provider that produced the answer.")
    model: str
    resources: list[ResourceCitation] = Field(
        default_factory=list,
        description="Verified resources cited. Built only from stored records.",
    )
    retrieval_performed: bool = Field(
        default=False, description="Whether the retrieval pipeline ran for this request."
    )
    needs_escalation: bool = Field(
        default=False,
        description=(
            "True when a person should follow up: no match, failed grounding, or a crisis signal."
        ),
    )
    escalation_reason: EscalationReason | None = None
    answer_source: AnswerSource = Field(
        default=AnswerSource.GENERATED,
        description="Whether the answer was generated, or assembled without a model.",
    )
    disclaimer: str
    latency_ms: int
