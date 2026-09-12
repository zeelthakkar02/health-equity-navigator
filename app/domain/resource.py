"""The Resource domain model.

A Resource is one verified community or support organization the Navigator can
point someone to. This module knows nothing about embeddings, vector stores, or
HTTP — it is the shared vocabulary those layers agree on.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ServiceCategory(StrEnum):
    """The needs the Navigator helps with."""

    TRANSPORTATION = "transportation"
    FOOD_ASSISTANCE = "food_assistance"
    HOUSING_SUPPORT = "housing_support"
    CAREGIVER_SUPPORT = "caregiver_support"
    MENTAL_HEALTH = "mental_health"
    COMMUNITY_SERVICES = "community_services"
    INSURANCE_NAVIGATION = "insurance_navigation"
    LANGUAGE_ASSISTANCE = "language_assistance"
    ACCESSIBILITY_SUPPORT = "accessibility_support"
    COMMUNITY_CLINIC = "community_clinic"
    APPOINTMENT_SUPPORT = "appointment_support"

    @property
    def label(self) -> str:
        """Human-readable form, used in embedding text and CLI output."""
        return self.value.replace("_", " ")


class ServiceScope(StrEnum):
    """How wide a net the organization casts."""

    LOCAL = "local"
    COUNTY = "county"
    STATEWIDE = "statewide"
    NATIONAL = "national"
    VIRTUAL = "virtual"

    @property
    def is_boundless(self) -> bool:
        """True when the organization serves people regardless of city."""
        return self in {ServiceScope.STATEWIDE, ServiceScope.NATIONAL, ServiceScope.VIRTUAL}


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    UNVERIFIED = "unverified"


class ServiceArea(BaseModel):
    """Where an organization actually serves people.

    Kept separate from a street address: an organization in one city routinely
    serves several, and some serve everyone in a state or online.
    """

    model_config = ConfigDict(frozen=True)

    scope: ServiceScope = ServiceScope.LOCAL
    cities: list[str] = Field(default_factory=list)
    counties: list[str] = Field(default_factory=list)
    state: str | None = None
    postal_codes: list[str] = Field(default_factory=list)

    @field_validator("cities", "counties", "postal_codes", mode="before")
    @classmethod
    def _clean_list(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    def describe(self) -> str:
        """One-line summary for prompts, embedding text, and CLI output."""
        if self.scope is ServiceScope.VIRTUAL:
            return "virtual / remote services"
        if self.scope is ServiceScope.NATIONAL:
            return "nationwide"
        if self.scope is ServiceScope.STATEWIDE:
            return f"statewide ({self.state})" if self.state else "statewide"
        places = self.cities or self.counties
        if places:
            suffix = f", {self.state}" if self.state else ""
            return ", ".join(places) + suffix
        return self.state or "service area not specified"


class Resource(BaseModel):
    """A verified community or support organization."""

    model_config = ConfigDict(frozen=True)

    resource_id: str = Field(min_length=1, max_length=64)
    organization_name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=10, max_length=2000)

    service_categories: list[ServiceCategory] = Field(min_length=1)
    services: list[str] = Field(
        default_factory=list, description="Concrete offerings, e.g. 'rides to dialysis'."
    )

    service_area: ServiceArea = Field(default_factory=ServiceArea)
    address: str | None = Field(default=None, max_length=300)

    eligibility: str | None = Field(
        default=None, max_length=1000, description="Who qualifies, in plain language."
    )
    languages: list[str] = Field(
        default_factory=list, description="Languages served, e.g. ['English', 'Spanish']."
    )
    accessibility: list[str] = Field(
        default_factory=list,
        description="Accommodations offered, e.g. ['wheelchair accessible', 'ASL interpretation'].",
    )
    cost: str | None = Field(
        default=None, max_length=200, description="e.g. 'free', 'sliding scale'."
    )

    phone: str | None = Field(default=None, max_length=40)
    website: str | None = Field(default=None, max_length=500)

    source: str = Field(min_length=1, max_length=200, description="Where this record came from.")
    last_verified: date = Field(description="When a human last confirmed these details.")
    verification_status: VerificationStatus = VerificationStatus.VERIFIED

    @field_validator("services", "languages", "accessibility", mode="before")
    @classmethod
    def _clean_list(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    @field_validator("resource_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(character.isspace() for character in cleaned):
            raise ValueError("resource_id must be non-empty and contain no whitespace")
        return cleaned

    @field_validator("last_verified")
    @classmethod
    def _reject_future_dates(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("last_verified cannot be in the future")
        return value

    @property
    def is_verified(self) -> bool:
        return self.verification_status is VerificationStatus.VERIFIED

    def age_in_days(self, *, as_of: date | None = None) -> int:
        return ((as_of or date.today()) - self.last_verified).days

    def is_stale(self, *, max_age_days: int, as_of: date | None = None) -> bool:
        """True when the record is older than the freshness window."""
        return self.age_in_days(as_of=as_of) > max_age_days
