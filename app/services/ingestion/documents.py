"""Turning a Resource into text to embed and metadata to filter on.

Kept out of the domain model so the retrieval representation can evolve without
touching the shared vocabulary. Everything a community member might describe
their need with — category names, concrete services, languages, accessibility
accommodations, the places served — goes into the embedding text.
"""

from __future__ import annotations

from typing import Any

from app.domain.resource import Resource


def build_embedding_text(resource: Resource) -> str:
    """Compose the document text that represents a resource in the index."""
    category_labels = ", ".join(category.label for category in resource.service_categories)
    lines = [
        f"Organization: {resource.organization_name}",
        f"Services offered: {category_labels}",
    ]
    if resource.services:
        lines.append(f"Specific help: {', '.join(resource.services)}")
    if resource.source_category:
        lines.append(f"Listed under: {resource.source_category}")
    if resource.search_tags:
        # The source's own keywords are what someone would search for.
        lines.append(f"Also known for: {', '.join(resource.search_tags)}")
    lines.append(f"Description: {resource.description}")
    lines.append(f"Service area: {resource.service_area.describe()}")
    if resource.eligibility:
        lines.append(f"Who qualifies: {resource.eligibility}")
    if resource.languages:
        lines.append(f"Languages spoken: {', '.join(resource.languages)}")
    if resource.accessibility:
        lines.append(f"Accessibility: {', '.join(resource.accessibility)}")
    if resource.ada_access:
        lines.append(f"ADA access: {resource.ada_access}")
    if resource.transit_access:
        lines.append(f"Public transit: {resource.transit_access}")
    if resource.application_required:
        lines.append(f"Application required: {resource.application_required}")
    if resource.cost:
        lines.append(f"Cost: {resource.cost}")
    lines.append(f"Verification status: {resource.verification_status.label}")
    if resource.confirmation_notes:
        # Carried into the prompt so the model can pass the caveat on rather
        # than presenting an unconfirmed detail as settled fact.
        lines.append(f"Still needs confirming: {resource.confirmation_notes}")
    return "\n".join(lines)


def build_metadata(resource: Resource) -> dict[str, Any]:
    """Flat, filterable fields. Values stay strings so any backend can index them."""
    area = resource.service_area
    return {
        "resource_id": resource.resource_id,
        "categories": [category.value for category in resource.service_categories],
        "scope": area.scope.value,
        "cities": [city.lower() for city in area.cities],
        "counties": [county.lower() for county in area.counties],
        "state": (area.state or "").lower(),
        "postal_codes": list(area.postal_codes),
        "languages": [language.lower() for language in resource.languages],
        "verification_status": resource.verification_status.value,
        "last_verified": resource.last_verified.isoformat() if resource.last_verified else "",
    }


def build_payload(resource: Resource) -> dict[str, Any]:
    """The record handed back verbatim by the store, JSON-safe."""
    return resource.model_dump(mode="json")
