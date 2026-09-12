"""The Resource domain model."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.domain.resource import (
    Resource,
    ServiceArea,
    ServiceCategory,
    ServiceScope,
    VerificationStatus,
)


def _resource(**overrides: object) -> Resource:
    base: dict[str, object] = {
        "resource_id": "test-001",
        "organization_name": "Example Support Center",
        "description": "A synthetic organization used only in tests.",
        "service_categories": [ServiceCategory.FOOD_ASSISTANCE],
        "source": "test-fixture",
        "last_verified": date(2026, 1, 15),
    }
    return Resource.model_validate(base | overrides)


def test_a_minimal_resource_validates() -> None:
    resource = _resource()

    assert resource.resource_id == "test-001"
    assert resource.is_verified
    assert resource.verification_status is VerificationStatus.VERIFIED
    assert resource.service_area.scope is ServiceScope.LOCAL


def test_at_least_one_category_is_required() -> None:
    with pytest.raises(ValidationError):
        _resource(service_categories=[])


def test_resource_id_rejects_whitespace() -> None:
    with pytest.raises(ValidationError, match="whitespace"):
        _resource(resource_id="has space")


def test_last_verified_cannot_be_in_the_future() -> None:
    with pytest.raises(ValidationError, match="future"):
        _resource(last_verified=date.today() + timedelta(days=1))


def test_blank_list_entries_are_dropped() -> None:
    resource = _resource(languages=["English", "  ", "Spanish"])
    assert resource.languages == ["English", "Spanish"]


def test_staleness_is_measured_against_a_window() -> None:
    resource = _resource(last_verified=date(2026, 1, 1))
    as_of = date(2026, 9, 11)

    assert resource.age_in_days(as_of=as_of) == 253
    assert not resource.is_stale(max_age_days=365, as_of=as_of)
    assert resource.is_stale(max_age_days=180, as_of=as_of)


def test_unverified_resources_report_themselves() -> None:
    resource = _resource(verification_status=VerificationStatus.NEEDS_REVIEW)
    assert not resource.is_verified


def test_category_labels_are_human_readable() -> None:
    assert ServiceCategory.FOOD_ASSISTANCE.label == "food assistance"
    assert ServiceCategory.APPOINTMENT_SUPPORT.label == "appointment support"


@pytest.mark.parametrize(
    ("area", "expected"),
    [
        (ServiceArea(scope=ServiceScope.VIRTUAL), "virtual / remote services"),
        (ServiceArea(scope=ServiceScope.NATIONAL), "nationwide"),
        (ServiceArea(scope=ServiceScope.STATEWIDE, state="CA"), "statewide (CA)"),
        (ServiceArea(cities=["Oakland"], state="CA"), "Oakland, CA"),
        (ServiceArea(), "service area not specified"),
    ],
)
def test_service_area_describes_itself(area: ServiceArea, expected: str) -> None:
    assert area.describe() == expected


def test_boundless_scopes_are_flagged() -> None:
    assert ServiceScope.STATEWIDE.is_boundless
    assert ServiceScope.VIRTUAL.is_boundless
    assert not ServiceScope.LOCAL.is_boundless
