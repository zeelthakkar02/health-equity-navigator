"""Post-generation grounding validation."""

from __future__ import annotations

import pytest

from app.domain.resource import Resource
from app.services.grounding import GroundingValidator


def _resource(**overrides: object) -> Resource:
    base: dict[str, object] = {
        "resource_id": "syn-food-001",
        "organization_name": "Oakland Community Food Pantry",
        "description": "Neighborhood food pantry providing free groceries.",
        "service_categories": ["food_assistance"],
        "phone": "555-0123",
        "website": "https://example.org/oakland-community-food-pantry",
        "source": "test-fixture",
        "last_verified": "2026-08-01",
    }
    return Resource.model_validate(base | overrides)


@pytest.fixture
def validator() -> GroundingValidator:
    return GroundingValidator.for_resources([_resource()])


# --- passing -------------------------------------------------------------


def test_an_answer_inside_its_context_passes(validator: GroundingValidator) -> None:
    result = validator.validate(
        "Oakland Community Food Pantry [1] gives out free groceries. "
        "Call 555-0123 or visit https://example.org/oakland-community-food-pantry."
    )

    assert result.passed
    assert result.summary() == "grounded"


def test_public_emergency_numbers_are_allowed(validator: GroundingValidator) -> None:
    assert validator.validate(
        "If this is an emergency call 911. For other referrals, dial 211 or text 988."
    ).passed


def test_generic_service_phrases_are_not_organizations(validator: GroundingValidator) -> None:
    assert validator.validate(
        "Ask about their social services and support services when you call."
    ).passed


def test_a_partial_organization_name_still_matches(validator: GroundingValidator) -> None:
    """A model shortening a name is not hallucinating it."""
    assert validator.validate("The Community Food Pantry [1] can help.").passed


def test_an_answer_with_no_specifics_passes(validator: GroundingValidator) -> None:
    assert validator.validate("I could not find a verified match for that.").passed


# --- failing -------------------------------------------------------------


def test_an_invented_organization_is_caught(validator: GroundingValidator) -> None:
    result = validator.validate("The Bayview Family Resource Center can help with food.")

    assert not result.passed
    assert result.issues[0].kind == "unknown_organization"


def test_an_invented_phone_number_is_caught(validator: GroundingValidator) -> None:
    result = validator.validate("Call Oakland Community Food Pantry [1] at 555-9999.")

    assert not result.passed
    assert [issue.kind for issue in result.issues] == ["unknown_phone"]


def test_an_invented_link_is_caught(validator: GroundingValidator) -> None:
    result = validator.validate("More details at https://bayareafoodhelp.example.com/apply.")

    assert not result.passed
    assert [issue.kind for issue in result.issues] == ["unknown_url"]


def test_a_citation_beyond_the_context_is_caught(validator: GroundingValidator) -> None:
    result = validator.validate("Oakland Community Food Pantry [3] is open today.")

    assert not result.passed
    assert result.issues[0].kind == "invalid_citation"


def test_citation_zero_is_caught(validator: GroundingValidator) -> None:
    assert not validator.validate("See [0] for details.").passed


def test_every_problem_is_reported_not_just_the_first(
    validator: GroundingValidator,
) -> None:
    result = validator.validate(
        "The Bayview Family Resource Center [9] is at 555-9999, see https://fake.example.net."
    )

    kinds = {issue.kind for issue in result.issues}
    assert kinds == {
        "unknown_organization",
        "unknown_phone",
        "unknown_url",
        "invalid_citation",
    }
    assert "unknown_phone" in result.summary()


# --- formatting tolerance ------------------------------------------------


@pytest.mark.parametrize(
    "written",
    ["555-0123", "(555) 0123", "555.0123", "555 0123"],
)
def test_a_known_number_is_recognised_however_it_is_punctuated(written: str) -> None:
    validator = GroundingValidator.for_resources([_resource(phone="555-0123")])

    assert validator.validate(f"Call them at {written}.").passed


@pytest.mark.parametrize(
    "written",
    [
        "https://example.org/oakland-community-food-pantry",
        "https://example.org/oakland-community-food-pantry/",
        "www.example.org/oakland-community-food-pantry",
        "https://example.org/oakland-community-food-pantry.",
    ],
)
def test_a_known_link_is_recognised_however_it_is_written(written: str) -> None:
    validator = GroundingValidator.for_resources([_resource()])

    assert validator.validate(f"Visit {written}").passed


def test_a_resource_without_contact_details_grounds_nothing(validator: GroundingValidator) -> None:
    bare = GroundingValidator.for_resources([_resource(phone=None, website=None)])

    assert not bare.validate("Call 555-0123.").passed


def test_validation_with_no_resources_rejects_every_citation() -> None:
    empty = GroundingValidator.for_resources([])

    assert not empty.validate("See [1].").passed
    assert empty.validate("I have no verified match for that.").passed
