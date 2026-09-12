"""Detecting the need a person explicitly stated."""

from __future__ import annotations

import pytest

from app.domain.resource import ServiceCategory
from app.services.retrieval.need_lexicon import detect_categories, matched_triggers


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("I need a ride to my dialysis appointment.", ServiceCategory.TRANSPORTATION),
        ("My mother needs wheelchair-accessible support.", ServiceCategory.ACCESSIBILITY_SUPPORT),
        ("We are running out of groceries.", ServiceCategory.FOOD_ASSISTANCE),
        ("I got an eviction notice.", ServiceCategory.HOUSING_SUPPORT),
        ("I'm caring for my husband and need a break.", ServiceCategory.CAREGIVER_SUPPORT),
        ("I have been feeling depressed.", ServiceCategory.MENTAL_HEALTH),
        ("I need help signing up for Medi-Cal.", ServiceCategory.INSURANCE_NAVIGATION),
        ("I need an interpreter for my visit.", ServiceCategory.LANGUAGE_ASSISTANCE),
        ("Where can I get a checkup?", ServiceCategory.COMMUNITY_CLINIC),
        ("I keep missing my appointments.", ServiceCategory.APPOINTMENT_SUPPORT),
        ("I need help with my utility bill.", ServiceCategory.COMMUNITY_SERVICES),
    ],
)
def test_explicit_needs_are_detected(query: str, expected: ServiceCategory) -> None:
    assert expected in detect_categories(query)


def test_kinship_alone_is_never_a_caregiver_signal() -> None:
    """The whole point: 'my mother' says who, not what."""
    for query in (
        "My mother needs wheelchair-accessible support.",
        "My father does not speak English.",
        "My son needs a checkup.",
        "My wife needs a ride to the clinic.",
    ):
        assert ServiceCategory.CAREGIVER_SUPPORT not in detect_categories(query), query


def test_caregiving_language_does_signal_caregiver_support() -> None:
    for query in (
        "I'm caring for my mother.",
        "I am a caregiver and need respite.",
        "I take care of my elderly father.",
    ):
        assert ServiceCategory.CAREGIVER_SUPPORT in detect_categories(query), query


def test_a_bare_appointment_mention_signals_nothing() -> None:
    """'Appointment' rides along with most needs, so it must not trigger."""
    detected = detect_categories("I need a ride to my appointment.")

    assert ServiceCategory.APPOINTMENT_SUPPORT not in detected
    assert ServiceCategory.TRANSPORTATION in detected


def test_appointment_support_needs_a_real_phrase() -> None:
    assert ServiceCategory.APPOINTMENT_SUPPORT in detect_categories(
        "I keep missing my appointment because of childcare."
    )


def test_off_topic_queries_state_no_need() -> None:
    for query in (
        "What is the weather forecast for tomorrow?",
        "Who won the football game last night?",
        "How do I fix a null pointer bug in my Python code?",
    ):
        assert detect_categories(query) == set(), query


def test_a_query_can_state_more_than_one_need() -> None:
    detected = detect_categories("I am Deaf and need an ASL interpreter.")

    assert ServiceCategory.LANGUAGE_ASSISTANCE in detected
    assert ServiceCategory.ACCESSIBILITY_SUPPORT in detected


def test_plurals_and_verb_forms_are_handled() -> None:
    assert ServiceCategory.TRANSPORTATION in detect_categories("I need rides.")
    assert ServiceCategory.TRANSPORTATION in detect_categories("I need a ride.")


def test_matched_triggers_explains_the_decision() -> None:
    matches = matched_triggers("My mother needs wheelchair-accessible support.")

    assert ServiceCategory.ACCESSIBILITY_SUPPORT in matches
    assert "wheelchair" in matches[ServiceCategory.ACCESSIBILITY_SUPPORT]
    assert ServiceCategory.CAREGIVER_SUPPORT not in matches
