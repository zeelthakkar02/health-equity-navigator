"""The deterministic crisis screen."""

from __future__ import annotations

import pytest

from app.services import crisis


@pytest.mark.parametrize(
    "message",
    [
        "I want to kill myself.",
        "I have been thinking about suicide.",
        "I feel suicidal and I don't know what to do.",
        "I don't want to live anymore.",
        "I took an overdose of my pills.",
    ],
)
def test_self_harm_language_triggers(message: str) -> None:
    signal = crisis.screen(message)

    assert signal.triggered
    assert signal.kind == "self_harm"


@pytest.mark.parametrize(
    "message",
    [
        "My father has chest pain right now.",
        "She can't breathe and is turning blue.",
        "I think my mother is having a stroke.",
        "He is having a seizure.",
    ],
)
def test_medical_emergency_language_triggers(message: str) -> None:
    signal = crisis.screen(message)

    assert signal.triggered
    assert signal.kind == "medical_emergency"


@pytest.mark.parametrize(
    "message",
    [
        "I need a ride to my dialysis appointment.",
        "I have been feeling depressed and need someone to talk to.",
        "I need help paying my rent.",
        "My mother needs wheelchair-accessible support.",
        "Where can I get free groceries?",
    ],
)
def test_ordinary_needs_do_not_trigger(message: str) -> None:
    """Sadness and hardship are not emergencies; over-triggering helps nobody."""
    assert not crisis.screen(message).triggered


def test_the_notice_leads_with_the_numbers_that_matter() -> None:
    assert "911" in crisis.SAFETY_NOTICE
    assert "988" in crisis.SAFETY_NOTICE
    assert "Suicide & Crisis Lifeline" in crisis.SAFETY_NOTICE


def test_an_empty_message_does_not_trigger() -> None:
    assert not crisis.screen("").triggered
