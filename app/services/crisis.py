"""Deterministic crisis screen.

Runs before retrieval and before any model call. If someone describes an
emergency or thoughts of self-harm, the right response is immediate, fixed, and
not generated: emergency numbers first, human escalation flagged, no model in
the loop deciding what to say.

Keyword matching is crude and will miss things. It is a floor, not a safety net:
it exists so the obvious cases can never depend on a model's judgement.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.text_normalization import normalize

_SELF_HARM_PHRASES = (
    "kill myself",
    "killing myself",
    "end my life",
    "want to die",
    "wanna die",
    "suicide",
    "suicidal",
    "take my own life",
    "hurt myself",
    "harm myself",
    "not want to live",
    "don't want to live",
    "better off dead",
    "overdose",
)

_MEDICAL_EMERGENCY_PHRASES = (
    "chest pain",
    "cannot breathe",
    "can't breathe",
    "trouble breathing",
    "heart attack",
    "stroke",
    "unconscious",
    "not breathing",
    "bleeding badly",
    "severe bleeding",
    "seizure",
)

_NORMALISED: dict[str, tuple[str, ...]] = {
    "self_harm": tuple(normalize(phrase).strip() for phrase in _SELF_HARM_PHRASES),
    "medical_emergency": tuple(normalize(phrase).strip() for phrase in _MEDICAL_EMERGENCY_PHRASES),
}

SAFETY_NOTICE = (
    "It sounds like this may be an emergency, and I want to make sure you get help "
    "from a person right now rather than from me.\n\n"
    "• If you are in immediate danger or this is a medical emergency, call 911.\n"
    "• If you are thinking about harming yourself or are in emotional crisis, call "
    "or text 988 to reach the Suicide & Crisis Lifeline, any time of day.\n"
    "• If you would rather chat, 988lifeline.org has a chat option.\n\n"
    "You deserve support from someone who can stay with you through this. Please "
    "reach out to one of the numbers above."
)


@dataclass(frozen=True)
class CrisisSignal:
    """Outcome of the crisis screen."""

    triggered: bool
    kind: str = ""
    matched: str = ""


def screen(query: str) -> CrisisSignal:
    """Check a query for emergency or self-harm language."""
    normalized = normalize(query)
    for kind, phrases in _NORMALISED.items():
        for phrase in phrases:
            if phrase and f" {phrase} " in normalized:
                return CrisisSignal(triggered=True, kind=kind, matched=phrase)
    return CrisisSignal(triggered=False)
