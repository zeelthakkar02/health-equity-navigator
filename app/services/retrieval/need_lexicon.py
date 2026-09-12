"""Detecting the need a person actually stated.

Embeddings score a query against a whole resource description, which means
incidental context can outweigh the explicit ask. "My mother needs
wheelchair-accessible support" is about a wheelchair; "my mother" is context,
not the need. Left to similarity alone, caregiver organisations rank high
because the sentence *sounds* like a caregiving situation.

This module reads the explicit signals out of a query with a small keyword
lexicon, so the retriever can boost the categories a person actually named.

Two rules make it work:

1. **Kinship words are never triggers.** "mother", "husband", "my son" say who
   the help is for, not what is needed. Caregiver support triggers on caregiving
   language — "caring for", "respite", "caregiver" — and nothing else.
2. **Ambiguous words are only triggers inside a phrase.** A bare "appointment"
   appears in half of all queries, including "a ride to my appointment", so it
   signals nothing; "missed my appointment" and "reschedule" do.

No model call, no classifier — just a lookup, so it stays cheap and inspectable.
"""

from __future__ import annotations

from app.domain.resource import ServiceCategory
from app.services.text_normalization import normalize

# Surface forms are normalised through the same stemmer as queries, so plurals
# and common verb forms are handled without listing every variant.
_TRIGGERS: dict[ServiceCategory, tuple[str, ...]] = {
    ServiceCategory.TRANSPORTATION: (
        "ride",
        "rides",
        "a ride",
        "transportation",
        "transport",
        "bus pass",
        "bus fare",
        "shuttle",
        "paratransit",
        "gas card",
        "cannot drive",
        "can't drive",
        "no car",
        "get me to",
        "drive me",
    ),
    ServiceCategory.ACCESSIBILITY_SUPPORT: (
        "wheelchair",
        "accessible",
        "accessibility",
        "ramp",
        "walker",
        "mobility",
        "grab bar",
        "shower chair",
        "stair lift",
        "disability",
        "disabled",
        "braille",
        "large print",
        "deaf",
        "hard of hearing",
        "blind",
        "accommodation",
    ),
    ServiceCategory.FOOD_ASSISTANCE: (
        "food",
        "groceries",
        "grocery",
        "hungry",
        "hunger",
        "meal",
        "meals",
        "pantry",
        "calfresh",
        "snap benefits",
        "wic",
        "nutrition",
        "eat",
    ),
    ServiceCategory.HOUSING_SUPPORT: (
        "rent",
        "eviction",
        "evicted",
        "housing",
        "shelter",
        "homeless",
        "landlord",
        "tenant",
        "place to stay",
        "lose my home",
        "utilities shutoff",
    ),
    ServiceCategory.CAREGIVER_SUPPORT: (
        # Caregiving language only — never a kinship noun on its own.
        "caregiver",
        "caregiving",
        "caring for",
        "care for my",
        "respite",
        "taking care of",
        "take care of my",
        "looking after",
        "a break from caring",
    ),
    ServiceCategory.MENTAL_HEALTH: (
        "depressed",
        "depression",
        "anxiety",
        "anxious",
        "mental health",
        "counseling",
        "counselling",
        "counselor",
        "therapy",
        "therapist",
        "psychiatric",
        "crisis",
        "someone to talk to",
        "grief",
        "substance use",
        "recovery",
    ),
    ServiceCategory.INSURANCE_NAVIGATION: (
        "insurance",
        "medi cal",
        "medicaid",
        "medicare",
        "coverage",
        "enroll",
        "premium",
        "denied claim",
        "claim",
        "medical bill",
        "hospital bill",
        "billing",
        "medical debt",
        "copay",
        "charity care",
        "uninsured",
    ),
    ServiceCategory.LANGUAGE_ASSISTANCE: (
        "interpreter",
        "interpretation",
        "translate",
        "translation",
        "translator",
        "asl",
        "sign language",
        "does not speak english",
        "doesn't speak english",
        "limited english",
        "in my language",
    ),
    ServiceCategory.COMMUNITY_CLINIC: (
        "clinic",
        "checkup",
        "check up",
        "primary care",
        "dentist",
        "dental",
        "pediatrician",
        "see a doctor",
        "need a doctor",
        "find a doctor",
        "vaccination",
        "screening",
        "prescription",
    ),
    ServiceCategory.APPOINTMENT_SUPPORT: (
        # Never a bare "appointment": it rides along with almost every need.
        "reschedule",
        "rescheduling",
        "missed my appointment",
        "miss my appointment",
        "missing my appointment",
        "keep missing",
        "childcare",
        "child care",
        "paperwork",
        "come with me",
        "remind me",
    ),
    ServiceCategory.COMMUNITY_SERVICES: (
        "utility bill",
        "electric bill",
        "legal aid",
        "job training",
        "community center",
        "benefits",
        "social services",
    ),
}

# Normalised once at import; matching is a substring test on stemmed text.
_NORMALISED_TRIGGERS: dict[ServiceCategory, tuple[str, ...]] = {
    category: tuple(normalized for phrase in phrases if (normalized := normalize(phrase).strip()))
    for category, phrases in _TRIGGERS.items()
}


def detect_categories(query: str) -> set[ServiceCategory]:
    """Return the service categories explicitly named in ``query``.

    An empty set means the query stated no recognisable need, in which case the
    retriever falls back to similarity alone.
    """
    normalized = normalize(query)
    return {
        category
        for category, triggers in _NORMALISED_TRIGGERS.items()
        if any(f" {trigger} " in normalized for trigger in triggers)
    }


def matched_triggers(query: str) -> dict[ServiceCategory, list[str]]:
    """Which phrases fired, for debugging and evaluation output."""
    normalized = normalize(query)
    matches: dict[ServiceCategory, list[str]] = {}
    for category, triggers in _NORMALISED_TRIGGERS.items():
        hits = [trigger for trigger in triggers if f" {trigger} " in normalized]
        if hits:
            matches[category] = hits
    return matches
