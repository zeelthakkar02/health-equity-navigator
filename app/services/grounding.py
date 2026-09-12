"""Post-generation grounding validation.

Checks that a generated answer only refers to organizations, phone numbers, and
links that were actually in the retrieved context. A model that invents a
plausible-sounding clinic and a plausible-sounding phone number sends someone to
a door that is not there, which is worse than saying nothing.

Three checks, in descending order of confidence:

* **Contact details** — exact. Every phone number and URL in the answer must
  appear in the retrieved resources (or be a public emergency/referral number).
* **Citation markers** — exact. Every ``[n]`` must index a resource in context.
* **Organization names** — heuristic. Capitalized spans ending in an
  organization word are matched against the retrieved names.

The name check is conservative and cannot prove the absence of hallucination: it
catches an invented organization, not a subtly wrong claim about a real one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from app.domain.resource import Resource

IssueKind = Literal["unknown_organization", "unknown_phone", "unknown_url", "invalid_citation"]

# Words that make a capitalized span look like an organization name.
_ORGANISATION_WORDS = (
    "center",
    "centre",
    "clinic",
    "program",
    "programme",
    "project",
    "network",
    "coalition",
    "alliance",
    "collective",
    "fund",
    "pantry",
    "hub",
    "line",
    "unit",
    "association",
    "foundation",
    "society",
    "agency",
    "bank",
    "department",
    "institute",
    "partnership",
    "initiative",
    "council",
)

# Public numbers and programs a grounded answer may reference without a resource.
_PUBLIC_PHONES = frozenset({"911", "988", "211", "741741"})
_PUBLIC_HOSTS = frozenset({"211.org", "988lifeline.org", "findahealthcenter.hrsa.gov"})
_PUBLIC_NAMES = frozenset(
    {
        "suicide crisis lifeline",
        "988 suicide crisis lifeline",
        "health equity navigator",
        "social services",
        "community services",
        "health services",
        "support services",
        "human services",
        "emergency services",
        "language services",
        "interpretation services",
        "medical services",
    }
)

# Words that can precede a name in a sentence without being part of it. Only a
# fixed list is stripped: dropping arbitrary leading words would let an invented
# "Berkeley Food Pantry" pass by matching the tail of a real name.
_LEADING_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "try",
        "call",
        "contact",
        "visit",
        "at",
        "their",
        "this",
        "that",
        "also",
        "and",
        "or",
        "to",
        "see",
        "ask",
        "for",
        "from",
        "with",
    }
)

_PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SHORTCODE_PATTERN = re.compile(r"\b(?:911|988|211|741741)\b")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\]\)\"']+", re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")
_NAME_PATTERN = re.compile(r"\b(?:[A-Z][\w'&.-]*(?:\s+(?:of|for|and|the|to|de|la)\s+)?\s*){1,7}")


@dataclass(frozen=True)
class GroundingIssue:
    kind: IssueKind
    detail: str


@dataclass(frozen=True)
class GroundingResult:
    """Whether an answer stayed inside its context."""

    passed: bool
    issues: list[GroundingIssue] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return "grounded"
        return "; ".join(f"{issue.kind}: {issue.detail}" for issue in self.issues)


class GroundingValidator:
    """Validates generated text against the resources it was given."""

    def __init__(
        self,
        *,
        organization_names: Sequence[str],
        phones: Sequence[str],
        urls: Sequence[str],
        citation_count: int,
    ) -> None:
        self._names = {_normalise_name(name) for name in organization_names}
        self._names.discard("")
        self._phones = {digits for phone in phones if (digits := _digits(phone))}
        self._urls = {key for url in urls if (key := _normalise_url(url))}
        self._citation_count = citation_count

    @classmethod
    def for_resources(cls, resources: Sequence[Resource]) -> GroundingValidator:
        return cls(
            organization_names=[resource.organization_name for resource in resources],
            phones=[resource.phone for resource in resources if resource.phone],
            urls=[resource.website for resource in resources if resource.website],
            citation_count=len(resources),
        )

    def validate(self, text: str) -> GroundingResult:
        issues: list[GroundingIssue] = []
        issues.extend(self._check_citations(text))
        issues.extend(self._check_phones(text))
        issues.extend(self._check_urls(text))
        issues.extend(self._check_names(text))
        return GroundingResult(passed=not issues, issues=issues)

    def _check_citations(self, text: str) -> list[GroundingIssue]:
        return [
            GroundingIssue(
                "invalid_citation",
                f"[{marker}] but only {self._citation_count} resources were provided",
            )
            for marker in _CITATION_PATTERN.findall(text)
            if not 1 <= int(marker) <= self._citation_count
        ]

    def _check_phones(self, text: str) -> list[GroundingIssue]:
        issues = []
        for candidate in _PHONE_PATTERN.findall(text):
            number = _digits(candidate)
            if not number or number in _PUBLIC_PHONES or number in self._phones:
                continue
            # A resource phone may be quoted with different punctuation.
            if any(number in known or known in number for known in self._phones):
                continue
            issues.append(GroundingIssue("unknown_phone", candidate.strip()))
        return issues

    def _check_urls(self, text: str) -> list[GroundingIssue]:
        issues = []
        for candidate in _URL_PATTERN.findall(text):
            key = _normalise_url(candidate)
            if not key:
                continue
            host = key.split("/", 1)[0]
            if host in _PUBLIC_HOSTS or key in self._urls:
                continue
            if any(key.startswith(known) or known.startswith(key) for known in self._urls):
                continue
            issues.append(GroundingIssue("unknown_url", candidate.rstrip(".,;:")))
        return issues

    def _check_names(self, text: str) -> list[GroundingIssue]:
        issues = []
        for candidate in _organization_candidates(text):
            normalized = _strip_leading_words(_normalise_name(candidate))
            if not normalized or normalized in _PUBLIC_NAMES:
                continue
            if self._is_known_name(normalized):
                continue
            issues.append(GroundingIssue("unknown_organization", normalized))
        return issues

    def _is_known_name(self, normalized: str) -> bool:
        if not normalized:
            return True
        return any(
            normalized == known or normalized in known or known in normalized
            for known in self._names
        )


def _organization_candidates(text: str) -> list[str]:
    """Capitalized spans that end in an organization word."""
    candidates = []
    for match in _NAME_PATTERN.finditer(text):
        span = match.group().strip(" .,:;\n")
        words = span.split()
        if len(words) < 2:
            continue
        if words[-1].lower().strip(".,") not in _ORGANISATION_WORDS:
            continue
        # Drop a leading sentence-start word that is not part of the name.
        candidates.append(span)
    return candidates


def _strip_leading_words(name: str) -> str:
    """Drop sentence scaffolding like 'try the' from the front of a name."""
    words = name.split()
    while words and words[0] in _LEADING_WORDS:
        words.pop(0)
    return " ".join(words) if len(words) >= 2 else ""


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", name.lower()).strip().replace("  ", " ")


def _digits(value: str) -> str:
    number = re.sub(r"\D", "", value)
    return number[1:] if len(number) == 11 and number.startswith("1") else number


def _normalise_url(value: str) -> str:
    cleaned = value.strip().rstrip(".,;:)")
    cleaned = re.sub(r"^https?://", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^www\.", "", cleaned, flags=re.IGNORECASE)
    return cleaned.rstrip("/").lower()
