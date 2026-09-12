"""Semantic retrieval of verified community resources.

    user need (+ optional location) -> embedded query -> vector search
        -> verification & freshness policy -> location-aware re-rank -> top K

Three deliberate choices:

* **Location boosts, it does not filter.** Someone in Oakland who needs a
  statewide caregiver line should still see it. Hard-filtering on place is how a
  resource directory quietly fails the people with the fewest options.
* **A stated need boosts too.** Similarity alone lets incidental context
  outrank the explicit ask — see :mod:`app.services.retrieval.need_lexicon`.
* **The relevance gate applies to the semantic score only**, before any boost.
  Being nearby, or naming a category, must never drag an irrelevant resource
  into the results.

Nothing here talks to a generation model. Connecting these results to Gemini is
the next phase.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

from app.domain.resource import Resource, ServiceCategory
from app.services.embeddings.base import EmbeddingService
from app.services.retrieval.need_lexicon import detect_categories
from app.services.vectorstore.base import SearchFilters, SearchHit, VectorStore

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[a-z0-9]+")

# How strongly each kind of place match counts, before the configured boost.
_POSTAL_MATCH = 1.0
_CITY_MATCH = 1.0
_COUNTY_MATCH = 0.8
_STATE_MATCH = 0.6
_BOUNDLESS_MATCH = 0.5


@dataclass(frozen=True)
class RetrievedResource:
    """A resource returned for a need, with the reasoning behind its rank.

    The scores are internal ranking signals. They are useful for evaluation and
    debugging and must not be exposed through the public API.
    """

    resource: Resource
    score: float
    semantic_score: float
    location_match: float
    rank: int
    category_match: float = 0.0

    @property
    def matched_location(self) -> bool:
        return self.location_match > 0.0

    @property
    def matched_stated_need(self) -> bool:
        return self.category_match > 0.0


class ResourceRetriever:
    """Finds the verified resources most relevant to a stated need."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        store: VectorStore,
        *,
        top_k: int = 5,
        min_score: float = 0.05,
        location_boost: float = 0.25,
        category_boost: float = 0.12,
        candidate_multiplier: int = 4,
        max_resource_age_days: int = 548,
        require_verified: bool = True,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._location_boost = location_boost
        self._category_boost = category_boost
        self._candidate_multiplier = candidate_multiplier
        self._max_resource_age_days = max_resource_age_days
        self._require_verified = require_verified

    async def retrieve(
        self,
        query: str,
        *,
        location: str | None = None,
        categories: list[ServiceCategory] | None = None,
        languages: list[str] | None = None,
        top_k: int | None = None,
        as_of: date | None = None,
    ) -> list[RetrievedResource]:
        """Return the most relevant verified resources for ``query``."""
        query = query.strip()
        if not query:
            return []

        limit = top_k or self._top_k
        embedding = await self._embeddings.embed_query(query)

        hits = await self._store.search(
            embedding,
            # Over-fetch: policy filtering and re-ranking happen after scoring,
            # so the store must return more than the caller ultimately wants.
            top_k=max(limit * self._candidate_multiplier, limit),
            filters=_build_filters(categories, languages),
        )

        stated_needs = detect_categories(query)
        results = self._rank(hits, location=location, stated_needs=stated_needs, as_of=as_of)[
            :limit
        ]
        # The query text itself is never logged: it can describe someone's
        # health, immigration status, or housing situation. Its length and the
        # needs detected from it are enough to debug retrieval.
        logger.info(
            "retrieval query_chars=%d has_location=%s stated_needs=%s hits=%d returned=%d",
            len(query),
            location is not None,
            sorted(category.value for category in stated_needs),
            len(hits),
            len(results),
        )
        return results

    def _rank(
        self,
        hits: list[SearchHit],
        *,
        location: str | None,
        stated_needs: set[ServiceCategory],
        as_of: date | None,
    ) -> list[RetrievedResource]:
        location_tokens = _tokenize(location) if location else frozenset()
        scored: list[_ScoredResource] = []

        for hit in hits:
            if hit.score < self._min_score:
                continue

            resource = Resource.model_validate(hit.payload)
            if not self._passes_policy(resource, as_of=as_of):
                continue

            location_match = _location_match(resource, location_tokens) if location_tokens else 0.0
            category_match = (
                1.0
                if stated_needs and stated_needs.intersection(resource.service_categories)
                else 0.0
            )
            final_score = (
                hit.score
                + self._location_boost * location_match
                + self._category_boost * category_match
            )
            scored.append(
                _ScoredResource(
                    final_score=final_score,
                    semantic_score=hit.score,
                    location_match=location_match,
                    category_match=category_match,
                    resource=resource,
                )
            )

        scored.sort(key=lambda item: (-item.final_score, item.resource.resource_id))

        return [
            RetrievedResource(
                resource=item.resource,
                score=round(item.final_score, 6),
                semantic_score=round(item.semantic_score, 6),
                location_match=item.location_match,
                category_match=item.category_match,
                rank=rank,
            )
            for rank, item in enumerate(scored, start=1)
        ]

    def _passes_policy(self, resource: Resource, *, as_of: date | None) -> bool:
        """Withhold anything unverified or overdue for re-verification."""
        if self._require_verified and not resource.is_verified:
            return False
        return not resource.is_stale(max_age_days=self._max_resource_age_days, as_of=as_of)


@dataclass(frozen=True)
class _ScoredResource:
    """Intermediate ranking row, before ranks are assigned."""

    final_score: float
    semantic_score: float
    location_match: float
    category_match: float
    resource: Resource


def _build_filters(
    categories: list[ServiceCategory] | None,
    languages: list[str] | None,
) -> SearchFilters | None:
    any_of: dict[str, frozenset[str]] = {}
    if categories:
        any_of["categories"] = frozenset(category.value for category in categories)
    if languages:
        any_of["languages"] = frozenset(language.lower().strip() for language in languages)
    return SearchFilters(any_of=any_of) if any_of else None


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_WORD_PATTERN.findall(text.lower()))


def _phrase_matches(phrase: str, tokens: frozenset[str]) -> bool:
    """True when every word of a place name appears in the user's location."""
    phrase_tokens = _tokenize(phrase)
    return bool(phrase_tokens) and phrase_tokens.issubset(tokens)


def _location_match(resource: Resource, location_tokens: frozenset[str]) -> float:
    """Score how well a resource's service area covers the stated location."""
    area = resource.service_area

    if any(code.lower() in location_tokens for code in area.postal_codes):
        return _POSTAL_MATCH
    if any(_phrase_matches(city, location_tokens) for city in area.cities):
        return _CITY_MATCH
    if any(_phrase_matches(county, location_tokens) for county in area.counties):
        return _COUNTY_MATCH
    if area.state and _phrase_matches(area.state, location_tokens):
        return _STATE_MATCH
    if area.scope.is_boundless:
        # Serves everyone, so it is never wrong — just less specific than a
        # neighbourhood organisation that matched by name.
        return _BOUNDLESS_MATCH
    return 0.0
