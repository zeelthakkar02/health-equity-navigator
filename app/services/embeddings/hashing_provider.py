"""Offline deterministic embeddings.

The default provider. It hashes normalised word stems and adjacent word pairs
into a fixed-width vector (the "hashing trick"), then L2-normalises the result,
so cosine similarity reflects **shared wording**.

Be clear about what this is: it is lexical overlap, not meaning. It will match
"food assistance" to a food pantry, but not "ride" to "transportation". Its jobs
are to make ingestion, storage, and ranking testable with no credentials and to
keep results reproducible across machines and runs. Real semantic retrieval
requires ``EMBEDDING_PROVIDER=vertex``.

Determinism note: Python's builtin ``hash()`` is salted per process, so a stable
digest (blake2b) is used instead.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Sequence

from app.services.embeddings.base import Embedding, EmbeddingService
from app.services.text_normalization import tokenize

_BIGRAM_WEIGHT = 0.5


def _features(text: str) -> Counter[str]:
    """Unigram stems plus adjacent pairs, so word order carries some weight."""
    tokens = tokenize(text)
    features: Counter[str] = Counter(tokens)
    for first, second in itertools.pairwise(tokens):
        features[f"{first}_{second}"] += _BIGRAM_WEIGHT
    return features


def _bucket(feature: str, dimensions: int) -> tuple[int, float]:
    """Map a feature to a (index, sign) pair with a process-stable digest."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    index = value % dimensions
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return index, sign


class HashingEmbeddingService(EmbeddingService):
    """Deterministic lexical embeddings; no network, no credentials."""

    provider_name = "hashing"
    model_name = "hashing-lexical-v1"

    def __init__(self, dimensions: int = 768) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [self.embed(text) for text in texts]

    async def embed_query(self, text: str) -> Embedding:
        return self.embed(text)

    def embed(self, text: str) -> Embedding:
        """Synchronous embedding, exposed for tests and tooling."""
        vector = [0.0] * self.dimensions
        for feature, count in _features(text).items():
            index, sign = _bucket(feature, self.dimensions)
            # Sublinear term frequency: the tenth mention matters far less than
            # the first, which keeps long descriptions from swamping short ones.
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # Text with no usable tokens; an all-zero vector scores 0 everywhere.
            return vector
        return [value / norm for value in vector]
