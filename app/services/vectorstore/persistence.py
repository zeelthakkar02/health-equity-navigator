"""On-disk cache for an embedded resource index.

Embedding 24 resources through Vertex takes ~12 seconds; a few thousand would
make a cold start untenable. This caches the vectors next to a manifest
describing exactly what produced them.

The cache is a derived artifact, never a source of truth: it is always
rebuildable from the resource data file, and it is discarded — not repaired —
whenever the embedding provider, model, dimensionality, or resource data
changes. A stale vector is worse than a slow start, because nothing downstream
can tell that a resource's text no longer matches its embedding.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.services.vectorstore.base import VectorRecord

logger = logging.getLogger(__name__)

# Bump when the on-disk layout changes so old caches are discarded, not misread.
SCHEMA_VERSION = 1

CacheStatus = Literal["hit", "missing", "stale", "unreadable"]


def fingerprint_file(path: Path | str) -> str:
    """Content hash of the resource data file, so edits invalidate the cache."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest[:32]}"


@dataclass(frozen=True)
class IndexManifest:
    """What produced a cached index."""

    embedding_provider: str
    embedding_model: str
    dimensions: int
    resource_fingerprint: str
    resource_count: int
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""

    def mismatch_reason(self, other: IndexManifest) -> str | None:
        """Why ``other`` cannot satisfy this manifest, or None if it can.

        ``created_at`` and ``resource_count`` are descriptive, not identifying:
        the fingerprint already covers the data, and the count is derived.
        """
        checks = (
            ("schema version", self.schema_version, other.schema_version),
            ("embedding provider", self.embedding_provider, other.embedding_provider),
            ("embedding model", self.embedding_model, other.embedding_model),
            ("dimensions", self.dimensions, other.dimensions),
            ("resource data", self.resource_fingerprint, other.resource_fingerprint),
        )
        for label, expected, found in checks:
            if expected != found:
                return f"{label} changed ({found!r} -> {expected!r})"
        return None


@dataclass(frozen=True)
class CacheLoad:
    """Outcome of a cache lookup."""

    status: CacheStatus
    records: list[VectorRecord] | None = None
    reason: str = ""

    @property
    def is_hit(self) -> bool:
        return self.status == "hit"


class ResourceIndexCache:
    """Reads and writes a cached index as a single compressed .npz file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self, expected: IndexManifest) -> CacheLoad:
        """Return cached records, or explain why they cannot be used."""
        if not self.path.is_file():
            return CacheLoad(status="missing", reason=f"no cache at {self.path}")

        try:
            with np.load(self.path, allow_pickle=False) as archive:
                manifest = IndexManifest(**json.loads(str(archive["manifest"].item())))
                entries = json.loads(str(archive["entries"].item()))
                vectors = archive["vectors"]
        except Exception as exc:  # a corrupt cache must never be fatal
            logger.warning("Discarding unreadable index cache at %s: %s", self.path, exc)
            return CacheLoad(status="unreadable", reason=f"{type(exc).__name__}: {exc}")

        reason = expected.mismatch_reason(manifest)
        if reason:
            return CacheLoad(status="stale", reason=reason)

        if len(entries) != vectors.shape[0]:
            return CacheLoad(
                status="unreadable",
                reason=f"{len(entries)} entries but {vectors.shape[0]} vectors",
            )
        if vectors.shape[1] != expected.dimensions:
            return CacheLoad(
                status="stale",
                reason=f"stored vectors are {vectors.shape[1]}-dimensional",
            )

        records = [
            VectorRecord(
                id=entry["id"],
                embedding=vectors[index].tolist(),
                metadata=entry["metadata"],
                payload=entry["payload"],
            )
            for index, entry in enumerate(entries)
        ]
        logger.info("Loaded %d cached embeddings from %s", len(records), self.path)
        return CacheLoad(status="hit", records=records)

    def save(self, manifest: IndexManifest, records: Sequence[VectorRecord]) -> None:
        """Write the index atomically so a crash cannot leave a torn cache."""
        if not records:
            return

        stamped = IndexManifest(
            **(
                asdict(manifest)
                | {
                    "resource_count": len(records),
                    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
        )
        entries: list[dict[str, Any]] = [
            {"id": record.id, "metadata": dict(record.metadata), "payload": dict(record.payload)}
            for record in records
        ]
        vectors = np.asarray([list(record.embedding) for record in records], dtype=np.float32)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                manifest=np.asarray(json.dumps(asdict(stamped))),
                entries=np.asarray(json.dumps(entries)),
                vectors=vectors,
            )
        temporary.replace(self.path)
        logger.info("Cached %d embeddings to %s", len(records), self.path)

    def invalidate(self) -> None:
        """Delete the cache if present."""
        self.path.unlink(missing_ok=True)
