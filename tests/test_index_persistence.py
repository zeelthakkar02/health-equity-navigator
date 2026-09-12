"""Persisting the embedded resource index."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.retrieval.bootstrap import bootstrap_retrieval, build_manifest
from app.services.vectorstore.base import VectorRecord
from app.services.vectorstore.memory_store import InMemoryVectorStore
from app.services.vectorstore.persistence import (
    IndexManifest,
    ResourceIndexCache,
    fingerprint_file,
)


def _manifest(**overrides: object) -> IndexManifest:
    base: dict[str, object] = {
        "embedding_provider": "hashing",
        "embedding_model": "hashing-lexical-v1",
        "dimensions": 4,
        "resource_fingerprint": "sha256:abc123",
        "resource_count": 2,
    }
    return IndexManifest(**(base | overrides))  # type: ignore[arg-type]


def _records() -> list[VectorRecord]:
    return [
        VectorRecord(
            id="a",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"categories": ["food_assistance"]},
            payload={"resource_id": "a", "organization_name": "A"},
        ),
        VectorRecord(
            id="b",
            embedding=[0.0, 1.0, 0.0, 0.0],
            metadata={"categories": ["housing_support"]},
            payload={"resource_id": "b", "organization_name": "B"},
        ),
    ]


def test_a_saved_index_round_trips(tmp_path: Path) -> None:
    cache = ResourceIndexCache(tmp_path / "index.npz")
    cache.save(_manifest(), _records())

    loaded = cache.load(_manifest())

    assert loaded.is_hit
    assert loaded.records is not None
    assert [record.id for record in loaded.records] == ["a", "b"]
    assert loaded.records[0].payload == {"resource_id": "a", "organization_name": "A"}
    assert loaded.records[0].metadata == {"categories": ["food_assistance"]}
    assert loaded.records[1].embedding == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_a_missing_cache_is_a_clean_miss(tmp_path: Path) -> None:
    loaded = ResourceIndexCache(tmp_path / "absent.npz").load(_manifest())

    assert loaded.status == "missing"
    assert loaded.records is None


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("embedding_provider", "vertex", "embedding provider changed"),
        ("embedding_model", "gemini-embedding-001", "embedding model changed"),
        ("dimensions", 768, "dimensions changed"),
        ("resource_fingerprint", "sha256:different", "resource data changed"),
        ("schema_version", 99, "schema version changed"),
    ],
)
def test_the_cache_is_discarded_when_anything_identifying_changes(
    tmp_path: Path, field: str, value: object, expected_reason: str
) -> None:
    cache = ResourceIndexCache(tmp_path / "index.npz")
    cache.save(_manifest(), _records())

    loaded = cache.load(_manifest(**{field: value}))

    assert loaded.status == "stale"
    assert expected_reason in loaded.reason
    assert loaded.records is None


def test_a_corrupt_cache_is_discarded_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    path.write_bytes(b"this is not an npz archive")

    loaded = ResourceIndexCache(path).load(_manifest())

    assert loaded.status == "unreadable"
    assert loaded.records is None


def test_saving_nothing_writes_no_file(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    ResourceIndexCache(path).save(_manifest(), [])

    assert not path.exists()


def test_invalidate_removes_the_cache(tmp_path: Path) -> None:
    cache = ResourceIndexCache(tmp_path / "index.npz")
    cache.save(_manifest(), _records())

    cache.invalidate()
    cache.invalidate()  # idempotent

    assert cache.load(_manifest()).status == "missing"


def test_the_fingerprint_follows_file_content(tmp_path: Path) -> None:
    path = tmp_path / "resources.json"
    path.write_text("[]", encoding="utf-8")
    before = fingerprint_file(path)

    path.write_text('[{"resource_id": "x"}]', encoding="utf-8")

    assert fingerprint_file(path) != before


# --- bootstrap integration -------------------------------------------------


def _cached_settings(settings: Settings, tmp_path: Path) -> Settings:
    return settings.model_copy(
        update={
            "resource_index_cache_enabled": True,
            "resource_index_cache_path": tmp_path / "index.npz",
        }
    )


async def test_bootstrap_writes_then_reuses_the_index(settings: Settings, tmp_path: Path) -> None:
    cached = _cached_settings(settings, tmp_path)

    _, first = await bootstrap_retrieval(cached, InMemoryVectorStore())
    _, second = await bootstrap_retrieval(cached, InMemoryVectorStore())

    assert first.source == "embedded"
    assert second.source == "cache"
    assert second.indexed == first.indexed == 24


async def test_editing_the_resource_file_rebuilds_the_index(
    settings: Settings, tmp_path: Path
) -> None:
    resources = tmp_path / "resources.json"
    resources.write_text(
        Path(settings.resource_data_path).read_text(encoding="utf-8"), encoding="utf-8"
    )
    cached = _cached_settings(settings, tmp_path).model_copy(
        update={"resource_data_path": resources}
    )

    _, first = await bootstrap_retrieval(cached, InMemoryVectorStore())
    resources.write_text(
        resources.read_text(encoding="utf-8").replace("Oakland Community Food Pantry", "Renamed"),
        encoding="utf-8",
    )
    _, second = await bootstrap_retrieval(cached, InMemoryVectorStore())

    assert first.source == "embedded"
    assert second.source == "embedded"
    assert "resource data changed" in second.cache_note


async def test_the_cache_can_be_turned_off(settings: Settings, tmp_path: Path) -> None:
    disabled = _cached_settings(settings, tmp_path).model_copy(
        update={"resource_index_cache_enabled": False}
    )

    _, report = await bootstrap_retrieval(disabled, InMemoryVectorStore())

    assert report.source == "embedded"
    assert not (tmp_path / "index.npz").exists()


def test_the_manifest_describes_the_current_configuration(settings: Settings) -> None:
    from app.services.embeddings.factory import build_embedding_service

    manifest = build_manifest(settings, build_embedding_service(settings))

    assert manifest.embedding_provider == "hashing"
    assert manifest.dimensions == 768
    assert manifest.resource_fingerprint.startswith("sha256:")
