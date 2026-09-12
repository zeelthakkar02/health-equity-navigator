"""Resource ingestion: loading, document building, and indexing."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.core.config import Settings
from app.domain.resource import Resource, ServiceArea, ServiceCategory, ServiceScope
from app.services.embeddings.hashing_provider import HashingEmbeddingService
from app.services.ingestion.documents import (
    build_embedding_text,
    build_metadata,
    build_payload,
)
from app.services.ingestion.loader import (
    ResourceFileError,
    load_resources,
    load_resources_from_file,
)
from app.services.ingestion.pipeline import ResourceIngestionPipeline
from app.services.vectorstore.memory_store import InMemoryVectorStore

VALID_RECORD: dict[str, object] = {
    "resource_id": "syn-test-001",
    "organization_name": "Example Food Pantry",
    "description": "A synthetic food pantry used only in tests.",
    "service_categories": ["food_assistance"],
    "source": "test-fixture",
    "last_verified": "2026-06-01",
}


# --- loading ---------------------------------------------------------------


def test_sample_file_loads_cleanly(settings: Settings) -> None:
    result = load_resources_from_file(settings.resource_data_path)

    assert result.loaded_count == 24
    assert result.rejected_count == 0
    assert all(resource.is_verified for resource in result.resources)


def test_sample_file_is_clearly_marked_synthetic(settings: Settings) -> None:
    document = json.loads(Path(settings.resource_data_path).read_text(encoding="utf-8"))

    assert document["_meta"]["contains_patient_data"] is False
    assert "SYNTHETIC" in document["_meta"]["warning"]
    assert all(resource["source"] == "synthetic-sample-data" for resource in document["resources"])


def test_sample_file_covers_every_service_category(settings: Settings) -> None:
    result = load_resources_from_file(settings.resource_data_path)
    covered = {
        category for resource in result.resources for category in resource.service_categories
    }

    assert covered == set(ServiceCategory)


def test_an_invalid_record_is_reported_without_sinking_the_batch() -> None:
    result = load_resources([VALID_RECORD, {"resource_id": "broken"}])

    assert result.loaded_count == 1
    assert result.rejected_count == 1
    assert result.issues[0].resource_id == "broken"
    assert "organization_name" in result.issues[0].reason


def test_duplicate_ids_are_rejected() -> None:
    result = load_resources([VALID_RECORD, dict(VALID_RECORD)])

    assert result.loaded_count == 1
    assert result.issues[0].reason == "duplicate resource_id"


def test_non_object_records_are_rejected() -> None:
    result = load_resources(["not a record"])  # type: ignore[list-item]

    assert result.loaded_count == 0
    assert "not an object" in result.issues[0].reason


def test_a_missing_file_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(ResourceFileError, match="not found"):
        load_resources_from_file(tmp_path / "nope.json")


def test_malformed_json_is_an_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ResourceFileError, match="not valid JSON"):
        load_resources_from_file(path)


def test_a_file_without_resources_is_an_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"_meta": {}}), encoding="utf-8")

    with pytest.raises(ResourceFileError, match="no 'resources' key"):
        load_resources_from_file(path)


def test_a_bare_list_of_records_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text(json.dumps([VALID_RECORD]), encoding="utf-8")

    assert load_resources_from_file(path).loaded_count == 1


# --- document building -----------------------------------------------------


def _sample_resource() -> Resource:
    return Resource.model_validate(
        VALID_RECORD
        | {
            "service_categories": ["food_assistance", "transportation"],
            "services": ["grocery distribution"],
            "service_area": {
                "scope": ServiceScope.LOCAL,
                "cities": ["Oakland"],
                "state": "CA",
                "postal_codes": ["94601"],
            },
            "eligibility": "Open to all residents.",
            "languages": ["English", "Spanish"],
            "accessibility": ["wheelchair accessible"],
            "cost": "free",
        }
    )


def test_embedding_text_carries_what_people_search_with() -> None:
    text = build_embedding_text(_sample_resource())

    assert "Example Food Pantry" in text
    assert "food assistance" in text  # the category label, not just its code
    assert "grocery distribution" in text
    assert "Oakland, CA" in text
    assert "Spanish" in text
    assert "wheelchair accessible" in text


def test_metadata_is_flat_and_lowercased_for_filtering() -> None:
    metadata = build_metadata(_sample_resource())

    assert metadata["categories"] == ["food_assistance", "transportation"]
    assert metadata["cities"] == ["oakland"]
    assert metadata["state"] == "ca"
    assert metadata["languages"] == ["english", "spanish"]
    assert metadata["verification_status"] == "verified"


def test_payload_round_trips_back_into_a_resource() -> None:
    original = _sample_resource()

    restored = Resource.model_validate(build_payload(original))

    assert restored == original


# --- pipeline --------------------------------------------------------------


async def test_pipeline_indexes_the_sample_file(settings: Settings) -> None:
    store = InMemoryVectorStore()
    pipeline = ResourceIngestionPipeline(HashingEmbeddingService(dimensions=256), store)

    report = await pipeline.ingest_from_file(settings.resource_data_path)

    assert report.loaded == 24
    assert report.indexed == 24
    assert report.rejected == 0
    assert report.dimensions == 256
    assert report.embedding_provider == "hashing"
    assert await store.count() == 24
    assert store.dimensions == 256


async def test_re_ingesting_replaces_rather_than_duplicates(settings: Settings) -> None:
    store = InMemoryVectorStore()
    pipeline = ResourceIngestionPipeline(HashingEmbeddingService(dimensions=256), store)

    await pipeline.ingest_from_file(settings.resource_data_path)
    await pipeline.ingest_from_file(settings.resource_data_path)

    assert await store.count() == 24


async def test_ingesting_nothing_is_not_an_error() -> None:
    store = InMemoryVectorStore()
    pipeline = ResourceIngestionPipeline(HashingEmbeddingService(), store)

    report = await pipeline.ingest([])

    assert report.indexed == 0
    assert await store.count() == 0


async def test_unverified_records_are_indexed_and_left_to_retrieval_policy() -> None:
    """Ingestion is policy-free; withholding happens at retrieval time."""
    store = InMemoryVectorStore()
    pipeline = ResourceIngestionPipeline(HashingEmbeddingService(), store)
    unverified = Resource.model_validate(
        VALID_RECORD | {"resource_id": "syn-test-002", "verification_status": "unverified"}
    )

    await pipeline.ingest([unverified])

    assert await store.count() == 1


async def test_reported_issues_survive_into_the_report(tmp_path: Path) -> None:
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps({"resources": [VALID_RECORD, {"resource_id": "broken"}]}), encoding="utf-8"
    )
    pipeline = ResourceIngestionPipeline(HashingEmbeddingService(), InMemoryVectorStore())

    report = await pipeline.ingest_from_file(path)

    assert report.indexed == 1
    assert report.rejected == 1
    assert "24 resources" not in report.summary()


def test_service_area_defaults_do_not_crash_document_building() -> None:
    resource = Resource.model_validate(VALID_RECORD | {"service_area": ServiceArea().model_dump()})

    assert "service area not specified" in build_embedding_text(resource)
    assert build_metadata(resource)["cities"] == []


def test_dates_survive_the_payload_round_trip() -> None:
    payload = build_payload(_sample_resource())

    assert payload["last_verified"] == "2026-06-01"
    assert Resource.model_validate(payload).last_verified == date(2026, 6, 1)
