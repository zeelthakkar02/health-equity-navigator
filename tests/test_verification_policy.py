"""Serving every verification status, honestly.

The earlier policy withheld anything not fully or partially verified. That hid
the only resources answering whole classes of question — every insulin-assistance
programme in the real directory sits at "needs verification" — so all three
statuses are now searchable and the uncertainty travels with the record instead.

These tests pin both halves of that: the records are reachable, and nothing
about them pretends to be verified.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.core.config import Settings
from app.domain.resource import Resource, ServiceCategory, VerificationStatus
from app.services.embeddings.hashing_provider import HashingEmbeddingService
from app.services.ingestion.documents import build_embedding_text
from app.services.ingestion.pipeline import ResourceIngestionPipeline
from app.services.llm.prompts import SYSTEM_INSTRUCTION, format_resources
from app.services.navigator_service import to_citation
from app.services.retrieval.bootstrap import resolve_servable_statuses
from app.services.retrieval.retriever import DEFAULT_SERVABLE_STATUSES, ResourceRetriever
from app.services.vectorstore.memory_store import InMemoryVectorStore

TODAY = date(2026, 10, 6)


def make_resource(
    resource_id: str,
    name: str,
    status: VerificationStatus,
    *,
    description: str = "Free groceries and emergency food boxes for local families.",
    notes: str | None = None,
) -> Resource:
    return Resource.model_validate(
        {
            "resource_id": resource_id,
            "organization_name": name,
            "description": description,
            "service_categories": [ServiceCategory.FOOD_ASSISTANCE.value],
            "source": "test-fixture",
            "verification_status": status.value,
            "confirmation_notes": notes,
        }
    )


async def retriever_over(resources: list[Resource], **kwargs: object) -> ResourceRetriever:
    embeddings = HashingEmbeddingService()
    store = InMemoryVectorStore()
    await ResourceIngestionPipeline(embeddings, store).ingest(resources)
    return ResourceRetriever(embeddings, store, min_score=0.0, **kwargs)  # type: ignore[arg-type]


# --- all three statuses are reachable ------------------------------------


def test_the_three_real_statuses_are_servable_by_default() -> None:
    assert DEFAULT_SERVABLE_STATUSES == {
        VerificationStatus.VERIFIED,
        VerificationStatus.PARTIALLY_VERIFIED,
        VerificationStatus.NEEDS_VERIFICATION,
    }


def test_an_unreadable_status_is_still_withheld() -> None:
    """UNVERIFIED means "we could not read the field", which is not the same
    thing as a directory telling us a record needs checking."""
    assert VerificationStatus.UNVERIFIED not in DEFAULT_SERVABLE_STATUSES
    assert VerificationStatus.NEEDS_REVIEW not in DEFAULT_SERVABLE_STATUSES


async def test_a_needs_verification_resource_is_returned() -> None:
    retriever = await retriever_over(
        [make_resource("nv-1", "Example Food Programme", VerificationStatus.NEEDS_VERIFICATION)]
    )

    results = await retriever.retrieve("free groceries food", as_of=TODAY)

    assert [r.resource.resource_id for r in results] == ["nv-1"]


async def test_a_needs_verification_resource_can_rank_first_when_most_relevant() -> None:
    """The whole point: being unverified must not bury a better match."""
    relevant = make_resource(
        "nv-1",
        "Example Insulin Cost Programme",
        VerificationStatus.NEEDS_VERIFICATION,
        description="Help paying for insulin and diabetes prescriptions and medication costs.",
    )
    verified_but_unrelated = make_resource(
        "v-1",
        "Example Housing Office",
        VerificationStatus.VERIFIED,
        description="Emergency rental assistance and eviction defence for tenants.",
    )
    retriever = await retriever_over([relevant, verified_but_unrelated])

    results = await retriever.retrieve("I cannot afford my insulin", as_of=TODAY)

    assert results
    assert results[0].resource.resource_id == "nv-1"


# --- verification breaks ties, it does not decide relevance --------------


async def test_verification_breaks_a_tie_between_equal_matches() -> None:
    identical = "Free groceries and emergency food boxes for local families."
    retriever = await retriever_over(
        [
            make_resource(
                "nv",
                "Example Pantry North",
                VerificationStatus.NEEDS_VERIFICATION,
                description=identical,
            ),
            make_resource(
                "pv",
                "Example Pantry South",
                VerificationStatus.PARTIALLY_VERIFIED,
                description=identical,
            ),
            make_resource(
                "v", "Example Pantry East", VerificationStatus.VERIFIED, description=identical
            ),
        ]
    )

    results = await retriever.retrieve("free groceries food boxes", top_k=3, as_of=TODAY)

    assert [r.resource.resource_id for r in results] == ["v", "pv", "nv"]


async def test_the_tie_breaker_is_small_enough_to_lose_to_relevance() -> None:
    """A verified near-miss must not outrank an unverified direct hit."""
    retriever = await retriever_over(
        [
            make_resource(
                "nv",
                "Example Diabetes Supply Help",
                VerificationStatus.NEEDS_VERIFICATION,
                description="Insulin and diabetes test strip cost assistance for uninsured adults.",
            ),
            make_resource(
                "v",
                "Example General Clinic",
                VerificationStatus.VERIFIED,
                description="General primary care clinic offering checkups and vaccinations.",
            ),
        ]
    )

    results = await retriever.retrieve("insulin diabetes cost assistance", as_of=TODAY)

    assert results[0].resource.resource_id == "nv"


async def test_the_boost_can_be_turned_off() -> None:
    """With every boost at zero the score is the similarity and nothing else."""
    retriever = await retriever_over(
        [make_resource("v", "Example Pantry", VerificationStatus.VERIFIED)],
        verified_boost=0.0,
        partially_verified_boost=0.0,
        category_boost=0.0,
    )

    results = await retriever.retrieve("free groceries food", as_of=TODAY)

    assert results[0].score == pytest.approx(results[0].semantic_score)


async def test_the_verification_boost_is_recorded_separately_from_relevance() -> None:
    retriever = await retriever_over(
        [make_resource("v", "Example Pantry", VerificationStatus.VERIFIED)],
        category_boost=0.0,
        verified_boost=0.06,
    )

    results = await retriever.retrieve("free groceries food", as_of=TODAY)

    assert results[0].score == pytest.approx(results[0].semantic_score + 0.06, abs=1e-6)


# --- nothing claims verification it does not have ------------------------


def test_status_reaches_the_embedding_text() -> None:
    text = build_embedding_text(
        make_resource("nv", "Example Pantry", VerificationStatus.NEEDS_VERIFICATION)
    )

    assert "Verification status: needs verification" in text


def test_status_reaches_the_prompt_listing() -> None:
    citation = to_citation(
        make_resource(
            "nv",
            "Example Pantry",
            VerificationStatus.NEEDS_VERIFICATION,
            notes="Hours and eligibility not confirmed",
        )
    )

    block = format_resources([citation])

    assert "verification status: needs_verification" in block
    assert "NOT YET CONFIRMED: Hours and eligibility not confirmed" in block


def test_the_prompt_forbids_calling_an_unverified_resource_verified() -> None:
    lowered = SYSTEM_INSTRUCTION.lower()

    assert "never describe a resource as" in lowered
    assert "verified" in lowered
    assert "have not been independently verified" in lowered


def test_the_citation_carries_status_and_caveat_to_the_client() -> None:
    citation = to_citation(
        make_resource(
            "pv", "Example Pantry", VerificationStatus.PARTIALLY_VERIFIED, notes="Confirm hours"
        )
    )

    assert citation.verification_status == "partially_verified"
    assert citation.confirmation_notes == "Confirm hours"


# --- configuration -------------------------------------------------------


def test_the_servable_set_is_configurable(settings: Settings) -> None:
    strict = settings.model_copy(update={"retrieval_servable_statuses": ["verified"]})

    assert resolve_servable_statuses(strict) == {VerificationStatus.VERIFIED}


def test_an_unknown_status_name_in_config_is_ignored(settings: Settings) -> None:
    messy = settings.model_copy(
        update={"retrieval_servable_statuses": ["verified", "not_a_status"]}
    )

    assert resolve_servable_statuses(messy) == {VerificationStatus.VERIFIED}


def test_servable_statuses_parse_from_a_comma_separated_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_SERVABLE_STATUSES", "verified,partially_verified")

    assert Settings(_env_file=None).retrieval_servable_statuses == [
        "verified",
        "partially_verified",
    ]


# --- production must not serve synthetic data ----------------------------


def _prod(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "_env_file": None,
        "environment": "prod",
        "auth_enabled": True,
        "google_cloud_project": "example-gcp-project",
        "cors_allow_origins": ["https://portal.example"],
    }
    return base | overrides


def test_production_refuses_the_bundled_synthetic_dataset() -> None:
    """A service quietly answering from invented organizations looks healthy."""
    with pytest.raises(ValueError, match="synthetic sample"):
        Settings(**_prod())  # type: ignore[arg-type]


def test_production_refuses_a_missing_dataset() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        Settings(**_prod(resource_data_path="app/data/private/absent.json"))  # type: ignore[arg-type]


def test_production_accepts_a_real_dataset_that_exists(tmp_path) -> None:
    real = tmp_path / "resources.json"
    real.write_text('{"resources": []}', encoding="utf-8")

    settings = Settings(**_prod(resource_data_path=str(real)))  # type: ignore[arg-type]

    assert settings.resource_data_path == real


def test_the_guard_does_not_fire_outside_production(settings: Settings) -> None:
    assert settings.resource_data_path.name == "sample_resources.json"


def test_the_guard_is_skipped_when_retrieval_is_disabled() -> None:
    settings = Settings(**_prod(retrieval_enabled=False))  # type: ignore[arg-type]
    assert settings.retrieval_enabled is False
