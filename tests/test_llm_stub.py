"""The offline stub provider."""

from __future__ import annotations

from app.schemas.navigator import ResourceCitation
from app.services.llm.base import LLMRequest
from app.services.llm.prompts import build_navigator_prompt, format_resources
from app.services.llm.stub_provider import StubLLMService


def _request(prompt: str) -> LLMRequest:
    return LLMRequest(prompt=prompt, system_instruction="system", max_output_tokens=256)


async def test_stub_generates_a_response() -> None:
    prompt = build_navigator_prompt(query="Where can I get a flu shot?", resources=[])

    response = await StubLLMService().generate(_request(prompt))

    assert response.provider == "stub"
    assert response.model == "stub-navigator-v1"
    assert response.finish_reason == "stop"
    assert "Where can I get a flu shot?" in response.text
    assert response.usage["prompt_tokens"] > 0


async def test_stub_is_deterministic() -> None:
    prompt = build_navigator_prompt(query="I need help paying my heating bill.", resources=[])
    service = StubLLMService()

    first = await service.generate(_request(prompt))
    second = await service.generate(_request(prompt))

    assert first.text == second.text


async def test_stub_notes_when_resources_are_available() -> None:
    prompt = build_navigator_prompt(
        query="I need a food pantry.",
        resources=[ResourceCitation(resource_id="r1", title="Eastside Food Pantry")],
    )

    response = await StubLLMService().generate(_request(prompt))

    assert "verified community resources" in response.text
    assert "211" not in response.text


def test_empty_resource_block_is_explicit() -> None:
    assert "none available" in format_resources([])


def test_resource_block_renders_details() -> None:
    block = format_resources(
        [
            ResourceCitation(
                resource_id="r1",
                title="Eastside Clinic",
                categories=["community_clinic"],
                service_area="Oakland, CA",
                phone="555-0100",
                url="https://example.org",
            )
        ]
    )
    assert "[1] Eastside Clinic" in block
    assert "helps with: community_clinic" in block
    assert "serves: Oakland, CA" in block
    assert "phone: 555-0100" in block


def test_resource_block_never_renders_a_score() -> None:
    """Scores stay internal, including inside the prompt."""
    block = format_resources(
        [ResourceCitation(resource_id="r1", title="Eastside Clinic", categories=["x"])]
    )

    assert "score" not in block.lower()
