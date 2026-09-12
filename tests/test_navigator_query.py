"""POST /api/v1/navigator/query."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ENDPOINT = "/api/v1/navigator/query"


def test_query_returns_a_grounded_answer(client: TestClient) -> None:
    response = client.post(
        ENDPOINT,
        json={"query": "I need a clinic that takes uninsured patients.", "location": "43210"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].strip()
    assert body["provider"] == "stub"
    assert body["model"] == "stub-navigator-v1"
    assert body["request_id"]
    assert body["disclaimer"].startswith("This is general information")
    assert body["latency_ms"] >= 0


def test_retrieval_runs_and_cites_verified_resources(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={"query": "Where can I get free dental care?"})

    body = response.json()
    assert body["retrieval_performed"] is True
    assert body["resources"]
    citation = body["resources"][0]
    assert citation["resource_id"].startswith("syn-")
    assert citation["title"]
    assert citation["last_verified"]


def test_answer_falls_back_to_211_without_verified_resources(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={"query": "I cannot afford my insulin."})
    assert "211" in response.json()["answer"]


def test_request_id_header_is_propagated_into_the_body(client: TestClient) -> None:
    response = client.post(
        ENDPOINT,
        json={"query": "Help finding a pediatrician."},
        headers={"X-Request-ID": "client-req-42"},
    )
    assert response.json()["request_id"] == "client-req-42"
    assert response.headers["X-Request-ID"] == "client-req-42"


def test_query_is_echoed_back_in_the_answer(client: TestClient) -> None:
    need = "I need transportation to a dialysis appointment."
    response = client.post(ENDPOINT, json={"query": need})
    assert need in response.json()["answer"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "hi"},
        {"query": "a" * 2001},
        {"query": "valid question here", "max_resources": 0},
        {"query": "valid question here", "max_resources": 99},
        {"query": "valid question here", "language": "e"},
    ],
)
def test_invalid_payloads_are_rejected(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post(ENDPOINT, json=payload).status_code == 422


def test_whitespace_is_stripped_from_the_query(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={"query": "   Where is the nearest food pantry?   "})
    assert response.status_code == 200
    assert "   " not in response.json()["answer"]


def test_no_relevance_score_is_exposed_through_the_api(client: TestClient) -> None:
    """Ranking scores are internal: not comparable across providers, and a number
    beside a community organization reads as a judgement about it."""
    schema = client.get("/openapi.json").json()
    citation = schema["components"]["schemas"]["ResourceCitation"]["properties"]

    assert "relevance_score" not in citation
    assert not any("score" in name.lower() for name in citation)


def test_the_response_body_carries_no_score_fields(client: TestClient) -> None:
    body = client.post(ENDPOINT, json={"query": "I need a ride to my appointment."}).json()

    assert not any("score" in key.lower() for key in body)
