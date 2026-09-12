"""Request hardening on the public API surface."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.conftest import VALID_TOKEN, FakeTokenVerifier

ENDPOINT = "/api/v1/navigator/query"
PAYLOAD = {"query": "I need help finding food."}
AUTH = {"Authorization": f"Bearer {VALID_TOKEN}"}


@pytest.fixture
def hardened_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "rate_limit_enabled": True,
            "rate_limit_requests": 3,
            "rate_limit_window_seconds": 60.0,
            "max_request_bytes": 512,
            # These tests exercise the address-keyed limiter; keep the per-user
            # one from firing first and masking what is being measured.
            "user_rate_limit_requests": 1000,
        }
    )


@pytest.fixture
def hardened_client(
    hardened_settings: Settings, verifier: FakeTokenVerifier
) -> Iterator[TestClient]:
    """Authenticated: these tests are about the middleware, not about auth."""
    app = create_app(hardened_settings)
    app.state.token_verifier = verifier
    with TestClient(app) as client:
        client.headers.update(AUTH)
        yield client


# --- security headers ------------------------------------------------------


def test_security_headers_are_present(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_answers_are_never_cached(client: TestClient) -> None:
    """An answer is specific to one person's situation."""
    response = client.post(ENDPOINT, json=PAYLOAD)

    assert response.headers["Cache-Control"] == "no-store"


# --- body size -------------------------------------------------------------


def test_an_oversized_body_is_rejected(hardened_client: TestClient) -> None:
    response = hardened_client.post(ENDPOINT, json={"query": "x" * 2000})

    assert response.status_code == 413
    assert response.json()["error"] == "request_too_large"


def test_a_normal_body_is_accepted(hardened_client: TestClient) -> None:
    assert hardened_client.post(ENDPOINT, json=PAYLOAD).status_code == 200


def test_a_malformed_content_length_is_rejected(hardened_client: TestClient) -> None:
    response = hardened_client.post(
        ENDPOINT,
        content=b'{"query": "hello there"}',
        headers={"Content-Type": "application/json", "Content-Length": "not-a-number"},
    )

    assert response.status_code == 400


# --- rate limiting ---------------------------------------------------------


def test_requests_beyond_the_limit_are_rejected(hardened_client: TestClient) -> None:
    for _ in range(3):
        assert hardened_client.post(ENDPOINT, json=PAYLOAD).status_code == 200

    response = hardened_client.post(ENDPOINT, json=PAYLOAD)

    assert response.status_code == 429
    assert response.json()["error"] == "rate_limited"
    assert int(response.headers["Retry-After"]) >= 1


def test_health_is_exempt_from_the_rate_limit(hardened_client: TestClient) -> None:
    """A liveness probe must never be throttled into reporting a false outage."""
    for _ in range(10):
        assert hardened_client.get("/health").status_code == 200


def test_the_rate_limiter_can_be_turned_off(
    settings: Settings, verifier: FakeTokenVerifier
) -> None:
    relaxed = settings.model_copy(
        update={
            "rate_limit_enabled": False,
            "rate_limit_requests": 1,
            # The per-user limiter is a separate control; keep it out of the way.
            "user_rate_limit_requests": 100,
        }
    )
    app = create_app(relaxed)
    app.state.token_verifier = verifier

    with TestClient(app) as client:
        client.headers.update(AUTH)
        for _ in range(5):
            assert client.post(ENDPOINT, json=PAYLOAD).status_code == 200


# --- request id ------------------------------------------------------------


def test_a_rejected_request_still_carries_a_request_id(hardened_client: TestClient) -> None:
    response = hardened_client.post(ENDPOINT, json={"query": "x" * 2000})

    assert response.headers["X-Request-ID"]
    assert response.json()["request_id"]


# --- input limits ----------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "a" * 2001},
        {"query": "valid question here", "location": "x" * 200},
        {"query": "valid question here", "session_id": "x" * 100},
        {"query": "valid question here", "categories": ["a"] * 11},
        {"query": "valid question here", "max_resources": 100},
    ],
)
def test_oversized_fields_are_rejected(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post(ENDPOINT, json=payload).status_code == 422
