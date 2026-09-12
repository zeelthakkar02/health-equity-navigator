"""Shared test fixtures.

Settings are always built explicitly with ``_env_file=None`` so a developer's
local .env can never change what the suite asserts.

Authentication is exercised for real: the app runs with ``auth_enabled=True`` and
a fake *verifier* is injected, so header parsing, error mapping, and the 401
paths are all real code. Only the cryptographic check against Google's keys is
stubbed — the one part that would need a network call and a live Identity
Platform project. No token here is a real credential.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.auth import TokenVerificationError, TokenVerifierUnavailableError
from app.core.config import Environment, LLMProvider, Settings
from app.main import create_app

# Synthetic strings, not credentials. Nothing here authenticates anywhere.
VALID_TOKEN = "synthetic-valid-id-token"  # noqa: S105
EXPIRED_TOKEN = "synthetic-expired-id-token"  # noqa: S105
UNAVAILABLE_TOKEN = "synthetic-keyserver-down-token"  # noqa: S105

TEST_UID = "member-uid-0001"
TEST_EMAIL = "member@example.test"


class FakeTokenVerifier:
    """Stands in for Firebase's verifier without network or credentials."""

    def __init__(self) -> None:
        self.verified_tokens: list[str] = []

    def verify(self, token: str) -> dict[str, Any]:
        self.verified_tokens.append(token)
        if token == VALID_TOKEN:
            return {"uid": TEST_UID, "email": TEST_EMAIL, "email_verified": True}
        if token == EXPIRED_TOKEN:
            raise TokenVerificationError("ExpiredIdTokenError")
        if token == UNAVAILABLE_TOKEN:
            raise TokenVerifierUnavailableError("CertificateFetchError")
        raise TokenVerificationError("InvalidIdTokenError")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment=Environment.LOCAL,
        llm_provider=LLMProvider.STUB,
        cors_allow_origins=["http://localhost:3000"],
        # Tests must never read or write the developer's on-disk index cache.
        resource_index_cache_enabled=False,
        auth_enabled=True,
        identity_platform_project_id="example-gcp-project",
    )


@pytest.fixture
def verifier() -> FakeTokenVerifier:
    return FakeTokenVerifier()


@pytest.fixture
def anonymous_client(settings: Settings, verifier: FakeTokenVerifier) -> Iterator[TestClient]:
    """A client that sends no Authorization header."""
    app = create_app(settings)
    app.state.token_verifier = verifier
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client(anonymous_client: TestClient) -> TestClient:
    """The default client: authenticated with a valid synthetic token."""
    anonymous_client.headers.update({"Authorization": f"Bearer {VALID_TOKEN}"})
    return anonymous_client
