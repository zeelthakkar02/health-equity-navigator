"""Identity Platform bearer-token authentication on the query endpoint.

No real token, credential, or PHI appears here. The fake verifier stands in for
the cryptographic check against Google's keys; every other step — header
parsing, error mapping, identity extraction, rate limiting — is the real code.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.auth import (
    AuthenticatedUser,
    FirebaseTokenVerifier,
    build_token_verifier,
    extract_bearer_token,
)
from app.core.config import Settings
from app.main import create_app
from tests.conftest import (
    EXPIRED_TOKEN,
    TEST_EMAIL,
    TEST_UID,
    UNAVAILABLE_TOKEN,
    VALID_TOKEN,
    FakeTokenVerifier,
)

ENDPOINT = "/api/v1/navigator/query"
PAYLOAD = {"query": "I need help finding a food pantry."}


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --- /health stays open ----------------------------------------------------


def test_health_works_without_authentication(anonymous_client: TestClient) -> None:
    """Cloud Run's probe has no token; a 401 here would look like an outage."""
    response = anonymous_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_ignores_a_bogus_token(anonymous_client: TestClient) -> None:
    response = anonymous_client.get("/health", headers=_auth("nonsense"))
    assert response.status_code == 200


# --- rejection paths -------------------------------------------------------


def test_query_without_a_token_is_rejected(anonymous_client: TestClient) -> None:
    response = anonymous_client.post(ENDPOINT, json=PAYLOAD)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_query_with_a_malformed_token_is_rejected(anonymous_client: TestClient) -> None:
    response = anonymous_client.post(ENDPOINT, json=PAYLOAD, headers=_auth("not-a-jwt"))
    assert response.status_code == 401


def test_query_with_an_expired_token_is_rejected(anonymous_client: TestClient) -> None:
    response = anonymous_client.post(ENDPOINT, json=PAYLOAD, headers=_auth(EXPIRED_TOKEN))
    assert response.status_code == 401


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Bearer",
        "Bearer ",
        VALID_TOKEN,  # no scheme
        f"Basic {VALID_TOKEN}",
        f"Token {VALID_TOKEN}",
    ],
)
def test_malformed_authorization_headers_are_rejected(
    anonymous_client: TestClient, header: str
) -> None:
    response = anonymous_client.post(ENDPOINT, json=PAYLOAD, headers={"Authorization": header})
    assert response.status_code == 401


def test_rejections_do_not_reveal_which_check_failed(anonymous_client: TestClient) -> None:
    """Distinguishable errors let an attacker probe for valid token shapes."""
    missing = anonymous_client.post(ENDPOINT, json=PAYLOAD).status_code
    malformed = anonymous_client.post(ENDPOINT, json=PAYLOAD, headers=_auth("abc")).status_code
    expired = anonymous_client.post(
        ENDPOINT, json=PAYLOAD, headers=_auth(EXPIRED_TOKEN)
    ).status_code

    assert missing == malformed == expired == 401


def test_a_rejected_request_never_reaches_the_model(
    anonymous_client: TestClient, verifier: FakeTokenVerifier
) -> None:
    """Authentication is also the first line of cost control."""
    response = anonymous_client.post(ENDPOINT, json=PAYLOAD)

    assert response.status_code == 401
    assert response.json().get("resources") is None


# --- the verified path -----------------------------------------------------


def test_a_valid_token_is_accepted(client: TestClient) -> None:
    response = client.post(ENDPOINT, json=PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["request_id"]


def test_the_token_is_actually_verified(client: TestClient, verifier: FakeTokenVerifier) -> None:
    client.post(ENDPOINT, json=PAYLOAD)

    assert verifier.verified_tokens == [VALID_TOKEN]


def test_a_key_server_outage_is_not_the_callers_fault(anonymous_client: TestClient) -> None:
    """Unable to verify is 503, not 401 — the credential may be perfectly good."""
    response = anonymous_client.post(ENDPOINT, json=PAYLOAD, headers=_auth(UNAVAILABLE_TOKEN))

    assert response.status_code == 503


# --- identity comes only from the token ------------------------------------


def test_identity_in_the_request_body_is_ignored(client: TestClient) -> None:
    """A UID in the body must never be able to impersonate another member."""
    response = client.post(
        ENDPOINT,
        json=PAYLOAD | {"uid": "someone-else", "email": "attacker@example.test"},
    )

    assert response.status_code == 200
    assert "someone-else" not in response.text


def test_claims_are_read_from_the_verified_token() -> None:
    verifier = FakeTokenVerifier()
    claims = verifier.verify(VALID_TOKEN)

    user = AuthenticatedUser(
        uid=claims["uid"], email=claims["email"], email_verified=claims["email_verified"]
    )

    assert user.uid == TEST_UID
    assert user.email == TEST_EMAIL
    assert user.rate_limit_key == f"uid:{TEST_UID}"


# --- tokens must not leak into logs ----------------------------------------


def test_a_rejected_token_is_never_logged(
    anonymous_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "synthetic-token-value-that-must-not-appear"  # noqa: S105

    with caplog.at_level("DEBUG"):
        anonymous_client.post(ENDPOINT, json=PAYLOAD, headers=_auth(secret))

    assert secret not in caplog.text


def test_an_accepted_token_is_never_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("DEBUG"):
        client.post(ENDPOINT, json=PAYLOAD)

    assert VALID_TOKEN not in caplog.text


def test_the_query_text_is_not_logged_by_default(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    query = "I need help with my HIV medication costs."

    with caplog.at_level("DEBUG"):
        client.post(ENDPOINT, json={"query": query})

    assert query not in caplog.text


# --- per-user rate limiting ------------------------------------------------


def test_a_single_user_is_rate_limited_by_uid(
    settings: Settings, verifier: FakeTokenVerifier
) -> None:
    limited = settings.model_copy(
        update={"user_rate_limit_requests": 2, "rate_limit_enabled": False}
    )
    app = create_app(limited)
    app.state.token_verifier = verifier

    with TestClient(app) as client:
        client.headers.update(_auth(VALID_TOKEN))
        assert client.post(ENDPOINT, json=PAYLOAD).status_code == 200
        assert client.post(ENDPOINT, json=PAYLOAD).status_code == 200
        blocked = client.post(ENDPOINT, json=PAYLOAD)

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


def test_the_user_limit_does_not_apply_to_health(
    settings: Settings, verifier: FakeTokenVerifier
) -> None:
    limited = settings.model_copy(update={"user_rate_limit_requests": 1})
    app = create_app(limited)
    app.state.token_verifier = verifier

    with TestClient(app) as client:
        for _ in range(5):
            assert client.get("/health").status_code == 200


# --- configuration ---------------------------------------------------------


def test_disabling_auth_builds_no_verifier(settings: Settings) -> None:
    assert build_token_verifier(settings.model_copy(update={"auth_enabled": False})) is None


def test_enabling_auth_builds_a_firebase_verifier(settings: Settings) -> None:
    verifier = build_token_verifier(settings)
    assert isinstance(verifier, FirebaseTokenVerifier)


def test_production_refuses_to_run_unauthenticated(settings: Settings) -> None:
    with pytest.raises(ValueError, match="AUTH_ENABLED must be true"):
        Settings(
            _env_file=None,
            environment="prod",
            auth_enabled=False,
            cors_allow_origins=["https://portal.example"],
        )


def test_production_refuses_wildcard_cors(settings: Settings) -> None:
    with pytest.raises(ValueError, match="explicit origins in production"):
        Settings(
            _env_file=None,
            environment="prod",
            auth_enabled=True,
            google_cloud_project="example-gcp-project",
            cors_allow_origins=["*"],
        )


def test_auth_requires_a_project_to_check_the_audience() -> None:
    with pytest.raises(ValueError, match="IDENTITY_PLATFORM_PROJECT_ID"):
        Settings(_env_file=None, auth_enabled=True, google_cloud_project=None)


def test_the_identity_project_falls_back_to_the_cloud_project() -> None:
    settings = Settings(_env_file=None, auth_enabled=True, google_cloud_project="fallback-project")
    assert settings.resolved_identity_project_id == "fallback-project"


def test_authorization_is_an_allowed_cors_header(client: TestClient) -> None:
    """Without this the browser cannot send the bearer token at all."""
    response = client.options(
        ENDPOINT,
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


# --- the header parser itself ----------------------------------------------


def test_extract_bearer_token_accepts_a_well_formed_header() -> None:
    assert extract_bearer_token(f"Bearer {VALID_TOKEN}") == VALID_TOKEN


def test_extract_bearer_token_is_scheme_case_insensitive() -> None:
    assert extract_bearer_token(f"bearer {VALID_TOKEN}") == VALID_TOKEN


# --- existing behaviour is untouched ---------------------------------------


def test_request_validation_still_applies_after_auth(client: TestClient) -> None:
    assert client.post(ENDPOINT, json={"query": "hi"}).status_code == 422
    assert client.post(ENDPOINT, json={}).status_code == 422


def test_escalation_behaviour_survives_authentication(client: TestClient) -> None:
    """A crisis message must still short-circuit, for an authenticated user too."""
    response = client.post(ENDPOINT, json={"query": "I want to kill myself."})

    body = response.json()
    assert body["answer_source"] == "safety_notice"
    assert body["needs_escalation"] is True
    assert body["escalation_reason"] == "possible_crisis"
    assert "988" in body["answer"]


def test_grounded_answer_shape_survives_authentication(client: TestClient) -> None:
    response = client.post(ENDPOINT, json={"query": "I need food assistance near Oakland."})

    body = response.json()
    assert body["resources"]
    assert body["disclaimer"]
    assert body["retrieval_performed"] is True
    assert not any("score" in key.lower() for key in body)
