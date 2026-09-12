"""Identity Platform / Firebase ID token verification.

The Navigator answers questions about someone's health, housing, and immigration
situation, and every answer costs a Vertex AI call. Both facts argue for knowing
who is asking, so ``/api/v1/navigator/query`` requires a verified bearer token.

Two rules shape this module:

* **Identity comes only from the verified token.** The UID and email are read
  from the token's claims after signature, audience, issuer, and expiry checks
  pass. A UID supplied in the request body is ignored — trusting one would let
  any caller impersonate any member.
* **Tokens never reach the logs.** Failures are logged by category and request
  id. A bearer token is a credential for as long as it lives, and a log line is
  a much longer-lived thing than an access token.

``/health`` deliberately stays unauthenticated so Cloud Run's probe can reach it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Annotated, Any, Protocol

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings

logger = logging.getLogger(__name__)

_BEARER_PREFIX = "bearer "


@dataclass(frozen=True)
class AuthenticatedUser:
    """Identity taken from a verified ID token, and from nowhere else."""

    uid: str
    email: str | None = None
    email_verified: bool = False

    @property
    def rate_limit_key(self) -> str:
        return f"uid:{self.uid}"


class TokenVerificationError(Exception):
    """The token was present but did not verify. Always answered with a 401."""


class TokenVerifierUnavailableError(Exception):
    """Verification could not be attempted — not the caller's fault."""


class TokenVerifier(Protocol):
    """Verifies an ID token and returns its claims."""

    def verify(self, token: str) -> dict[str, Any]: ...


class FirebaseTokenVerifier:
    """Verifies Identity Platform ID tokens with the Firebase Admin SDK.

    The SDK is used rather than a hand-rolled JWT check because it caches and
    rotates Google's signing keys. Verifying against certificates fetched on
    every request would add a network round trip to each call and fail closed
    whenever that fetch hiccups.

    The SDK is imported lazily so importing this module — which the app does at
    startup — pulls in no cloud SDK, matching how the Vertex providers behave.
    """

    def __init__(self, project_id: str) -> None:
        if not project_id:
            raise ValueError("project_id is required to verify ID tokens")
        self._project_id = project_id
        self._app: Any = None
        self._auth: Any = None

    def _ensure_initialised(self) -> None:
        if self._auth is not None:
            return
        try:
            import firebase_admin
            from firebase_admin import auth as firebase_auth
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise TokenVerifierUnavailableError(
                "The 'firebase-admin' package is required to verify ID tokens."
            ) from exc

        name = f"navigator-auth-{self._project_id}"
        try:
            self._app = firebase_admin.get_app(name)
        except ValueError:
            # No credential is passed: verifying a token needs the project id as
            # the expected audience and Google's public keys, not our identity.
            self._app = firebase_admin.initialize_app(
                options={"projectId": self._project_id}, name=name
            )
        self._auth = firebase_auth

    def verify(self, token: str) -> dict[str, Any]:
        self._ensure_initialised()
        assert self._auth is not None  # noqa: S101 - narrowed by _ensure_initialised

        try:
            return dict(self._auth.verify_id_token(token, app=self._app))
        except Exception as exc:
            name = type(exc).__name__
            # A certificate fetch failure is our outage, not a bad credential.
            if "CertificateFetch" in name:
                raise TokenVerifierUnavailableError(name) from exc
            raise TokenVerificationError(name) from exc


def build_token_verifier(settings: Settings) -> TokenVerifier | None:
    """Construct the verifier, or None when authentication is disabled."""
    if not settings.auth_enabled:
        logger.warning(
            "AUTH_ENABLED is false — /api/v1/navigator/query is UNAUTHENTICATED. "
            "This is for local development only."
        )
        return None
    project_id = settings.resolved_identity_project_id
    if not project_id:  # pragma: no cover - Settings validates this
        raise ValueError("No Identity Platform project configured")
    return FirebaseTokenVerifier(project_id)


def _unauthorized(detail: str) -> HTTPException:
    """One shape for every rejection, so probing cannot distinguish causes."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def extract_bearer_token(header_value: str | None) -> str:
    """Pull the credential out of an Authorization header.

    Raises:
        HTTPException: 401, with no echo of the header's contents.
    """
    if not header_value:
        raise _unauthorized("Missing Authorization header.")
    if not header_value.lower().startswith(_BEARER_PREFIX):
        raise _unauthorized("Authorization header must use the Bearer scheme.")
    token = header_value[len(_BEARER_PREFIX) :].strip()
    if not token:
        raise _unauthorized("Bearer token is empty.")
    return token


async def require_authenticated_user(request: Request) -> AuthenticatedUser:
    """FastAPI dependency: 401 unless the request carries a valid ID token."""
    verifier: TokenVerifier | None = getattr(request.app.state, "token_verifier", None)
    request_id = getattr(request.state, "request_id", "-")

    if verifier is None:
        # Authentication is switched off; the startup log already warned loudly.
        return AuthenticatedUser(uid="anonymous-development-user")

    token = extract_bearer_token(request.headers.get("Authorization"))

    try:
        # verify_id_token does blocking IO on its first call per key rotation.
        claims = await asyncio.to_thread(verifier.verify, token)
    except TokenVerificationError as exc:
        logger.warning("ID token rejected (%s)", exc, extra={"request_id": request_id})
        raise _unauthorized("Invalid or expired ID token.") from exc
    except TokenVerifierUnavailableError as exc:
        logger.error(
            "ID token verification unavailable (%s)", exc, extra={"request_id": request_id}
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is temporarily unavailable.",
        ) from exc

    uid = claims.get("uid") or claims.get("user_id") or claims.get("sub")
    if not uid:
        logger.warning("Verified token carried no subject", extra={"request_id": request_id})
        raise _unauthorized("Token is missing a subject claim.")

    return AuthenticatedUser(
        uid=str(uid),
        email=claims.get("email"),
        email_verified=bool(claims.get("email_verified", False)),
    )


CurrentUser = Annotated[AuthenticatedUser, Depends(require_authenticated_user)]


async def enforce_user_rate_limit(request: Request, user: CurrentUser) -> AuthenticatedUser:
    """Rate limit by verified UID, on top of the address-keyed middleware.

    Keying on the verified subject is the part an attacker cannot rotate: an
    address limiter cannot separate two members behind one NAT, and cannot stop
    one account cycling through addresses.
    """
    limiter = getattr(request.app.state, "user_rate_limiter", None)
    if limiter is None:
        return user

    retry_after = limiter.check(user.rate_limit_key)
    if retry_after is not None:
        logger.warning(
            "per-user rate limit reached",
            extra={"request_id": getattr(request.state, "request_id", "-")},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
    return user


RateLimitedUser = Annotated[AuthenticatedUser, Depends(enforce_user_rate_limit)]
