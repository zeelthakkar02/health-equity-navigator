"""Request hardening for a publicly reachable API.

Deliberately small and dependency-free. Each piece closes a specific way an
unauthenticated endpoint backed by a paid model can be abused or fall over:

* a body-size cap, so a huge payload is rejected before it is parsed
* a per-client rate limit, so one caller cannot drain the Vertex budget
* a request deadline, so a hung upstream cannot pin a worker forever
* conservative response headers, since answers are per-person and must not be
  cached by a proxy or embedded in someone else's page

The rate limiter is per process and in memory. That is honest for a single
instance and explicitly not enough behind a load balancer — several replicas
each allow the full quota. A shared limiter belongs at the gateway.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)

CallNext = Callable[[Request], Awaitable]

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # Answers are specific to one person's situation; never let them be cached.
    "Cache-Control": "no-store",
}


def _error(status_code: int, code: str, detail: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=code,
            detail=detail,
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds conservative headers to every response."""

    async def dispatch(self, request: Request, call_next: CallNext):
        response = await call_next(request)
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects oversized payloads before they are read or parsed."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: CallNext):
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self._max_bytes:
                    return self._too_large(request)
            except ValueError:
                return _error(400, "invalid_content_length", "Malformed Content-Length.", request)
        return await call_next(request)

    def _too_large(self, request: Request) -> JSONResponse:
        logger.warning("rejected oversized request body (limit %d bytes)", self._max_bytes)
        return _error(
            413,
            "request_too_large",
            f"Request body exceeds the {self._max_bytes} byte limit.",
            request,
        )


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Bounds total handling time so a hung upstream cannot pin a worker."""

    def __init__(self, app: ASGIApp, *, timeout_seconds: float) -> None:
        super().__init__(app)
        self._timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: CallNext):
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await call_next(request)
        except TimeoutError:
            logger.error("request exceeded %.1fs deadline", self._timeout_seconds)
            return _error(
                504,
                "request_timeout",
                f"The request took longer than {self._timeout_seconds:g}s.",
                request,
            )


class FixedWindowLimiter:
    """Counts requests per key inside a sliding window, in process memory.

    Shared by the IP-keyed middleware and the UID-keyed dependency so the two
    cannot drift apart. Per process only: several Cloud Run instances each allow
    the full quota, which is why this is a brake on one abusive caller rather
    than a quota system.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> int | None:
        """Record a hit. Returns retry-after seconds when over the limit."""
        now = time.monotonic()
        window = self._hits.setdefault(key, deque())

        while window and now - window[0] > self._window_seconds:
            window.popleft()

        if len(window) >= self._max_requests:
            return max(1, int(self._window_seconds - (now - window[0])))

        window.append(now)
        self._prune(now)
        return None

    def _prune(self, now: float) -> None:
        """Drop keys whose windows have fully expired, bounding memory."""
        if len(self._hits) < 1024:
            return
        stale = [
            key
            for key, window in self._hits.items()
            if not window or now - window[-1] > self._window_seconds
        ]
        for key in stale:
            del self._hits[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window-per-client limiter, kept in memory.

    Per process only: several replicas each allow the full quota. Enough to stop
    a single abusive client on one instance, not a substitute for a gateway.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_requests: int,
        window_seconds: float,
        trust_proxy_headers: bool = False,
        exempt_paths: frozenset[str] = frozenset({"/health"}),
    ) -> None:
        super().__init__(app)
        self._limiter = FixedWindowLimiter(max_requests, window_seconds)
        self._trust_proxy_headers = trust_proxy_headers
        self._exempt_paths = exempt_paths

    async def dispatch(self, request: Request, call_next: CallNext):
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        retry_after = self._limiter.check(self._client_key(request))
        if retry_after is not None:
            logger.warning("rate limit reached for a client address")
            response = _error(
                429,
                "rate_limited",
                f"Too many requests. Try again in {retry_after}s.",
                request,
            )
            response.headers["Retry-After"] = str(retry_after)
            return response

        return await call_next(request)

    def _client_key(self, request: Request) -> str:
        if self._trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
