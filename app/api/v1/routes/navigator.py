"""Navigator query endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status

from app.api.auth import RateLimitedUser
from app.api.deps import NavigatorServiceDep
from app.schemas.common import ErrorResponse
from app.schemas.navigator import NavigatorQueryRequest, NavigatorQueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/navigator", tags=["navigator"])


@router.post(
    "/query",
    response_model=NavigatorQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the Health Equity Navigator",
    responses={
        401: {"model": ErrorResponse, "description": "Missing, invalid, or expired ID token."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
        502: {"model": ErrorResponse, "description": "The LLM provider failed."},
        503: {"model": ErrorResponse, "description": "The LLM provider is misconfigured."},
        504: {"model": ErrorResponse, "description": "The LLM provider timed out."},
    },
)
async def query_navigator(
    payload: NavigatorQueryRequest,
    navigator: NavigatorServiceDep,
    request: Request,
    user: RateLimitedUser,
) -> NavigatorQueryResponse:
    """Answer a community member's need.

    Requires a verified Identity Platform ID token. The caller's identity comes
    from that token and never from the request body, so ``session_id`` and any
    other client-supplied field cannot be used to act as another member.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        "navigator query accepted for a verified user",
        extra={"request_id": request_id or "-", "uid": user.uid},
    )
    return await navigator.answer(payload, request_id=request_id)
