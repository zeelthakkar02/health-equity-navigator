"""Navigator query endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from app.api.deps import NavigatorServiceDep
from app.schemas.common import ErrorResponse
from app.schemas.navigator import NavigatorQueryRequest, NavigatorQueryResponse

router = APIRouter(prefix="/navigator", tags=["navigator"])


@router.post(
    "/query",
    response_model=NavigatorQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the Health Equity Navigator",
    responses={
        502: {"model": ErrorResponse, "description": "The LLM provider failed."},
        503: {"model": ErrorResponse, "description": "The LLM provider is misconfigured."},
        504: {"model": ErrorResponse, "description": "The LLM provider timed out."},
    },
)
async def query_navigator(
    payload: NavigatorQueryRequest,
    navigator: NavigatorServiceDep,
    request: Request,
) -> NavigatorQueryResponse:
    """Answer a community member's need.

    Phase 1 returns a grounded answer with an empty ``resources`` list; RAG
    retrieval over verified community resources arrives in Phase 2.
    """
    request_id = getattr(request.state, "request_id", None)
    return await navigator.answer(payload, request_id=request_id)
