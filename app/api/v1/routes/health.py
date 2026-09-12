"""Liveness endpoint.

Deliberately dependency-free: it reports configuration only and never calls the
LLM provider, so a Vertex outage can never make the container look unhealthy.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import SettingsDep
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service liveness check")
async def health(settings: SettingsDep, request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment.value,
        llm_provider=settings.llm_provider.value,
        embedding_provider=settings.embedding_provider.value,
        resources_indexed=getattr(request.app.state, "resources_indexed", 0),
    )
