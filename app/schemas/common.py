"""Schemas shared across endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "service": "Health Equity Navigator",
                "version": "0.1.0",
                "environment": "local",
                "llm_provider": "stub",
            }
        }
    )

    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str
    llm_provider: str = Field(description="Configured runtime AI backend: 'stub' or 'vertex'.")
    embedding_provider: str = Field(
        default="", description="Configured embedding backend: 'hashing' or 'vertex'."
    )
    resources_indexed: int = Field(
        default=0, description="Verified resources currently in the retrieval index."
    )


class ErrorResponse(BaseModel):
    """Uniform error envelope, so a client can handle failures predictably."""

    error: str = Field(description="Stable machine-readable error code.")
    detail: str = Field(description="Human-readable explanation, safe to log.")
    request_id: str | None = None
