"""Aggregate router for the versioned API surface."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import navigator

api_router = APIRouter()
api_router.include_router(navigator.router)
