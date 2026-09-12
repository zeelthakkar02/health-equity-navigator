"""FastAPI dependencies.

Long-lived objects (settings, the LLM client) are built once during startup and
held on ``app.state``; these helpers hand them to route functions.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.services.llm.base import LLMService
from app.services.navigator_service import NavigatorService
from app.services.retrieval.retriever import ResourceRetriever


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


def get_retriever(request: Request) -> ResourceRetriever | None:
    """The retriever built at startup, or None when retrieval is disabled."""
    return getattr(request.app.state, "retriever", None)


def get_navigator_service(
    llm: Annotated[LLMService, Depends(get_llm_service)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    retriever: Annotated[ResourceRetriever | None, Depends(get_retriever)],
) -> NavigatorService:
    return NavigatorService(llm=llm, settings=settings, retriever=retriever)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
NavigatorServiceDep = Annotated[NavigatorService, Depends(get_navigator_service)]
