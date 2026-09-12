"""FastAPI application entry point."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestTimeoutMiddleware,
    SecurityHeadersMiddleware,
)
from app.api.v1.router import api_router
from app.api.v1.routes import health
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.schemas.common import ErrorResponse
from app.services.llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.services.llm.factory import build_llm_service
from app.services.retrieval.bootstrap import bootstrap_retrieval

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

_ERROR_STATUS: dict[type[LLMError], int] = {
    LLMConfigurationError: 503,
    LLMTimeoutError: 504,
    LLMUpstreamError: 502,
}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Tests call this directly with custom settings."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format.value)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = settings
        application.state.llm_service = build_llm_service(settings)
        application.state.retriever = None
        application.state.retrieval_stack = None
        application.state.resources_indexed = 0

        if settings.retrieval_enabled:
            # Cold start embeds the resource file; the persisted index makes
            # every later start near-instant.
            stack, report = await bootstrap_retrieval(settings)
            application.state.retrieval_stack = stack
            application.state.retriever = stack.retriever
            application.state.resources_indexed = report.indexed
            logger.info("Retrieval ready: %s", report.summary())
        else:
            logger.warning("Retrieval is disabled; the Navigator will not cite resources.")

        logger.info(
            "%s v%s starting (environment=%s, llm_provider=%s, embedding_provider=%s)",
            settings.app_name,
            settings.app_version,
            settings.environment.value,
            settings.llm_provider.value,
            settings.embedding_provider.value,
        )
        try:
            yield
        finally:
            if application.state.retrieval_stack is not None:
                await application.state.retrieval_stack.aclose()
            await application.state.llm_service.aclose()
            logger.info("%s shutting down", settings.app_name)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Phase 1 API for the Health Equity Navigator. Returns grounded answers "
            "about community health resources. RAG retrieval arrives in Phase 2."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    application.state.settings = settings

    # Starlette runs middleware in reverse registration order, so the request-id
    # middleware is added last and therefore runs first: everything below it,
    # including rejections, can report a request id.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", REQUEST_ID_HEADER],
    )
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(
        RequestTimeoutMiddleware, timeout_seconds=settings.request_timeout_seconds
    )
    if settings.rate_limit_enabled:
        application.add_middleware(
            RateLimitMiddleware,
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
            trust_proxy_headers=settings.trust_proxy_headers,
        )
    application.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    @application.middleware("http")
    async def attach_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    @application.exception_handler(LLMError)
    async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
        status_code = _ERROR_STATUS.get(type(exc), 502)
        request_id = getattr(request.state, "request_id", None)
        logger.error("LLM failure (%s): %s", exc.code, exc, extra={"request_id": request_id or "-"})
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=exc.code, detail=str(exc), request_id=request_id
            ).model_dump(),
        )

    application.include_router(health.router)
    application.include_router(api_router, prefix=settings.api_v1_prefix)

    return application


app = create_app()
