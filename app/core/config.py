"""Application configuration.

Every value is environment-driven (12-factor). Nothing secret is ever stored in
code: Vertex AI access relies on Application Default Credentials, so there are
no API keys or service-account paths in this file.
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LLMProvider(StrEnum):
    """Runtime AI backends.

    ``STUB`` keeps the whole service usable offline with no cloud credentials.
    ``VERTEX`` is Google Gemini served through Vertex AI.
    """

    STUB = "stub"
    VERTEX = "vertex"


class LogFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


class EmbeddingProvider(StrEnum):
    """Embedding backends for retrieval.

    ``HASHING`` produces deterministic lexical vectors with no network call, so
    ingestion and retrieval are fully testable offline. It matches on shared
    wording, not meaning — real semantic matching needs ``VERTEX``.
    """

    HASHING = "hashing"
    VERTEX = "vertex"


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SAMPLE_DATA_PATH = _PACKAGE_ROOT / "data" / "sample_resources.json"
# Where an imported real dataset is expected to live. Gitignored, but included
# in the Cloud Build upload so it reaches the image.
PRIVATE_DATA_PATH = _PACKAGE_ROOT / "data" / "private" / "cents_resources.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "Health Equity Navigator"
    app_version: str = "0.1.0"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.TEXT
    # Off by default: a community member's message can describe their health,
    # housing, or immigration situation and does not belong in logs.
    log_query_text: bool = False
    api_v1_prefix: str = "/api/v1"

    # Origins allowed to call the API. The consuming website's origin is
    # added here when the two systems are wired together.
    # NoDecode is required: without it pydantic-settings JSON-decodes a list
    # field read from the environment *before* any validator runs, so a plain
    # comma-separated CORS_ALLOW_ORIGINS raises a JSONDecodeError at startup.
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.STUB
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Gemini 3.x spends part of this budget on reasoning before it writes, so a
    # 1024 cap truncated real Navigator answers mid-sentence.
    llm_max_output_tokens: int = Field(default=3072, gt=0, le=32768)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Google Cloud / Vertex AI (used only when llm_provider is VERTEX) ---
    google_cloud_project: str | None = None
    google_cloud_location: str = "global"
    gemini_model: str = "gemini-3.8-flash"
    # Unset leaves the model's own default. Lower levels trade reasoning depth
    # for latency and cost; measure with scripts/benchmark_navigator.py before
    # changing it.
    gemini_thinking_level: str | None = None

    # --- Embeddings (retrieval side; independent of the generation provider) ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HASHING
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(default=768, gt=0, le=3072)
    # gemini-embedding-001 accepts a single input per request on Vertex AI;
    # other embedding models allow larger batches.
    embedding_batch_size: int = Field(default=1, gt=0, le=250)
    embedding_max_concurrency: int = Field(default=4, gt=0, le=32)

    # --- Retrieval ---
    resource_data_path: Path = _DEFAULT_SAMPLE_DATA_PATH
    eval_queries_path: Path = _PACKAGE_ROOT / "data" / "eval_queries.json"
    retrieval_top_k: int = Field(default=5, gt=0, le=50)
    # Gate on semantic similarity alone, before any boost is applied. Score
    # scales differ by an order of magnitude between providers, so leaving this
    # unset selects the value tuned for the configured provider.
    retrieval_min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_location_boost: float = Field(default=0.25, ge=0.0, le=1.0)
    # Weight for a category the person explicitly named in their query.
    retrieval_category_boost: float = Field(default=0.12, ge=0.0, le=1.0)
    retrieval_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    # Records verified longer ago than this are withheld from results.
    retrieval_max_resource_age_days: int = Field(default=548, gt=0)
    retrieval_require_verified: bool = True
    # Statuses a member may be shown. All three statuses a real directory uses
    # are servable: withholding "needs verification" entirely hid the only
    # resources that answered whole classes of question. A status we could not
    # read stays out — unreadable is not the same as known-unverified.
    retrieval_servable_statuses: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["verified", "partially_verified", "needs_verification"]
    )
    # Verification breaks ties; it does not decide relevance. Kept small on
    # purpose so a clearly better "needs verification" match still wins.
    retrieval_verified_boost: float = Field(default=0.06, ge=0.0, le=0.5)
    retrieval_partially_verified_boost: float = Field(default=0.03, ge=0.0, le=0.5)
    # Turning this off makes the Navigator answer without retrieval, which is
    # only useful for isolating the generation path in development.
    retrieval_enabled: bool = True

    # --- Index persistence ---
    # A derived cache of resource embeddings, rebuilt whenever the provider,
    # model, dimensions, or resource data change.
    resource_index_cache_enabled: bool = True
    resource_index_cache_path: Path = Path(".cache/resource_index.npz")

    # --- Authentication ---
    # Identity Platform / Firebase ID tokens are verified server-side.
    #
    # Default off so a fresh clone runs with zero configuration, which the whole
    # repo depends on. That is only safe because ENVIRONMENT=prod refuses to
    # start with auth disabled (see _reject_unauthenticated_production), so the
    # insecure setting cannot survive a deployment.
    auth_enabled: bool = False
    # Audience the ID token must carry. Defaults to GOOGLE_CLOUD_PROJECT when
    # unset, which is the usual case: Identity Platform lives in the same project.
    identity_platform_project_id: str | None = None

    # --- Request hardening (public API surface) ---
    max_request_bytes: int = Field(default=16_384, gt=0, le=1_048_576)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=30, gt=0)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0)
    # Trust a proxy's forwarded client IP. Only turn this on behind a proxy that
    # actually sets it, or callers can spoof their way around the rate limit.
    trust_proxy_headers: bool = False
    # Second limiter, keyed by the verified UID. The IP limiter cannot tell two
    # users behind one NAT apart, and cannot stop one account cycling addresses.
    user_rate_limit_requests: int = Field(default=20, gt=0)
    user_rate_limit_window_seconds: float = Field(default=60.0, gt=0)

    # --- Evaluation ---
    eval_navigator_path: Path = _PACKAGE_ROOT / "data" / "eval_navigator.json"
    eval_decoy_resources_path: Path = _PACKAGE_ROOT / "data" / "eval_decoy_resources.json"

    @field_validator("retrieval_servable_statuses", mode="before")
    @classmethod
    def _split_statuses(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        return [item.strip().lower() for item in text.split(",") if item.strip()]

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        """Accept a JSON list or a plain comma-separated string.

        This field carries NoDecode, so nothing has decoded it before we get
        here — both shapes are handled in one place instead of half by
        pydantic-settings and half by this validator.
        """
        if not isinstance(value, str):
            return value

        text = value.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Fall through: a malformed JSON list is still worth splitting
                # rather than failing with a decoder error the operator cannot act on.
                pass
        return [origin.strip() for origin in text.split(",") if origin.strip()]

    @field_validator("gemini_thinking_level", mode="before")
    @classmethod
    def _normalise_thinking_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned or None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @model_validator(mode="after")
    def _reject_wildcard_cors_in_production(self) -> Settings:
        """A wildcard origin plus bearer-token auth is a credential-leak shape."""
        if self.environment is Environment.PROD and "*" in self.cors_allow_origins:
            raise ValueError(
                "CORS_ALLOW_ORIGINS must name explicit origins in production, not '*'."
            )
        return self

    @model_validator(mode="after")
    def _require_real_resource_data_in_production(self) -> Settings:
        """Production must not fall back to the synthetic sample set.

        The bundled sample data exists for tests and offline development. Serving
        it to a member would mean handing out invented organizations and invented
        phone numbers, so production has to point somewhere else and that file
        has to exist. Failing at startup is the only acceptable outcome: a
        service that quietly answers from synthetic data looks healthy.
        """
        if self.environment is not Environment.PROD or not self.retrieval_enabled:
            return self

        if self.resource_data_path == _DEFAULT_SAMPLE_DATA_PATH:
            raise ValueError(
                "RESOURCE_DATA_PATH still points at the bundled synthetic sample "
                "data. Production must be given the real resource dataset."
            )
        if not Path(self.resource_data_path).is_file():
            raise ValueError(
                f"RESOURCE_DATA_PATH does not exist: {self.resource_data_path}. "
                "Production will not start without the real resource dataset."
            )
        return self

    @model_validator(mode="after")
    def _reject_unauthenticated_production(self) -> Settings:
        """Production may not serve the query endpoint without authentication."""
        if self.environment is Environment.PROD and not self.auth_enabled:
            raise ValueError(
                "AUTH_ENABLED must be true when ENVIRONMENT=prod: the query "
                "endpoint calls a paid model and answers personal questions."
            )
        return self

    @model_validator(mode="after")
    def _check_auth_requirements(self) -> Settings:
        """Fail at startup rather than 401-ing every caller at runtime."""
        if self.auth_enabled and not self.resolved_identity_project_id:
            raise ValueError(
                "AUTH_ENABLED requires IDENTITY_PLATFORM_PROJECT_ID (or "
                "GOOGLE_CLOUD_PROJECT) so ID tokens can be checked against an audience."
            )
        return self

    @model_validator(mode="after")
    def _check_vertex_requirements(self) -> Settings:
        """Fail fast at startup rather than on the first user request."""
        uses_vertex = (
            self.llm_provider is LLMProvider.VERTEX
            or self.embedding_provider is EmbeddingProvider.VERTEX
        )
        if uses_vertex:
            missing = [
                name
                for name, value in (
                    ("GOOGLE_CLOUD_PROJECT", self.google_cloud_project),
                    ("GOOGLE_CLOUD_LOCATION", self.google_cloud_location),
                    ("GEMINI_MODEL", self.gemini_model),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Using Vertex AI (LLM_PROVIDER and/or EMBEDDING_PROVIDER) requires: "
                    + ", ".join(missing)
                )
        return self

    @property
    def resolved_identity_project_id(self) -> str | None:
        """Identity Platform project, falling back to the Google Cloud project."""
        return self.identity_platform_project_id or self.google_cloud_project

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PROD


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (cached; clear the cache in tests)."""
    return Settings()
