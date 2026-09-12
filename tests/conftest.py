"""Shared test fixtures.

Settings are always built explicitly with ``_env_file=None`` so a developer's
local .env can never change what the suite asserts.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Environment, LLMProvider, Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment=Environment.LOCAL,
        llm_provider=LLMProvider.STUB,
        cors_allow_origins=["http://localhost:3000"],
        # Tests must never read or write the developer's on-disk index cache.
        resource_index_cache_enabled=False,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A client whose app has completed startup (lifespan builds the provider)."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
