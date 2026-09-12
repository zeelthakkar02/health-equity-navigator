"""Configuration behaviour."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Environment, LLMProvider, Settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults_are_offline_safe() -> None:
    settings = _settings()
    assert settings.llm_provider is LLMProvider.STUB
    assert settings.environment is Environment.LOCAL
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.google_cloud_location == "global"
    assert settings.gemini_model == "gemini-3.8-flash"


def test_cors_origins_accept_comma_separated_string() -> None:
    settings = _settings(cors_allow_origins="http://a.test, http://b.test")
    assert settings.cors_allow_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_parse_from_an_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path that actually matters in production, and the one that broke.

    Passing the value as a constructor argument skips pydantic-settings'
    environment source entirely, so it cannot catch a decoding failure there.
    """
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://a.test,https://b.test")

    assert _settings().cors_allow_origins == ["https://a.test", "https://b.test"]


def test_a_single_cors_origin_parses_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://portal.example")

    assert _settings().cors_allow_origins == ["https://portal.example"]


def test_json_list_cors_origins_still_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", '["https://a.test", "https://b.test"]')

    assert _settings().cors_allow_origins == ["https://a.test", "https://b.test"]


def test_log_level_is_normalised() -> None:
    assert _settings(log_level="debug").log_level == "DEBUG"


def test_vertex_requires_a_project() -> None:
    with pytest.raises(ValidationError, match="GOOGLE_CLOUD_PROJECT"):
        _settings(llm_provider=LLMProvider.VERTEX, google_cloud_project=None)


def test_vertex_config_is_accepted_when_complete() -> None:
    settings = _settings(
        llm_provider=LLMProvider.VERTEX,
        google_cloud_project="example-gcp-project",
        google_cloud_location="global",
        gemini_model="gemini-3.8-flash",
    )
    assert settings.llm_provider is LLMProvider.VERTEX
    assert settings.google_cloud_project == "example-gcp-project"


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "256")
    settings = _settings()
    assert settings.log_level == "WARNING"
    assert settings.llm_max_output_tokens == 256


def test_min_score_is_unset_by_default_so_the_provider_default_applies() -> None:
    assert _settings().retrieval_min_score is None


def test_min_score_can_be_pinned_explicitly() -> None:
    assert _settings(retrieval_min_score=0.42).retrieval_min_score == 0.42
