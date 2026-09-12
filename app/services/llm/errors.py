"""Exception hierarchy for the LLM layer.

The API layer maps these onto HTTP status codes, so provider-specific SDK
exceptions never leak past this package.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every failure originating in the LLM layer."""

    code = "llm_error"


class LLMConfigurationError(LLMError):
    """The provider cannot be constructed: missing config, SDK, or credentials."""

    code = "llm_configuration_error"


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""

    code = "llm_timeout"


class LLMUpstreamError(LLMError):
    """The provider was reached but returned an error or an unusable response."""

    code = "llm_upstream_error"
