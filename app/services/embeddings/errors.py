"""Exception hierarchy for the embedding layer.

Mirrors :mod:`app.services.llm.errors`: provider-specific SDK exceptions never
escape this package.
"""

from __future__ import annotations


class EmbeddingError(Exception):
    """Base class for every failure originating in the embedding layer."""

    code = "embedding_error"


class EmbeddingConfigurationError(EmbeddingError):
    """The provider cannot be constructed: missing config, SDK, or credentials."""

    code = "embedding_configuration_error"


class EmbeddingTimeoutError(EmbeddingError):
    """The provider did not answer within the configured timeout."""

    code = "embedding_timeout"


class EmbeddingUpstreamError(EmbeddingError):
    """The provider was reached but returned an error or an unusable response."""

    code = "embedding_upstream_error"
