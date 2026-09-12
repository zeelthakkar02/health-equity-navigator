"""LLM abstraction layer.

Nothing outside this package talks to a model SDK directly. Importing this
package must never import a cloud SDK — provider modules import their SDK lazily
so the service runs offline with the stub provider.
"""

from app.services.llm.base import LLMRequest, LLMResponse, LLMService
from app.services.llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.services.llm.factory import build_llm_service

__all__ = [
    "LLMConfigurationError",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "LLMService",
    "LLMTimeoutError",
    "LLMUpstreamError",
    "build_llm_service",
]
