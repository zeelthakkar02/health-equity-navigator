"""The LLMService abstraction every provider implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """Provider-agnostic generation request."""

    prompt: str = Field(min_length=1)
    system_instruction: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Non-sensitive correlation data (e.g. request_id)."
    )


class LLMResponse(BaseModel):
    """Provider-agnostic generation result."""

    text: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)


class LLMService(ABC):
    """Interface for runtime AI backends.

    Implementations must not raise provider-native SDK exceptions; they wrap
    failures in :mod:`app.services.llm.errors` types.
    """

    provider_name: str = "unknown"
    model_name: str = "unknown"

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for ``request``."""

    async def aclose(self) -> None:
        """Release provider resources. Overridden when a client needs teardown."""
        return None

    def describe(self) -> dict[str, str]:
        """Non-sensitive provider description, safe for /health and logs."""
        return {"provider": self.provider_name, "model": self.model_name}
