"""Offline stub provider.

The default backend. It performs no network calls and needs no credentials, so
the full request path — routing, validation, prompt assembly, response shaping —
is exercisable on any machine and in CI. Output is deterministic for a given
request, which keeps tests stable.
"""

from __future__ import annotations

from app.services.llm.base import LLMRequest, LLMResponse, LLMService
from app.services.llm.prompts import MESSAGE_CLOSE_TAG, MESSAGE_OPEN_TAG

STUB_MODEL_NAME = "stub-navigator-v1"

_NO_RESOURCES_MARKER = "(none available"


class StubLLMService(LLMService):
    """Deterministic, dependency-free stand-in for a real model."""

    provider_name = "stub"
    model_name = STUB_MODEL_NAME

    async def generate(self, request: LLMRequest) -> LLMResponse:
        text = self._compose(request.prompt)
        return LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            finish_reason="stop",
            usage={
                # Rough word-count proxies; the stub never calls a tokenizer.
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(text.split()),
            },
        )

    def _compose(self, prompt: str) -> str:
        need = _extract_need(prompt)
        grounded = _NO_RESOURCES_MARKER not in prompt

        lines = [
            "[STUB RESPONSE — no live model was called]",
            "",
            f"Here is what I understood you need help with: {need}",
            "",
        ]
        if grounded:
            lines += [
                "I would answer using only the verified community resources listed for "
                "this request, citing each one by its bracketed number.",
            ]
        else:
            lines += [
                "I do not have any verified local resources to share for this request yet. "
                "For local referrals to health care, food, housing, and utility help, call "
                "211 or visit 211.org. If this is a medical emergency, call 911.",
            ]
        lines += [
            "",
            "Set LLM_PROVIDER=vertex with Application Default Credentials configured to "
            "get a real Gemini response.",
        ]
        return "\n".join(lines)


def _extract_need(prompt: str) -> str:
    """Pull the community member's words back out of the assembled prompt.

    Reads the delimiter the prompt builder exports rather than re-encoding the
    prompt's wording here, which drifted twice before.
    """
    if MESSAGE_OPEN_TAG not in prompt:
        stripped = prompt.strip()
        return stripped.splitlines()[0] if stripped else ""
    tail = prompt.split(MESSAGE_OPEN_TAG, 1)[1]
    return tail.split(MESSAGE_CLOSE_TAG, 1)[0].strip()
