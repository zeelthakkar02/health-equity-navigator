#!/usr/bin/env python
"""One-shot Vertex AI connectivity check.

Sends a single harmless message through the existing ``VertexGeminiService`` to
confirm that authentication, API access, model availability, and IAM permissions
are all in place. Nothing is mocked.

Run it from the project root once the package is installed (``pip install -e .``):

    LLM_PROVIDER=vertex python scripts/verify_vertex.py

All configuration is read from the environment (or ``.env``) through the normal
``Settings`` object — this script defines no defaults of its own:

    LLM_PROVIDER          must be 'vertex'
    GOOGLE_CLOUD_PROJECT  e.g. your-gcp-project-id
    GOOGLE_CLOUD_LOCATION e.g. global
    GEMINI_MODEL          e.g. gemini-3.8-flash

Authentication is Application Default Credentials only. This script never reads,
writes, or prints a credential, key, or token — it reports the credential *type*
and the resolved project id, nothing more.

Exit codes: 0 success, 1 misconfiguration, 2 client construction failed,
3 the request failed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running directly from a checkout where the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import LLMProvider, Settings
from app.services.llm.base import LLMRequest
from app.services.llm.errors import LLMError
from app.services.llm.factory import build_llm_service

TEST_MESSAGE = "Reply with: Vertex connection successful."

# Generous enough that reasoning tokens cannot starve the visible reply.
MAX_OUTPUT_TOKENS = 2048


def check_adc() -> None:
    """Report whether ADC resolves. Prints no credential material."""
    print("STEP 1 — Application Default Credentials")
    try:
        import google.auth

        credentials, project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as exc:
        print("  result          : NOT AVAILABLE")
        print(f"  error           : {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print("  fix             : gcloud auth application-default login")
        return

    print("  result          : AVAILABLE")
    print(f"  credential type : {type(credentials).__name__}")
    print(f"  ADC project id  : {project_id or '(not set)'}")


def print_cause_chain(exc: BaseException) -> None:
    """Surface the underlying provider error (permission / API / model errors)."""
    cause: BaseException | None = exc.__cause__
    depth = 0
    while cause is not None and depth < 4:
        message = str(cause).strip().replace("\n", "\n                    ")
        print(f"  caused by       : {type(cause).__name__}: {message}")
        cause = cause.__cause__
        depth += 1


async def main() -> int:
    check_adc()
    print()

    settings = Settings()

    if settings.llm_provider is not LLMProvider.VERTEX:
        print(
            f"Refusing to run: LLM_PROVIDER is '{settings.llm_provider.value}', not 'vertex'. "
            "This script verifies the Vertex AI connection only."
        )
        return 1

    print("STEP 2 — Vertex AI request through VertexGeminiService")
    print(f"  project         : {settings.google_cloud_project}")
    print(f"  location        : {settings.google_cloud_location}")
    print(f"  model           : {settings.gemini_model}")
    print(f"  test message    : {TEST_MESSAGE!r}")
    print("  sampling params : none sent (no temperature / top_p / top_k)")
    print("  thinking_level  : not set (provider default)")
    print()

    try:
        service = build_llm_service(settings)
    except LLMError as exc:
        print(f"  result          : FAILED at client construction ({exc.code})")
        print(f"  detail          : {exc}")
        print_cause_chain(exc)
        return 2

    try:
        response = await service.generate(
            LLMRequest(prompt=TEST_MESSAGE, max_output_tokens=MAX_OUTPUT_TOKENS)
        )
    except LLMError as exc:
        print(f"  result          : FAILED at generate_content ({exc.code})")
        print(f"  detail          : {exc}")
        print_cause_chain(exc)
        return 3
    finally:
        await service.aclose()

    print("  result          : SUCCESS")
    print(f"  model returned  : {response.model}")
    print(f"  finish reason   : {response.finish_reason}")
    print(f"  usage           : {response.usage}")
    print(f"  text            : {response.text.strip()!r}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
