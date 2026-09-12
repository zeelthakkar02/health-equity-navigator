"""The Vertex providers must stay isolated.

The service has to run with LLM_PROVIDER=stub and EMBEDDING_PROVIDER=hashing on a
machine with no Google Cloud credentials and no cloud SDK. These tests pin that
guarantee: importing the app or the retrieval stack — both of which import
factories that import the Vertex modules — must not pull in ``google.genai`` or
touch credentials.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from app.services.llm.errors import LLMConfigurationError
from app.services.llm.vertex_provider import VertexGeminiService

_PROBE = """
import sys
import app.main  # noqa: F401
from app.services.llm import vertex_provider  # noqa: F401
from app.services.embeddings import vertex_provider as embedding_vertex_provider  # noqa: F401
from app.services.retrieval import bootstrap  # noqa: F401
leaked = [name for name in sys.modules if name.startswith("google.genai")]
print("LEAKED" if leaked else "CLEAN")
"""


def test_importing_the_app_does_not_import_the_google_sdk() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "CLEAN", result.stderr


def test_vertex_service_rejects_an_empty_project() -> None:
    with pytest.raises(LLMConfigurationError, match="GOOGLE_CLOUD_PROJECT"):
        VertexGeminiService(project="", location="global", model="gemini-3.8-flash")


def test_embedding_vertex_service_rejects_an_empty_project() -> None:
    from app.services.embeddings.errors import EmbeddingConfigurationError
    from app.services.embeddings.vertex_provider import VertexEmbeddingService

    with pytest.raises(EmbeddingConfigurationError, match="GOOGLE_CLOUD_PROJECT"):
        VertexEmbeddingService(
            project="", location="global", model="gemini-embedding-001", dimensions=768
        )
