"""Resource ingestion: file -> validated domain objects -> embedded index."""

from app.services.ingestion.documents import (
    build_embedding_text,
    build_metadata,
    build_payload,
)
from app.services.ingestion.loader import (
    LoadIssue,
    LoadResult,
    ResourceFileError,
    load_resources,
    load_resources_from_file,
)
from app.services.ingestion.pipeline import IngestionReport, ResourceIngestionPipeline

__all__ = [
    "IngestionReport",
    "LoadIssue",
    "LoadResult",
    "ResourceFileError",
    "ResourceIngestionPipeline",
    "build_embedding_text",
    "build_metadata",
    "build_payload",
    "load_resources",
    "load_resources_from_file",
]
