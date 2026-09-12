"""Loading resources from a data file.

One bad record must not sink a whole ingest: each entry is validated on its own
and rejects are reported rather than raised. That is the behaviour we want when
real directory exports arrive in a later phase.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.domain.resource import Resource

logger = logging.getLogger(__name__)


class ResourceFileError(Exception):
    """The data file is missing or is not shaped like a resource file."""


@dataclass(frozen=True)
class LoadIssue:
    """One record that could not be loaded."""

    index: int
    resource_id: str | None
    reason: str


@dataclass(frozen=True)
class LoadResult:
    resources: list[Resource] = field(default_factory=list)
    issues: list[LoadIssue] = field(default_factory=list)

    @property
    def loaded_count(self) -> int:
        return len(self.resources)

    @property
    def rejected_count(self) -> int:
        return len(self.issues)


def load_resources(raw_records: Iterable[Mapping[str, Any]]) -> LoadResult:
    """Validate records into :class:`Resource` objects, collecting failures."""
    resources: list[Resource] = []
    issues: list[LoadIssue] = []
    seen_ids: set[str] = set()

    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            issues.append(LoadIssue(index, None, "record is not an object"))
            continue

        raw_id = raw.get("resource_id")
        candidate_id = str(raw_id) if raw_id is not None else None

        try:
            resource = Resource.model_validate(raw)
        except ValidationError as exc:
            issues.append(LoadIssue(index, candidate_id, _summarise(exc)))
            continue

        if resource.resource_id in seen_ids:
            issues.append(LoadIssue(index, resource.resource_id, "duplicate resource_id"))
            continue

        seen_ids.add(resource.resource_id)
        resources.append(resource)

    return LoadResult(resources=resources, issues=issues)


def load_resources_from_file(path: Path | str) -> LoadResult:
    """Read a resource JSON file.

    Accepts either a bare list of records or an object with a ``resources`` key
    (which is how the sample file carries its "synthetic data" warning block).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ResourceFileError(f"Resource data file not found: {file_path}")

    try:
        document = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResourceFileError(f"{file_path} is not valid JSON: {exc}") from exc

    if isinstance(document, Mapping):
        records = document.get("resources")
        if records is None:
            raise ResourceFileError(f"{file_path} has no 'resources' key.")
    else:
        records = document

    if not isinstance(records, list):
        raise ResourceFileError(f"{file_path} must contain a list of resources.")

    result = load_resources(records)
    logger.info(
        "Loaded %d resources from %s (%d rejected)",
        result.loaded_count,
        file_path,
        result.rejected_count,
    )
    return result


def _summarise(error: ValidationError) -> str:
    """Compact, log-safe description of why a record failed validation."""
    parts = []
    for item in error.errors()[:3]:
        location = ".".join(str(piece) for piece in item["loc"]) or "record"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)
