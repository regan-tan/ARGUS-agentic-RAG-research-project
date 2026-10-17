"""Small, append-only helpers for interruption-safe Standard RAG experiment artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO


class ArtifactError(ValueError):
    """An experiment artifact is malformed or incompatible with a requested resume."""


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object, raising an actionable error for an invalid artifact."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ArtifactError(f"Could not read artifact {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ArtifactError(f"Artifact {path} contains invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"Artifact {path} must contain a JSON object")
    return value


def load_prediction_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load append-only prediction rows and reject malformed or duplicate claim IDs."""

    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ArtifactError(f"Could not read prediction artifact {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactError(
                f"Prediction artifact {path} has invalid JSON on line {line_number}: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ArtifactError(
                f"Prediction artifact {path} has a non-object row on line {line_number}"
            )
        claim_id = record.get("claim_id")
        if not isinstance(claim_id, (int, str)) or isinstance(claim_id, bool):
            raise ArtifactError(
                f"Prediction artifact {path} has an invalid claim_id on line {line_number}"
            )
        key = str(claim_id)
        if key in records:
            raise ArtifactError(
                f"Prediction artifact {path} has duplicate claim_id {claim_id!r}; "
                "refusing to combine ambiguous records"
            )
        records[key] = record
    return records


def ensure_resume_compatible(
    metadata: Mapping[str, Any], expected_configuration: Mapping[str, Any]
) -> None:
    """Reject combining predictions when a correctness-relevant configuration differs."""

    recorded = metadata.get("resume_configuration")
    if not isinstance(recorded, dict):
        raise ArtifactError(
            "This run predates resumable artifacts or has no resume_configuration; "
            "start a new run rather than combining records."
        )
    if recorded != dict(expected_configuration):
        raise ArtifactError(
            "Resume configuration does not match the existing run. Model identity, retrieval, "
            "prompt, split, or selected claims differ; refusing to combine results."
        )


def append_prediction_record(handle: TextIO, record: Mapping[str, object]) -> None:
    """Durably append exactly one completed claim record before continuing the run."""

    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
