"""Loading utilities for the original AllenAI SciFact JSONL release."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from .models import (
    Claim,
    CorpusDocument,
    DocumentId,
    EvidenceAnnotation,
    EvidenceLabel,
    SciFactDataset,
)


class SciFactDataError(ValueError):
    """Raised when a SciFact input file cannot be read."""


class SciFactSchemaError(SciFactDataError):
    """Raised when a JSONL record does not match the expected SciFact schema."""


def _records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise SciFactDataError(f"SciFact file is missing: {path}")
    if not path.is_file():
        raise SciFactDataError(f"SciFact path is not a file: {path}")

    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SciFactDataError(
                        f"Malformed JSON in {path} at line {line_number}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise SciFactSchemaError(
                        "Expected an object in "
                        f"{path} at line {line_number}, got {type(record).__name__}"
                    )
                yield line_number, record
    except OSError as error:
        raise SciFactDataError(f"Could not read SciFact file {path}: {error}") from error


def _schema_error(path: Path, line_number: int, message: str) -> SciFactSchemaError:
    return SciFactSchemaError(f"Invalid SciFact record in {path} at line {line_number}: {message}")


def _require(record: dict[str, Any], field: str, path: Path, line_number: int) -> Any:
    if field not in record:
        raise _schema_error(path, line_number, f"missing required field '{field}'")
    return record[field]


def _is_id(value: Any) -> bool:
    return isinstance(value, (int, str)) and not isinstance(value, bool)


def _parse_document_id(value: Any, path: Path, line_number: int, field: str) -> DocumentId:
    if not _is_id(value):
        raise _schema_error(path, line_number, f"field '{field}' must be an integer or string ID")
    return cast(DocumentId, value)


def load_corpus(path: Path) -> tuple[CorpusDocument, ...]:
    """Load corpus records without changing sentence order or document IDs."""

    documents: list[CorpusDocument] = []
    for line_number, record in _records(path):
        doc_id = _parse_document_id(
            _require(record, "doc_id", path, line_number), path, line_number, "doc_id"
        )
        title = _require(record, "title", path, line_number)
        abstract = _require(record, "abstract", path, line_number)
        structured = _require(record, "structured", path, line_number)
        if not isinstance(title, str):
            raise _schema_error(path, line_number, "field 'title' must be a string")
        if not isinstance(abstract, list) or not all(
            isinstance(sentence, str) for sentence in abstract
        ):
            raise _schema_error(path, line_number, "field 'abstract' must be a list of strings")
        if not isinstance(structured, bool):
            raise _schema_error(path, line_number, "field 'structured' must be a boolean")
        documents.append(CorpusDocument(doc_id, title, tuple(abstract), structured))
    return tuple(documents)


def _parse_evidence(
    value: Any, path: Path, line_number: int
) -> dict[str, tuple[EvidenceAnnotation, ...]]:
    if not isinstance(value, dict):
        raise _schema_error(path, line_number, "field 'evidence' must be a mapping of document IDs")

    evidence: dict[str, tuple[EvidenceAnnotation, ...]] = {}
    for document_id, raw_annotations in value.items():
        if not isinstance(document_id, str):
            raise _schema_error(path, line_number, "evidence document IDs must be strings")
        if not isinstance(raw_annotations, list):
            raise _schema_error(
                path, line_number, f"evidence for document '{document_id}' must be a list"
            )

        parsed: list[EvidenceAnnotation] = []
        for annotation in raw_annotations:
            if not isinstance(annotation, dict):
                raise _schema_error(path, line_number, "each evidence annotation must be an object")
            label = annotation.get("label")
            sentences = annotation.get("sentences")
            if label not in {"SUPPORT", "CONTRADICT"}:
                raise _schema_error(
                    path,
                    line_number,
                    "evidence label must be 'SUPPORT' or 'CONTRADICT'",
                )
            if not isinstance(sentences, list) or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in sentences
            ):
                raise _schema_error(
                    path,
                    line_number,
                    "evidence sentences must be a list of non-negative integer indices",
                )
            parsed.append(EvidenceAnnotation(cast(EvidenceLabel, label), tuple(sentences)))
        evidence[document_id] = tuple(parsed)
    return evidence


def load_claims(path: Path, *, labels_required: bool) -> tuple[Claim, ...]:
    """Load claims, retaining evidence at its original document/rationale granularity."""

    claims: list[Claim] = []
    for line_number, record in _records(path):
        claim_id = _parse_document_id(
            _require(record, "id", path, line_number), path, line_number, "id"
        )
        claim_text = _require(record, "claim", path, line_number)
        cited_doc_ids = (
            _require(record, "cited_doc_ids", path, line_number)
            if labels_required
            else record.get("cited_doc_ids", [])
        )
        if not isinstance(claim_text, str):
            raise _schema_error(path, line_number, "field 'claim' must be a string")
        if not isinstance(cited_doc_ids, list):
            raise _schema_error(path, line_number, "field 'cited_doc_ids' must be a list")
        parsed_citations = tuple(
            _parse_document_id(item, path, line_number, "cited_doc_ids") for item in cited_doc_ids
        )

        if labels_required:
            evidence = _parse_evidence(
                _require(record, "evidence", path, line_number), path, line_number
            )
        else:
            raw_evidence = record.get("evidence")
            if raw_evidence not in (None, {}):
                raise _schema_error(
                    path,
                    line_number,
                    "unlabeled claims must not contain evidence annotations",
                )
            evidence = None
        claims.append(Claim(claim_id, claim_text, parsed_citations, evidence))
    return tuple(claims)


def load_scifact_dataset(data_dir: Path) -> SciFactDataset:
    """Load the official SciFact corpus and three claim splits from ``data_dir``."""

    return SciFactDataset(
        corpus=load_corpus(data_dir / "corpus.jsonl"),
        train_claims=load_claims(data_dir / "claims_train.jsonl", labels_required=True),
        dev_claims=load_claims(data_dir / "claims_dev.jsonl", labels_required=True),
        test_claims=load_claims(data_dir / "claims_test.jsonl", labels_required=False),
    )
