from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.data.models import SciFactDataset
from argus.data.scifact import (
    SciFactDataError,
    SciFactSchemaError,
    load_claims,
    load_corpus,
    load_scifact_dataset,
)
from argus.data.validation import DatasetValidationError, validate_scifact_dataset


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {
                "doc_id": 10,
                "title": "First document",
                "abstract": ["Sentence zero.", "Sentence one."],
                "structured": False,
            },
            {
                "doc_id": "20",
                "title": "Second document",
                "abstract": ["Another sentence."],
                "structured": True,
            },
        ],
    )
    write_jsonl(
        tmp_path / "claims_train.jsonl",
        [
            {
                "id": 1,
                "claim": "A training claim.",
                "cited_doc_ids": [10, "20"],
                "evidence": {
                    "10": [
                        {"label": "SUPPORT", "sentences": [0, 1]},
                        {"label": "SUPPORT", "sentences": [1]},
                    ],
                    "20": [{"label": "CONTRADICT", "sentences": [0]}],
                },
            },
            {"id": 2, "claim": "No annotation.", "cited_doc_ids": [], "evidence": {}},
        ],
    )
    write_jsonl(
        tmp_path / "claims_dev.jsonl",
        [{"id": 3, "claim": "Development claim.", "cited_doc_ids": [10], "evidence": {}}],
    )
    write_jsonl(
        tmp_path / "claims_test.jsonl",
        [{"id": 4, "claim": "Unlabeled claim."}],
    )
    return tmp_path


def test_loads_and_preserves_multiple_evidence_rationales(dataset_dir: Path) -> None:
    dataset = load_scifact_dataset(dataset_dir)
    claim = dataset.train_claims[0]

    assert dataset.corpus[0].abstract == ("Sentence zero.", "Sentence one.")
    assert claim.cited_doc_ids == (10, "20")
    assert claim.evidence is not None
    assert tuple(claim.evidence) == ("10", "20")
    assert claim.evidence["10"][0].sentences == (0, 1)
    assert claim.evidence["10"][1].label == "SUPPORT"
    validate_scifact_dataset(dataset, expected_sizes=None)


def test_empty_evidence_and_unlabeled_test_claim_are_distinct(dataset_dir: Path) -> None:
    dataset = load_scifact_dataset(dataset_dir)

    assert dataset.train_claims[1].evidence == {}
    assert dataset.test_claims[0].evidence is None
    assert dataset.test_claims[0].cited_doc_ids == ()


def test_validation_rejects_missing_corpus_references(dataset_dir: Path) -> None:
    dataset = load_scifact_dataset(dataset_dir)
    bad_claim = dataset.train_claims[0].__class__(
        claim_id=1,
        claim="Bad reference",
        cited_doc_ids=(999,),
        evidence={},
    )
    bad_dataset = SciFactDataset(
        dataset.corpus, (bad_claim,), dataset.dev_claims, dataset.test_claims
    )

    with pytest.raises(DatasetValidationError, match="missing corpus document"):
        validate_scifact_dataset(bad_dataset, expected_sizes=None)


def test_validation_rejects_out_of_range_rationale_index(dataset_dir: Path) -> None:
    path = dataset_dir / "claims_train.jsonl"
    records = [
        {
            "id": 1,
            "claim": "Bad rationale.",
            "cited_doc_ids": [10],
            "evidence": {"10": [{"label": "SUPPORT", "sentences": [8]}]},
        }
    ]
    write_jsonl(path, records)
    dataset = load_scifact_dataset(dataset_dir)

    with pytest.raises(DatasetValidationError, match="references sentence 8"):
        validate_scifact_dataset(dataset, expected_sizes=None)


def test_validation_rejects_duplicate_ids(dataset_dir: Path) -> None:
    path = dataset_dir / "corpus.jsonl"
    document = {
        "doc_id": 10,
        "title": "Duplicate",
        "abstract": ["A sentence."],
        "structured": False,
    }
    write_jsonl(path, [document, document])
    dataset = load_scifact_dataset(dataset_dir)

    with pytest.raises(DatasetValidationError, match="Duplicate corpus document ID"):
        validate_scifact_dataset(dataset, expected_sizes=None)


def test_loader_reports_malformed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"doc_id": 1, "title": "Valid", "abstract": [], "structured": false}\nnot-json\n',
        encoding="utf-8",
    )

    with pytest.raises(SciFactDataError, match="Malformed JSON"):
        load_corpus(path)


def test_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SciFactDataError, match="missing"):
        load_corpus(tmp_path / "absent.jsonl")


def test_loader_rejects_invalid_evidence_structure(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    write_jsonl(
        path,
        [{"id": 1, "claim": "Claim", "cited_doc_ids": [], "evidence": {"1": {}}}],
    )

    with pytest.raises(SciFactSchemaError, match="must be a list"):
        load_claims(path, labels_required=True)
