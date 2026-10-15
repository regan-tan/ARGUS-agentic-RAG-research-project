"""Non-mutating validation and high-level statistics for SciFact data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .models import Claim, CorpusDocument, EvidenceLabel, SciFactDataset


class DatasetValidationError(ValueError):
    """Raised when SciFact records violate dataset integrity constraints."""


@dataclass(frozen=True)
class ExpectedSizes:
    corpus: int = 5_183
    train: int = 809
    dev: int = 300
    test: int = 300


@dataclass(frozen=True)
class EvidenceLabelConflict:
    """Distinct labels attached to rationales for one claim/document pair."""

    claim_id: int | str
    document_id: str
    labels: frozenset[EvidenceLabel]


@dataclass(frozen=True)
class DatasetValidationReport:
    """Information preserved during validation, including label conflicts."""

    label_conflicts: tuple[EvidenceLabelConflict, ...]


def _canonical_id(value: int | str) -> str:
    return str(value)


def _check_unique_ids(records: Iterable[CorpusDocument | Claim], label: str) -> None:
    seen: set[str] = set()
    for record in records:
        raw_id = record.doc_id if isinstance(record, CorpusDocument) else record.claim_id
        record_id = _canonical_id(raw_id)
        if record_id in seen:
            raise DatasetValidationError(f"Duplicate {label} ID: {raw_id!r}")
        seen.add(record_id)


def _validate_claims(
    claims: Iterable[Claim], corpus: dict[str, CorpusDocument], split: str
) -> list[EvidenceLabelConflict]:
    conflicts: list[EvidenceLabelConflict] = []
    for claim in claims:
        for cited_id in claim.cited_doc_ids:
            if _canonical_id(cited_id) not in corpus:
                raise DatasetValidationError(
                    f"{split} claim {claim.claim_id!r} cites missing corpus document {cited_id!r}"
                )

        if claim.evidence is None:
            if split != "test":
                raise DatasetValidationError(
                    f"{split} claim {claim.claim_id!r} is unexpectedly unlabeled"
                )
            continue
        if split == "test":
            raise DatasetValidationError(
                f"Test claim {claim.claim_id!r} must be treated as unlabeled"
            )

        for document_id, rationales in claim.evidence.items():
            document = corpus.get(_canonical_id(document_id))
            if document is None:
                raise DatasetValidationError(
                    f"{split} claim {claim.claim_id!r} has evidence for missing corpus document "
                    f"{document_id!r}"
                )
            labels: set[EvidenceLabel] = set()
            for rationale in rationales:
                if rationale.label not in {"SUPPORT", "CONTRADICT"}:
                    raise DatasetValidationError(
                        f"{split} claim {claim.claim_id!r} has invalid label {rationale.label!r}"
                    )
                labels.add(rationale.label)
                for sentence_index in rationale.sentences:
                    if sentence_index >= len(document.abstract):
                        raise DatasetValidationError(
                            f"{split} claim {claim.claim_id!r} references sentence "
                            f"{sentence_index} in document {document_id!r}; it has "
                            f"{len(document.abstract)} sentences"
                        )
            if len(labels) > 1:
                conflicts.append(
                    EvidenceLabelConflict(claim.claim_id, document_id, frozenset(labels))
                )
    return conflicts


def validate_scifact_dataset(
    dataset: SciFactDataset, *, expected_sizes: ExpectedSizes | None = ExpectedSizes()
) -> DatasetValidationReport:
    """Validate SciFact integrity without modifying any loaded source records.

    Label conflicts are returned in the report rather than collapsed or silently resolved.
    """

    if expected_sizes is not None:
        actual_sizes = (
            len(dataset.corpus),
            len(dataset.train_claims),
            len(dataset.dev_claims),
            len(dataset.test_claims),
        )
        required_sizes = (
            expected_sizes.corpus,
            expected_sizes.train,
            expected_sizes.dev,
            expected_sizes.test,
        )
        if actual_sizes != required_sizes:
            raise DatasetValidationError(
                "Unexpected SciFact record counts: "
                f"got corpus/train/dev/test={actual_sizes}, expected={required_sizes}"
            )

    _check_unique_ids(dataset.corpus, "corpus document")
    _check_unique_ids(dataset.train_claims, "training claim")
    _check_unique_ids(dataset.dev_claims, "development claim")
    _check_unique_ids(dataset.test_claims, "test claim")
    corpus_by_id = {_canonical_id(document.doc_id): document for document in dataset.corpus}

    conflicts = [
        *_validate_claims(dataset.train_claims, corpus_by_id, "train"),
        *_validate_claims(dataset.dev_claims, corpus_by_id, "dev"),
        *_validate_claims(dataset.test_claims, corpus_by_id, "test"),
    ]
    return DatasetValidationReport(tuple(conflicts))


def training_statistics(claims: Iterable[Claim]) -> dict[str, int]:
    """Return non-development-label summary statistics for training claims only."""

    claim_list = tuple(claims)
    label_counts: Counter[str] = Counter()
    evidence_documents = 0
    empty_evidence_claims = 0
    for claim in claim_list:
        if claim.evidence is None:
            raise DatasetValidationError("Training statistics require labeled claims")
        if not claim.evidence:
            empty_evidence_claims += 1
        evidence_documents += len(claim.evidence)
        for rationales in claim.evidence.values():
            label_counts.update(rationale.label for rationale in rationales)

    return {
        "claims": len(claim_list),
        "claims_with_empty_evidence": empty_evidence_claims,
        "evidence_documents": evidence_documents,
        "contradict_rationales": label_counts["CONTRADICT"],
        "support_rationales": label_counts["SUPPORT"],
    }
