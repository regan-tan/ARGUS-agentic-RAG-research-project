"""Typed representations of the official SciFact JSONL records."""

from dataclasses import dataclass
from typing import Literal

type DocumentId = int | str
type EvidenceLabel = Literal["SUPPORT", "CONTRADICT"]


@dataclass(frozen=True)
class CorpusDocument:
    """One SciFact corpus abstract with sentences in their source order."""

    doc_id: DocumentId
    title: str
    abstract: tuple[str, ...]
    structured: bool


@dataclass(frozen=True)
class EvidenceAnnotation:
    """One evidence-level rationale annotation for a cited corpus document."""

    label: EvidenceLabel
    sentences: tuple[int, ...]


@dataclass(frozen=True)
class Claim:
    """One SciFact claim and its unflattened evidence annotations.

    ``evidence is None`` denotes an unlabeled claim (the official test split).
    ``evidence == {}`` denotes a labeled split record with no annotated evidence.
    """

    claim_id: int | str
    claim: str
    cited_doc_ids: tuple[DocumentId, ...]
    evidence: dict[str, tuple[EvidenceAnnotation, ...]] | None


@dataclass(frozen=True)
class SciFactDataset:
    """The four official SciFact files loaded together."""

    corpus: tuple[CorpusDocument, ...]
    train_claims: tuple[Claim, ...]
    dev_claims: tuple[Claim, ...]
    test_claims: tuple[Claim, ...]
