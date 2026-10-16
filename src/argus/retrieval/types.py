"""Shared types and document-text helpers for retrieval backends."""

from __future__ import annotations

from dataclasses import dataclass

from argus.data.models import CorpusDocument, DocumentId


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked document returned by a retriever."""

    doc_id: DocumentId
    score: float
    rank: int
    document: CorpusDocument


def canonical_document_id(doc_id: DocumentId) -> str:
    """Return a stable comparable representation without changing the source ID."""

    return str(doc_id)


def document_text(document: CorpusDocument) -> str:
    """Build the one-document retrieval representation used by every backend."""

    return " ".join(part for part in (document.title, *document.abstract) if part).strip()
