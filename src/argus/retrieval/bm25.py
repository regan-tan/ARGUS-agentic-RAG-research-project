"""Deterministic BM25 retrieval over complete SciFact abstracts."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from argus.data.models import CorpusDocument

from .types import RetrievalResult, canonical_document_id, document_text

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Apply the shared lowercase regex tokenizer to document and query text."""

    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """A small, in-memory BM25Okapi retriever with stable tie handling."""

    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        self.documents = tuple(documents)
        document_ids = [canonical_document_id(document.doc_id) for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("BM25 documents must have unique document IDs")
        self._index = BM25Okapi([tokenize(document_text(document)) for document in self.documents])

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        """Return up to ``k`` documents, or no results for an empty query or nonpositive k."""

        if k <= 0 or not query.strip() or not self.documents:
            return ()
        scores = self._index.get_scores(tokenize(query))
        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda index: (
                -float(scores[index]),
                canonical_document_id(self.documents[index].doc_id),
            ),
        )[: min(k, len(self.documents))]
        return tuple(
            RetrievalResult(
                doc_id=self.documents[index].doc_id,
                score=float(scores[index]),
                rank=rank,
                document=self.documents[index],
            )
            for rank, index in enumerate(ranked_indices, start=1)
        )
