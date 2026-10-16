"""Reciprocal Rank Fusion of lexical and dense retrieval rankings."""

from __future__ import annotations

from typing import Protocol

from .types import RetrievalResult, canonical_document_id


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]: ...


class HybridRetriever:
    """Fuse BM25 and dense rankings with fixed, score-scale-independent RRF."""

    def __init__(
        self,
        bm25: Retriever,
        dense: Retriever,
        *,
        candidate_pool: int = 50,
        rrf_constant: int = 60,
    ) -> None:
        if candidate_pool <= 0 or rrf_constant < 0:
            raise ValueError("candidate_pool must be positive and rrf_constant non-negative")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_pool = candidate_pool
        self.rrf_constant = rrf_constant

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        if k <= 0 or not query.strip():
            return ()
        candidate_count = max(k, self.candidate_pool)
        rankings = (
            self.bm25.retrieve(query, candidate_count),
            self.dense.retrieve(query, candidate_count),
        )
        fused_scores: dict[str, float] = {}
        by_id: dict[str, RetrievalResult] = {}
        for ranking in rankings:
            for result in ranking:
                identifier = canonical_document_id(result.doc_id)
                fused_scores[identifier] = fused_scores.get(identifier, 0.0) + 1.0 / (
                    self.rrf_constant + result.rank
                )
                by_id.setdefault(identifier, result)
        ordered_ids = sorted(
            fused_scores, key=lambda identifier: (-fused_scores[identifier], identifier)
        )
        return tuple(
            RetrievalResult(
                doc_id=by_id[identifier].doc_id,
                score=fused_scores[identifier],
                rank=rank,
                document=by_id[identifier].document,
            )
            for rank, identifier in enumerate(ordered_ids[:k], start=1)
        )
