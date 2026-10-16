"""Benchmark-oriented retrieval evaluation helpers."""

from __future__ import annotations

import statistics
import time
from collections.abc import Mapping, Sequence

from argus.data.models import Claim
from argus.retrieval.hybrid import Retriever
from argus.retrieval.types import RetrievalResult, canonical_document_id

from .retrieval import (
    MetricSummary,
    evidence_category,
    gold_document_ids,
    summarize_by_category,
    summarize_metrics,
)


def latency_summary(samples_seconds: Sequence[float]) -> dict[str, float | int]:
    """Summarize measured warm query durations without including index construction."""

    if not samples_seconds:
        return {"queries": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(samples_seconds)
    percentile_index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95)))
    return {
        "queries": len(samples_seconds),
        "mean_ms": statistics.fmean(samples_seconds) * 1_000,
        "median_ms": statistics.median(samples_seconds) * 1_000,
        "p95_ms": ordered[percentile_index] * 1_000,
    }


def retrieve_and_measure(
    retriever: Retriever, claims: Sequence[Claim], k: int
) -> tuple[dict[str, tuple[RetrievalResult, ...]], dict[str, float | int]]:
    """Warm a retriever once, then measure one retrieval per claim."""

    if claims:
        retriever.retrieve(claims[0].claim, k)
    predictions: dict[str, tuple[RetrievalResult, ...]] = {}
    durations: list[float] = []
    for claim in claims:
        started = time.perf_counter()
        predictions[str(claim.claim_id)] = retriever.retrieve(claim.claim, k)
        durations.append(time.perf_counter() - started)
    return predictions, latency_summary(durations)


def metric_payload(summary: MetricSummary) -> dict[str, object]:
    """Convert a typed metric summary to JSON-safe, self-describing data."""

    return {
        "recall_at_k": {str(k): value for k, value in summary.recall_at_k.items()},
        "hit_at_k": {str(k): value for k, value in summary.hit_at_k.items()},
        "eligible_claims": summary.eligible_claims,
        "excluded_no_evidence_claims": summary.excluded_no_evidence_claims,
    }


def per_claim_payload(
    claims: Sequence[Claim],
    predictions: Mapping[str, Sequence[RetrievalResult]],
    k_values: Sequence[int],
) -> list[dict[str, object]]:
    """Make inspectable per-claim benchmark records, retaining gold only for evaluation."""

    records: list[dict[str, object]] = []
    for claim in claims:
        gold = gold_document_ids(claim)
        results = predictions[str(claim.claim_id)]
        record: dict[str, object] = {
            "claim_id": claim.claim_id,
            "claim": claim.claim,
            "evidence_category": evidence_category(claim),
            "gold_document_ids": sorted(gold),
            "retrieved_documents": [
                {
                    "doc_id": result.doc_id,
                    "rank": result.rank,
                    "score": result.score,
                    "title": result.document.title,
                }
                for result in results
            ],
            "gold_ranks": {
                str(result.doc_id): result.rank
                for result in results
                if canonical_document_id(result.doc_id) in gold
            },
        }
        record["hit_at_k"] = {
            str(k): bool(
                gold.intersection(canonical_document_id(result.doc_id) for result in results[:k])
            )
            for k in sorted(set(k_values))
        }
        records.append(record)
    return records


def qualitative_examples(
    claims: Sequence[Claim],
    predictions_by_method: Mapping[str, Mapping[str, Sequence[RetrievalResult]]],
    document_titles: Mapping[str, str],
    k: int,
) -> dict[str, dict[str, object] | None]:
    """Select at most one deterministic representative for requested error patterns."""

    def hit(method: str, claim: Claim) -> bool:
        gold = gold_document_ids(claim)
        return bool(
            gold.intersection(
                canonical_document_id(result.doc_id)
                for result in predictions_by_method[method][str(claim.claim_id)][:k]
            )
        )

    def render(claim: Claim) -> dict[str, object]:
        gold = gold_document_ids(claim)
        return {
            "claim_id": claim.claim_id,
            "claim": claim.claim,
            "evidence_category": evidence_category(claim),
            "gold_document_ids": sorted(gold),
            "gold_documents": [
                {"doc_id": document_id, "title": document_titles.get(document_id)}
                for document_id in sorted(gold)
            ],
            "methods": {
                method: [
                    {"doc_id": result.doc_id, "rank": result.rank, "title": result.document.title}
                    for result in predictions[str(claim.claim_id)][:k]
                ]
                for method, predictions in predictions_by_method.items()
            },
        }

    ordered_claims = sorted(claims, key=lambda claim: str(claim.claim_id))
    conditions = {
        "bm25_fails_dense_succeeds": lambda claim: not hit("bm25", claim) and hit("dense", claim),
        "dense_fails_bm25_succeeds": lambda claim: not hit("dense", claim) and hit("bm25", claim),
        "hybrid_succeeds_when_underlying_method_misses": lambda claim: (
            hit("hybrid", claim) and (not hit("bm25", claim) or not hit("dense", claim))
        ),
        "all_methods_fail": lambda claim: all(
            not hit(method, claim) for method in predictions_by_method
        ),
        "contradict_difficult": lambda claim: (
            evidence_category(claim) == "contradict"
            and all(not hit(method, claim) for method in predictions_by_method)
        ),
    }
    examples: dict[str, dict[str, object] | None] = {}
    for name, condition in conditions.items():
        example = next(
            (claim for claim in ordered_claims if gold_document_ids(claim) and condition(claim)),
            None,
        )
        examples[name] = render(example) if example else None
    return examples


def summarize_method(
    claims: Sequence[Claim],
    predictions: Mapping[str, Sequence[RetrievalResult]],
    k_values: Sequence[int],
) -> dict[str, object]:
    """Return overall and label-category retrieval metrics for one method."""

    return {
        "overall": metric_payload(summarize_metrics(claims, predictions, k_values)),
        "by_evidence_category": {
            category: metric_payload(summary)
            for category, summary in summarize_by_category(claims, predictions, k_values).items()
        },
    }
