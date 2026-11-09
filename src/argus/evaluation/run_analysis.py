"""Offline trajectory, cost, ranking and failure-attribution analysis of saved prediction records.

These helpers only read the ``predictions.jsonl`` records that the Standard RAG, Iterative RAG and
ARGUS runs already write. They make no model or retrieval calls, so they cannot change a result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from argus.data.models import Claim
from argus.evaluation.retrieval import ndcg_at_k, reciprocal_rank
from argus.retrieval.types import canonical_document_id

type PredictionRecord = Mapping[str, Any]

VERDICT_CALL_PURPOSES = frozenset({"verdict", "judge"})
VERIFIER_CALL_PURPOSE = "evidence_verifier"
EVIDENCE_LABELS = frozenset({"SUPPORT", "CONTRADICT"})

FAILURE_CATEGORIES = (
    "invalid_output",
    "retrieval_miss",
    "verifier_wrong_evidence",
    "conflicting_evidence",
    "verifier_miss",
    "verdict",
)


class RunAnalysisError(ValueError):
    """Raised when saved prediction records cannot be analyzed consistently."""


def is_correct(record: PredictionRecord) -> bool:
    """A claim is correct only when a valid response's verdict equals the gold verdict."""

    return record.get("status") == "success" and record.get("prediction") == record["gold_label"]


def llm_calls(record: PredictionRecord) -> list[Mapping[str, Any]]:
    """Return recorded LLM calls; Standard RAG stores its single call outside ``trace``."""

    trace = record.get("trace")
    if trace is not None:
        return [step for step in trace if step.get("type") == "llm_call"]
    if record.get("model_call_attempted") and record.get("generation") is not None:
        return [{"purpose": "verdict", "generation": record["generation"]}]
    return []


def retrieval_call_count(record: PredictionRecord) -> int:
    counts = record.get("counts")
    if counts is not None:
        return int(counts["retrieval_calls"])
    return int(bool(record.get("retrieval_call_attempted")))


def _call_failed(call: Mapping[str, Any], record: PredictionRecord) -> bool:
    if "error" in call:
        return True
    status = call.get("validation_status")
    if status is not None:
        return status != "valid"
    return call.get("purpose") in VERDICT_CALL_PURPOSES and record.get("status") != "success"


def recorded_tokens(record: PredictionRecord) -> int:
    total = 0
    for call in llm_calls(record):
        generation = call.get("generation") or {}
        total += int(generation.get("prompt_tokens") or 0)
        total += int(generation.get("completion_tokens") or 0)
    return total


def uses_verifier(record: PredictionRecord) -> bool:
    return any(call.get("purpose") == VERIFIER_CALL_PURPOSE for call in llm_calls(record))


def first_ranking(record: PredictionRecord) -> list[str]:
    """Ranked document IDs of the first retrieval call (the claim itself as the query)."""

    if "retrieved_documents" in record:
        ranked = sorted(record["retrieved_documents"], key=lambda item: item["rank"])
        return [canonical_document_id(item["doc_id"]) for item in ranked]
    for step in record.get("trace") or ():
        if step.get("type") != "retrieval":
            continue
        if "retrieved_doc_ids" in step:
            return [canonical_document_id(doc_id) for doc_id in step["retrieved_doc_ids"]]
        ranked = sorted(step["retrieved"], key=lambda item: item["rank"])
        return [canonical_document_id(item["doc_id"]) for item in ranked]
    return []


def evidence_pool(record: PredictionRecord) -> list[str]:
    """Document IDs that reached the final verdict call (or the Verifier, for ARGUS)."""

    if "retrieved_documents" in record:
        return first_ranking(record)
    if "evidence_doc_ids" in record:
        return [canonical_document_id(doc_id) for doc_id in record["evidence_doc_ids"]]
    return [canonical_document_id(item["doc_id"]) for item in record.get("evidence") or ()]


def _gold_document_labels(claim: Claim) -> dict[str, frozenset[str]]:
    if claim.evidence is None:
        raise RunAnalysisError(f"claim {claim.claim_id!r} has no gold labels")
    return {
        canonical_document_id(doc_id): frozenset(annotation.label for annotation in annotations)
        for doc_id, annotations in claim.evidence.items()
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_trajectory(records: Sequence[PredictionRecord]) -> dict[str, Any]:
    """Agent steps (LLM calls plus retrieval calls), call failures and stop reasons."""

    llm_counts = [len(llm_calls(record)) for record in records]
    retrieval_counts = [retrieval_call_count(record) for record in records]
    steps = [llm + retrieval for llm, retrieval in zip(llm_counts, retrieval_counts, strict=True)]
    calls_by_purpose: Counter[str] = Counter()
    failures_by_purpose: Counter[str] = Counter()
    for record in records:
        for call in llm_calls(record):
            purpose = str(call.get("purpose"))
            calls_by_purpose[purpose] += 1
            failures_by_purpose[purpose] += int(_call_failed(call, record))
    total_calls = sum(calls_by_purpose.values())
    total_failures = sum(failures_by_purpose.values())
    stop_reasons = Counter(
        str(record["stop_reason"]) for record in records if record.get("stop_reason") is not None
    )
    budget_flags = [
        bool(record["budget_exhausted"]) for record in records if "budget_exhausted" in record
    ]
    return {
        "claims": len(records),
        "mean_llm_calls": _mean(llm_counts),
        "mean_retrieval_calls": _mean(retrieval_counts),
        "mean_steps": _mean(steps),
        "max_steps": max(steps, default=0),
        "llm_calls": total_calls,
        "llm_call_failures": total_failures,
        "llm_call_success_rate": (total_calls - total_failures) / total_calls
        if total_calls
        else None,
        "llm_calls_by_purpose": dict(sorted(calls_by_purpose.items())),
        "llm_call_failures_by_purpose": dict(sorted(failures_by_purpose.items())),
        "valid_output_rate": _mean(
            [float(record.get("status") == "success") for record in records]
        ),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "budget_exhausted_claims": sum(budget_flags) if budget_flags else None,
    }


def summarize_cost(records: Sequence[PredictionRecord]) -> dict[str, Any]:
    """Totals and per-correct-claim cost; failed outputs count as cost but never as correct."""

    correct = sum(is_correct(record) for record in records)
    seconds = sum(float(record.get("end_to_end_latency_seconds") or 0.0) for record in records)
    tokens = sum(recorded_tokens(record) for record in records)
    calls = sum(len(llm_calls(record)) for record in records)

    def per_correct(total: float) -> float | None:
        return total / correct if correct else None

    return {
        "claims": len(records),
        "correct": correct,
        "total_seconds": seconds,
        "total_tokens": tokens,
        "total_llm_calls": calls,
        "seconds_per_correct": per_correct(seconds),
        "tokens_per_correct": per_correct(tokens),
        "llm_calls_per_correct": per_correct(calls),
    }


def summarize_ranking(
    records: Sequence[PredictionRecord], claims_by_id: Mapping[str, Claim], k: int
) -> dict[str, Any]:
    """MRR@k, nDCG@k, Recall@k and Hit@k of the first retrieval, plus evidence-pool coverage.

    Only claims with annotated gold documents are scored; no-evidence claims are counted
    separately, matching the existing Recall@k/Hit@k denominator.
    """

    reciprocal: list[float] = []
    ndcg: list[float] = []
    recall: list[float] = []
    hit: list[float] = []
    pool_recall: list[float] = []
    pool_hit: list[float] = []
    excluded = 0
    for record in records:
        gold = frozenset(_gold_document_labels(_claim_for(record, claims_by_id)))
        if not gold:
            excluded += 1
            continue
        ranked = first_ranking(record)[:k]
        reciprocal.append(reciprocal_rank(ranked, gold))
        ndcg.append(ndcg_at_k(ranked, gold, k))
        found = gold.intersection(ranked)
        recall.append(len(found) / len(gold))
        hit.append(float(bool(found)))
        pooled = gold.intersection(evidence_pool(record))
        pool_recall.append(len(pooled) / len(gold))
        pool_hit.append(float(bool(pooled)))
    return {
        "k": k,
        "eligible_claims": len(reciprocal),
        "excluded_no_evidence_claims": excluded,
        f"mrr_at_{k}": _mean(reciprocal),
        f"ndcg_at_{k}": _mean(ndcg),
        f"recall_at_{k}": _mean(recall),
        f"hit_at_{k}": _mean(hit),
        "evidence_pool_recall": _mean(pool_recall),
        "evidence_pool_hit": _mean(pool_hit),
    }


def _claim_for(record: PredictionRecord, claims_by_id: Mapping[str, Claim]) -> Claim:
    claim_id = str(record["claim_id"])
    try:
        return claims_by_id[claim_id]
    except KeyError as error:
        raise RunAnalysisError(f"prediction for unknown claim {claim_id!r}") from error


def attribute_failure(record: PredictionRecord, claim: Claim) -> str | None:
    """Assign one wrong claim to the earliest pipeline stage that explains the error.

    Returns ``None`` for a correct claim. Order: invalid output; gold evidence never retrieved;
    the Verifier labelled some abstract with the wrong final verdict and no gold abstract
    correctly (``verifier_wrong_evidence``) or also some gold abstract correctly, so the Judge
    picked the wrong side (``conflicting_evidence``); the Verifier labelled no gold abstract
    correctly (``verifier_miss``); otherwise the final verdict call (the Judge in ARGUS, the
    single generator in the baselines). This is a heuristic attribution, not a causal proof.
    """

    if is_correct(record):
        return None
    if record.get("status") != "success":
        return "invalid_output"
    gold_labels = _gold_document_labels(claim)
    prediction = record.get("prediction")
    if gold_labels and not set(gold_labels).intersection(evidence_pool(record)):
        return "retrieval_miss"
    if uses_verifier(record):
        verified = {
            canonical_document_id(item["doc_id"]): item["label"]
            for item in record.get("verified_evidence") or ()
        }
        wrong_evidence = prediction in EVIDENCE_LABELS and any(
            label == prediction and prediction not in gold_labels.get(doc_id, frozenset())
            for doc_id, label in verified.items()
        )
        correct_evidence = any(
            verified.get(doc_id) in labels for doc_id, labels in gold_labels.items()
        )
        if wrong_evidence:
            return "conflicting_evidence" if correct_evidence else "verifier_wrong_evidence"
        if gold_labels and not correct_evidence:
            return "verifier_miss"
    return "verdict"


def summarize_failures(
    records: Sequence[PredictionRecord], claims_by_id: Mapping[str, Claim]
) -> dict[str, Any]:
    """Count wrong claims per attributed stage, overall and per gold verdict."""

    overall: Counter[str] = Counter()
    by_gold: dict[str, Counter[str]] = {}
    for record in records:
        category = attribute_failure(record, _claim_for(record, claims_by_id))
        if category is None:
            continue
        overall[category] += 1
        by_gold.setdefault(str(record["gold_label"]), Counter())[category] += 1
    return {
        "wrong_claims": sum(overall.values()),
        "by_category": {name: overall[name] for name in FAILURE_CATEGORIES},
        "by_gold_label": {
            label: {name: counts[name] for name in FAILURE_CATEGORIES}
            for label, counts in sorted(by_gold.items())
        },
    }


def analyze_run(
    records: Sequence[PredictionRecord], claims_by_id: Mapping[str, Claim], *, k: int = 5
) -> dict[str, Any]:
    """Combine the four offline analyses for one saved run."""

    if not records:
        raise RunAnalysisError("a run must contain at least one prediction record")
    return {
        "trajectory": summarize_trajectory(records),
        "cost": summarize_cost(records),
        "retrieval": summarize_ranking(records, claims_by_id, k),
        "failures": summarize_failures(records, claims_by_id),
    }
