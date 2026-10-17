"""Failure-aware claim-verification and citation metrics for Standard RAG."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from argus.data.models import Claim, EvidenceLabel
from argus.rag.standard import StandardRAGResult, Verdict
from argus.retrieval.types import canonical_document_id

VERDICTS: tuple[Verdict, ...] = ("SUPPORT", "CONTRADICT", "INSUFFICIENT")
PredictionColumn = Literal["SUPPORT", "CONTRADICT", "INSUFFICIENT", "FAILURE"]


@dataclass(frozen=True)
class GoldVerdict:
    """A label derivation that keeps SciFact mixed annotations explicit."""

    verdict: Verdict | None
    policy: str


def derive_gold_verdict(claim: Claim) -> GoldVerdict:
    """Derive a claim label without silently collapsing conflicting annotations."""

    if claim.evidence is None:
        raise ValueError("Claim verification evaluation requires labeled SciFact claims")
    if not claim.evidence:
        return GoldVerdict("INSUFFICIENT", "no_annotated_evidence")
    labels: set[EvidenceLabel] = {
        annotation.label for annotations in claim.evidence.values() for annotation in annotations
    }
    if labels == {"SUPPORT"}:
        return GoldVerdict("SUPPORT", "support_only")
    if labels == {"CONTRADICT"}:
        return GoldVerdict("CONTRADICT", "contradict_only")
    return GoldVerdict(None, "mixed_evidence_excluded_from_three_class_metrics")


def select_internal_validation_claims(
    claims: Sequence[Claim], *, seed: int = 42
) -> tuple[Claim, ...]:
    """Return the existing Milestone 2 deterministic validation subset by stable ID."""

    from .retrieval import make_internal_split

    split = make_internal_split(claims, seed=seed)
    selected = {str(claim_id) for claim_id in split.validation_claim_ids}
    return tuple(
        sorted(
            (claim for claim in claims if str(claim.claim_id) in selected),
            key=lambda c: str(c.claim_id),
        )
    )


def assert_expected_internal_validation(claims: Sequence[Claim]) -> None:
    """Guard the documented 162/101/61 split against accidental dataset/split drift."""

    evidence_bearing = sum(1 for claim in claims if claim.evidence)
    no_evidence = sum(1 for claim in claims if claim.evidence == {})
    if (len(claims), evidence_bearing, no_evidence) != (162, 101, 61):
        raise ValueError(
            "Expected the Milestone 2 seed-42 validation subset to contain "
            f"162 claims (101 evidence-bearing, 61 no-evidence), got "
            f"{len(claims)}, {evidence_bearing}, {no_evidence}"
        )


def select_smoke_claims(claims: Sequence[Claim], *, limit: int = 5) -> tuple[Claim, ...]:
    """Select a deterministic, label-stratified smoke sample without using predictions."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    ordered = tuple(sorted(claims, key=lambda claim: str(claim.claim_id)))
    chosen: list[Claim] = []
    for verdict in VERDICTS:
        match = next(
            (claim for claim in ordered if derive_gold_verdict(claim).verdict == verdict), None
        )
        if match is not None:
            chosen.append(match)
    for claim in ordered:
        if len(chosen) >= limit:
            break
        if claim not in chosen:
            chosen.append(claim)
    return tuple(chosen[:limit])


def _gold_sentence_pairs(claim: Claim) -> set[tuple[str, int]]:
    if claim.evidence is None:
        raise ValueError("Claim verification evaluation requires labeled SciFact claims")
    return {
        (str(doc_id), sentence)
        for doc_id, annotations in claim.evidence.items()
        for annotation in annotations
        for sentence in annotation.sentences
    }


def _percentage(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _latency_summary(samples: Sequence[float]) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "mean_ms": None, "median_ms": None}
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples) * 1_000,
        "median_ms": statistics.median(samples) * 1_000,
    }


def evaluate_verification_attempts(
    claims: Sequence[Claim], attempts: Mapping[str, StandardRAGResult]
) -> dict[str, object]:
    """Score every attempted claim, counting invalid/missing outcomes as failures.

    Mixed-label claims are retained in artifacts and failure counts, but excluded from the
    three-class benchmark because no unambiguous official claim label can be derived.
    """

    confusion: dict[str, dict[PredictionColumn, int]] = {
        actual: {predicted: 0 for predicted in (*VERDICTS, "FAILURE")} for actual in VERDICTS
    }
    per_class_counts: dict[Verdict, Counter[str]] = {verdict: Counter() for verdict in VERDICTS}
    failures: Counter[str] = Counter()
    valid_responses = 0
    valid_citation_references = 0
    cited_docs = matched_docs = cited_sentences = matched_sentences = 0
    gold_docs = gold_sentences = 0
    e2e_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    prompt_tokens = completion_tokens = 0
    prompt_token_measurements = completion_token_measurements = 0
    evaluated = mixed = 0

    for claim in claims:
        attempt = attempts.get(str(claim.claim_id))
        if attempt is None:
            failures["missing_attempt"] += 1
        else:
            e2e_latencies.append(attempt.end_to_end_latency_seconds)
            if attempt.retrieval_latency_seconds is not None:
                retrieval_latencies.append(attempt.retrieval_latency_seconds)
            if attempt.generation is not None:
                generation_latencies.append(attempt.generation.wall_clock_seconds)
                if attempt.generation.prompt_tokens is not None:
                    prompt_tokens += attempt.generation.prompt_tokens
                    prompt_token_measurements += 1
                if attempt.generation.completion_tokens is not None:
                    completion_tokens += attempt.generation.completion_tokens
                    completion_token_measurements += 1
            if attempt.status != "success":
                failures[attempt.status] += 1

        gold = derive_gold_verdict(claim)
        if gold.verdict is None:
            mixed += 1
            continue
        evaluated += 1
        prediction = attempt.prediction if attempt and attempt.status == "success" else None
        column: PredictionColumn = prediction if prediction is not None else "FAILURE"
        confusion[gold.verdict][column] += 1
        for verdict in VERDICTS:
            if gold.verdict == verdict and prediction == verdict:
                per_class_counts[verdict]["tp"] += 1
            elif gold.verdict != verdict and prediction == verdict:
                per_class_counts[verdict]["fp"] += 1
            elif gold.verdict == verdict and prediction != verdict:
                per_class_counts[verdict]["fn"] += 1

        if attempt and attempt.status == "success" and attempt.response is not None:
            valid_responses += 1
            references = attempt.response.evidence
            valid_citation_references += len(references)
            predicted_docs = {canonical_document_id(reference.doc_id) for reference in references}
            predicted_sentences = {
                (canonical_document_id(reference.doc_id), index)
                for reference in references
                for index in reference.sentence_indices
            }
            gold_document_ids = set(claim.evidence or {})
            gold_pairs = _gold_sentence_pairs(claim)
            cited_docs += len(predicted_docs)
            matched_docs += len(predicted_docs.intersection(gold_document_ids))
            cited_sentences += len(predicted_sentences)
            matched_sentences += len(predicted_sentences.intersection(gold_pairs))
            gold_docs += len(gold_document_ids)
            gold_sentences += len(gold_pairs)

    class_metrics: dict[str, dict[str, float | int]] = {}
    class_f1: list[float] = []
    for verdict in VERDICTS:
        counts = per_class_counts[verdict]
        precision = _percentage(counts["tp"], counts["tp"] + counts["fp"]) or 0.0
        recall = _percentage(counts["tp"], counts["tp"] + counts["fn"]) or 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_f1.append(f1)
        class_metrics[verdict] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[verdict].values()),
        }
    correct = sum(confusion[label][label] for label in VERDICTS)
    return {
        "classification": {
            "attempted_claims": len(claims),
            "evaluated_claims": evaluated,
            "excluded_mixed_gold_claims": mixed,
            "correct_predictions": correct,
            "accuracy": correct / evaluated if evaluated else 0.0,
            "macro_f1": statistics.fmean(class_f1) if class_f1 else 0.0,
            "per_class": class_metrics,
            "confusion_matrix": confusion,
            "failure_policy": (
                "Invalid, failed, and missing predictions remain in denominators as FAILURE."
            ),
            "mixed_label_policy": (
                "Mixed SUPPORT/CONTRADICT gold annotations are excluded from three-class metrics "
                "and reported separately."
            ),
        },
        "citations": {
            "valid_structured_responses": valid_responses,
            "valid_structural_citation_references": valid_citation_references,
            "invalid_or_failed_outputs": len(claims) - valid_responses,
            "cited_documents": cited_docs,
            "gold_document_matches": matched_docs,
            "document_precision": _percentage(matched_docs, cited_docs),
            "document_recall": _percentage(matched_docs, gold_docs),
            "cited_sentence_references": cited_sentences,
            "gold_sentence_matches": matched_sentences,
            "sentence_precision": _percentage(matched_sentences, cited_sentences),
            "sentence_recall": _percentage(matched_sentences, gold_sentences),
            "caveat": (
                "Exact annotated document/sentence overlap is not a semantic-faithfulness judgment."
            ),
        },
        "failures": dict(sorted(failures.items())),
        "efficiency": {
            "retrieval_calls": sum(
                1 for attempt in attempts.values() if attempt.retrieval_call_attempted
            ),
            "model_calls": sum(1 for attempt in attempts.values() if attempt.model_call_attempted),
            "end_to_end_latency": _latency_summary(e2e_latencies),
            "retrieval_latency": _latency_summary(retrieval_latencies),
            "generation_latency": _latency_summary(generation_latencies),
            "prompt_tokens_measured": prompt_tokens,
            "prompt_token_measurements": prompt_token_measurements,
            "completion_tokens_measured": completion_tokens,
            "completion_token_measurements": completion_token_measurements,
        },
    }
