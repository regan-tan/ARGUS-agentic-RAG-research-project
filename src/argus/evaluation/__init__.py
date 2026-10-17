"""Evaluation helpers for reproducible retrieval and claim verification experiments."""

from .retrieval import (
    InternalSplit,
    MetricSummary,
    evidence_category,
    gold_document_ids,
    make_internal_split,
    summarize_by_category,
    summarize_metrics,
)
from .verification import (
    assert_expected_internal_validation,
    derive_gold_verdict,
    evaluate_verification_attempts,
    select_internal_validation_claims,
    select_smoke_claims,
)

__all__ = [
    "InternalSplit",
    "MetricSummary",
    "assert_expected_internal_validation",
    "derive_gold_verdict",
    "evaluate_verification_attempts",
    "evidence_category",
    "gold_document_ids",
    "make_internal_split",
    "select_internal_validation_claims",
    "select_smoke_claims",
    "summarize_by_category",
    "summarize_metrics",
]
