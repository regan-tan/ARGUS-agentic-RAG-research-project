"""Offline trajectory, cost, ranking and failure-attribution analysis of saved predictions."""

from __future__ import annotations

import math
from typing import Any

import pytest

from argus.data.models import Claim, EvidenceAnnotation
from argus.evaluation.retrieval import ndcg_at_k, reciprocal_rank
from argus.evaluation.run_analysis import (
    RunAnalysisError,
    analyze_run,
    attribute_failure,
    summarize_cost,
    summarize_ranking,
    summarize_trajectory,
)

SUPPORTED = Claim(1, "supported claim", (10,), {"10": (EvidenceAnnotation("SUPPORT", (0,)),)})
NO_EVIDENCE = Claim(2, "unverifiable claim", (), {})
CLAIMS = {"1": SUPPORTED, "2": NO_EVIDENCE}


def generation(prompt: int = 100, completion: int = 10) -> dict[str, int]:
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def rag_record(
    claim_id: int = 1, prediction: str = "SUPPORT", gold: str = "SUPPORT", **overrides: Any
) -> dict[str, Any]:
    record = {
        "claim_id": claim_id,
        "gold_label": gold,
        "prediction": prediction,
        "status": "success",
        "model_call_attempted": True,
        "retrieval_call_attempted": True,
        "generation": generation(),
        "end_to_end_latency_seconds": 2.0,
        "retrieved_documents": [
            {"doc_id": 30, "rank": 2},
            {"doc_id": 10, "rank": 1},
        ],
    }
    record.update(overrides)
    return record


def argus_record(
    verified: dict[int, str],
    *,
    prediction: str,
    gold: str = "SUPPORT",
    claim_id: int = 1,
    evidence: tuple[int, ...] = (10, 30),
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = [
        {"type": "retrieval", "agent": "support", "retrieved_doc_ids": list(evidence)},
    ]
    trace += [
        {
            "type": "llm_call",
            "purpose": "evidence_verifier",
            "validation_status": "valid",
            "generation": generation(),
        }
        for _ in verified
    ]
    trace.append({"type": "llm_call", "purpose": "judge", "generation": generation()})
    return {
        "claim_id": claim_id,
        "gold_label": gold,
        "prediction": prediction,
        "status": "success",
        "counts": {"llm_calls": len(verified) + 1, "retrieval_calls": 1},
        "end_to_end_latency_seconds": 5.0,
        "evidence": [{"doc_id": doc_id} for doc_id in evidence],
        "verified_evidence": [
            {"doc_id": doc_id, "label": label} for doc_id, label in verified.items()
        ],
        "trace": trace,
    }


def test_reciprocal_rank_and_ndcg_use_first_gold_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert ndcg_at_k(["g", "x"], {"g"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "g"], {"g"}, 2) == pytest.approx(1 / math.log2(3))
    # Two gold documents, one retrieved at rank 1: DCG = 1, ideal = 1 + 1/log2(3).
    assert ndcg_at_k(["g1", "x"], {"g1", "g2"}, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["x"], {"g"}, 1) == 0.0
    with pytest.raises(ValueError):
        ndcg_at_k(["g"], {"g"}, 0)
    with pytest.raises(ValueError):
        ndcg_at_k(["g"], set(), 1)


def test_ranking_scores_evidence_bearing_claims_in_rank_order() -> None:
    records = [rag_record(), rag_record(claim_id=2, prediction="INSUFFICIENT", gold="INSUFFICIENT")]
    summary = summarize_ranking(records, CLAIMS, k=1)
    assert summary["eligible_claims"] == 1
    assert summary["excluded_no_evidence_claims"] == 1
    # retrieved_documents are reordered by rank, so doc 10 (gold) is rank 1.
    assert summary["mrr_at_1"] == 1.0
    assert summary["recall_at_1"] == 1.0
    assert summary["evidence_pool_hit"] == 1.0


def test_trajectory_counts_steps_failures_and_stop_reasons() -> None:
    failed = rag_record(status="invalid_json", prediction=None)
    iterative = {
        "claim_id": 1,
        "gold_label": "SUPPORT",
        "prediction": "SUPPORT",
        "status": "success",
        "counts": {"llm_calls": 2, "retrieval_calls": 2},
        "stop_reason": "max_retrieval_rounds",
        "budget_exhausted": True,
        "evidence_doc_ids": [10],
        "trace": [
            {"type": "retrieval", "retrieved": [{"doc_id": 10, "rank": 1}]},
            {"type": "llm_call", "purpose": "sufficiency", "error": "bad json"},
            {"type": "llm_call", "purpose": "verdict", "generation": generation()},
        ],
    }
    standard = summarize_trajectory([rag_record(), failed])
    assert standard["mean_steps"] == 2.0
    assert standard["llm_call_failures"] == 1
    assert standard["llm_call_success_rate"] == 0.5
    assert standard["budget_exhausted_claims"] is None

    looped = summarize_trajectory([iterative])
    assert looped["mean_steps"] == 4.0
    assert looped["llm_call_failures_by_purpose"] == {"sufficiency": 1, "verdict": 0}
    assert looped["stop_reasons"] == {"max_retrieval_rounds": 1}
    assert looped["budget_exhausted_claims"] == 1


def test_cost_per_correct_counts_failed_outputs_as_cost_only() -> None:
    records = [rag_record(), rag_record(status="invalid_json", prediction="SUPPORT")]
    cost = summarize_cost(records)
    assert cost["correct"] == 1
    assert cost["total_tokens"] == 220
    assert cost["seconds_per_correct"] == 4.0
    assert cost["llm_calls_per_correct"] == 2.0
    assert summarize_cost([rag_record(prediction="CONTRADICT")])["tokens_per_correct"] is None


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (argus_record({10: "SUPPORT"}, prediction="SUPPORT"), None),
        (rag_record(status="invalid_response", prediction="SUPPORT"), "invalid_output"),
        (
            argus_record({30: "INSUFFICIENT"}, prediction="INSUFFICIENT", evidence=(30,)),
            "retrieval_miss",
        ),
        (
            argus_record({10: "INSUFFICIENT", 30: "CONTRADICT"}, prediction="CONTRADICT"),
            "verifier_wrong_evidence",
        ),
        (
            argus_record({10: "SUPPORT", 30: "CONTRADICT"}, prediction="CONTRADICT"),
            "conflicting_evidence",
        ),
        (
            argus_record({10: "INSUFFICIENT", 30: "INSUFFICIENT"}, prediction="INSUFFICIENT"),
            "verifier_miss",
        ),
        (argus_record({10: "SUPPORT", 30: "INSUFFICIENT"}, prediction="INSUFFICIENT"), "verdict"),
        (
            argus_record({10: "SUPPORT"}, prediction="SUPPORT", gold="INSUFFICIENT", claim_id=2),
            "verifier_wrong_evidence",
        ),
        (rag_record(prediction="CONTRADICT"), "verdict"),
    ],
)
def test_failure_attribution_picks_earliest_explaining_stage(
    record: dict[str, Any], expected: str | None
) -> None:
    assert attribute_failure(record, CLAIMS[str(record["claim_id"])]) == expected


def test_analyze_run_rejects_empty_runs_and_unknown_claims() -> None:
    with pytest.raises(RunAnalysisError):
        analyze_run([], CLAIMS)
    with pytest.raises(RunAnalysisError):
        analyze_run([rag_record(claim_id=99)], CLAIMS)
    result = analyze_run([rag_record(), rag_record(prediction="CONTRADICT")], CLAIMS, k=2)
    assert result["failures"]["wrong_claims"] == 1
    assert result["failures"]["by_gold_label"]["SUPPORT"]["verdict"] == 1
