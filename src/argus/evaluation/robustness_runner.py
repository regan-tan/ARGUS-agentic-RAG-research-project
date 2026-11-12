"""Shared entrypoints and scoring for controlled retrieval-noise experiments.

The original Standard, Iterative, and ARGUS pipelines remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from argus.agents.argus import ArgusAgent, ArgusConfig, ArgusResult, evaluate_argus_attempts
from argus.agents.iterative import (
    IterativeRAGAgent,
    IterativeRAGConfig,
    IterativeRAGResult,
    evaluate_iterative_attempts,
)
from argus.data.models import Claim
from argus.evaluation.robustness import NoiseConfig, NoisyRetriever
from argus.evaluation.verification import evaluate_verification_attempts
from argus.rag.standard import (
    StandardRAGPipeline,
    StandardRAGResult,
    StructuredGenerator,
)
from argus.retrieval.hybrid import Retriever

SystemName = Literal["standard", "iterative", "argus", "argus-no-challenge"]
SYSTEMS: tuple[SystemName, ...] = (
    "standard",
    "iterative",
    "argus",
    "argus-no-challenge",
)
Attempt = StandardRAGResult | IterativeRAGResult | ArgusResult


def build_system(
    system: SystemName,
    retriever: Retriever,
    generator: StructuredGenerator,
    *,
    k: int = 5,
    num_ctx: int = 8192,
) -> StandardRAGPipeline | IterativeRAGAgent | ArgusAgent:
    """Build existing frozen pipelines with their original Experiment 1–3 budgets."""

    if k <= 0 or num_ctx <= 0:
        raise ValueError("k and num_ctx must be positive")
    if system == "standard":
        return StandardRAGPipeline(retriever, generator, k=k, context_window_tokens=num_ctx)
    if system == "iterative":
        return IterativeRAGAgent(
            retriever,
            generator,
            IterativeRAGConfig(
                initial_k=k,
                followup_k=k,
                max_retrieval_rounds=3,
                max_llm_calls=6,
                max_evidence_documents=max(15, k),
                context_window_tokens=num_ctx,
            ),
        )
    if system in {"argus", "argus-no-challenge"}:
        return ArgusAgent(
            retriever,
            generator,
            ArgusConfig(
                retrieval_k=k,
                max_evidence_documents=max(10, k),
                max_llm_calls=12,
                challenge_enabled=system == "argus",
                context_window_tokens=num_ctx,
            ),
        )
    raise ValueError(f"Unknown robustness system {system!r}")


def run_noisy_claim(
    system: SystemName,
    base_retriever: Retriever,
    generator: StructuredGenerator,
    claim: Claim,
    noise: NoiseConfig,
    *,
    k: int = 5,
    num_ctx: int = 8192,
) -> tuple[Attempt, tuple[dict[str, object], ...]]:
    """Run one labeled claim; gold is used only to protect original evidence slots."""

    if claim.evidence is None:
        raise ValueError("Robustness requires a labeled claim (not the official test set)")
    noisy = NoisyRetriever(
        base_retriever,
        claim_id=claim.claim_id,
        config=noise,
        protected_doc_ids=tuple(claim.evidence),
    )
    pipeline = build_system(system, noisy, generator, k=k, num_ctx=num_ctx)
    attempt = pipeline.verify(claim.claim, claim_id=claim.claim_id)
    return attempt, tuple(event.to_dict() for event in noisy.events)


def evaluate_noisy_attempts(
    system: SystemName, claims: Sequence[Claim], attempts: Mapping[str, Attempt]
) -> dict[str, object]:
    """Use each original system's metrics rather than introducing new scoring rules."""

    if system == "standard":
        return evaluate_verification_attempts(
            claims,
            {key: value for key, value in attempts.items() if isinstance(value, StandardRAGResult)},
        )
    if system == "iterative":
        return evaluate_iterative_attempts(
            claims,
            {
                key: value
                for key, value in attempts.items()
                if isinstance(value, IterativeRAGResult)
            },
        )
    if system in {"argus", "argus-no-challenge"}:
        return evaluate_argus_attempts(
            claims,
            {key: value for key, value in attempts.items() if isinstance(value, ArgusResult)},
        )
    raise ValueError(f"Unknown robustness system {system!r}")


def summarize_noise_events(records: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    """Expose actual noise doses so nominal levels are not mistaken for delivered ones."""

    event_count = actual = requested = underdelivered = 0
    injected_ids: set[str] = set()
    per_claim: dict[str, dict[str, int]] = {}
    for claim_id, record in records.items():
        events = record.get("noise_events")
        if not isinstance(events, list):
            raise ValueError(f"Claim {claim_id!r} has no noise_events array")
        claim_actual = claim_requested = 0
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("Noise events must be JSON objects")
            replacements = event.get("replacements")
            if not isinstance(replacements, list):
                raise ValueError("Each noise event must contain replacements")
            dose = event.get("requested_replacements")
            if not isinstance(dose, int):
                raise ValueError("Noise event lacks requested_replacements")
            event_count += 1
            requested += dose
            actual += len(replacements)
            claim_requested += dose
            claim_actual += len(replacements)
            underdelivered += len(replacements) < dose
            for change in replacements:
                if not isinstance(change, dict) or "inserted_doc_id" not in change:
                    raise ValueError("Noise replacement is missing its source document ID")
                injected_ids.add(str(change["inserted_doc_id"]))
        per_claim[claim_id] = {
            "requested_replacements": claim_requested,
            "actual_replacements": claim_actual,
            "retrieval_calls": len(events),
        }
    return {
        "retrieval_calls": event_count,
        "requested_replacements": requested,
        "actual_replacements": actual,
        "underdelivered_retrieval_calls": underdelivered,
        "distinct_injected_document_ids": len(injected_ids),
        "per_claim": per_claim,
    }
