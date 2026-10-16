"""Deterministic internal splitting and evidence-document retrieval metrics."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from argus.data.models import Claim, EvidenceLabel
from argus.retrieval.types import RetrievalResult, canonical_document_id


@dataclass(frozen=True)
class InternalSplit:
    """Stable IDs for development and validation subsets derived from train claims."""

    seed: int
    validation_fraction: float
    development_claim_ids: tuple[int | str, ...]
    validation_claim_ids: tuple[int | str, ...]


@dataclass(frozen=True)
class MetricSummary:
    recall_at_k: dict[int, float]
    hit_at_k: dict[int, float]
    eligible_claims: int
    excluded_no_evidence_claims: int


def evidence_category(claim: Claim) -> str:
    """Classify labeled training claims without collapsing mixed evidence labels."""

    if claim.evidence is None:
        raise ValueError("Evidence categories require labeled claims")
    if not claim.evidence:
        return "no_evidence"
    labels: set[EvidenceLabel] = {
        annotation.label for annotations in claim.evidence.values() for annotation in annotations
    }
    if labels == {"SUPPORT"}:
        return "support"
    if labels == {"CONTRADICT"}:
        return "contradict"
    return "mixed"


def _claim_sort_key(claim: Claim) -> str:
    return str(claim.claim_id)


def make_internal_split(
    claims: Sequence[Claim], *, seed: int = 42, validation_fraction: float = 0.2
) -> InternalSplit:
    """Create a seed-42 reproducible, approximately stratified train validation split."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        groups[evidence_category(claim)].append(claim)
    target_size = round(len(claims) * validation_fraction)
    ideal_sizes = {category: len(group) * validation_fraction for category, group in groups.items()}
    selected_counts = {category: int(size) for category, size in ideal_sizes.items()}
    remaining = target_size - sum(selected_counts.values())
    for category in sorted(groups, key=lambda key: (-(ideal_sizes[key] % 1), key))[:remaining]:
        selected_counts[category] += 1

    rng = random.Random(seed)
    validation_ids: set[str] = set()
    for category in sorted(groups):
        candidates = sorted(groups[category], key=_claim_sort_key)
        rng.shuffle(candidates)
        validation_ids.update(
            str(claim.claim_id) for claim in candidates[: selected_counts[category]]
        )
    ordered_claims = sorted(claims, key=_claim_sort_key)
    return InternalSplit(
        seed=seed,
        validation_fraction=validation_fraction,
        development_claim_ids=tuple(
            claim.claim_id for claim in ordered_claims if str(claim.claim_id) not in validation_ids
        ),
        validation_claim_ids=tuple(
            claim.claim_id for claim in ordered_claims if str(claim.claim_id) in validation_ids
        ),
    )


def gold_document_ids(claim: Claim) -> frozenset[str]:
    """Return unique annotated document IDs, excluding no-evidence claims."""

    if claim.evidence is None:
        raise ValueError("Gold evidence is unavailable for unlabeled claims")
    return frozenset(claim.evidence)


def summarize_metrics(
    claims: Iterable[Claim],
    predictions: Mapping[str, Sequence[RetrievalResult]],
    k_values: Sequence[int],
) -> MetricSummary:
    """Compute macro document Recall@k and Hit@k over evidence-bearing claims only."""

    ordered_k = tuple(sorted(set(k_values)))
    if not ordered_k or any(k <= 0 for k in ordered_k):
        raise ValueError("k_values must contain positive integers")
    recalls: dict[int, list[float]] = {k: [] for k in ordered_k}
    hits: dict[int, list[float]] = {k: [] for k in ordered_k}
    excluded = 0
    for claim in claims:
        gold_ids = gold_document_ids(claim)
        if not gold_ids:
            excluded += 1
            continue
        retrieved = [
            canonical_document_id(result.doc_id)
            for result in predictions.get(str(claim.claim_id), ())
        ]
        for k in ordered_k:
            found = gold_ids.intersection(retrieved[:k])
            recalls[k].append(len(found) / len(gold_ids))
            hits[k].append(float(bool(found)))
    eligible = len(next(iter(recalls.values())))
    return MetricSummary(
        recall_at_k={
            k: sum(values) / eligible if eligible else 0.0 for k, values in recalls.items()
        },
        hit_at_k={k: sum(values) / eligible if eligible else 0.0 for k, values in hits.items()},
        eligible_claims=eligible,
        excluded_no_evidence_claims=excluded,
    )


def summarize_by_category(
    claims: Iterable[Claim],
    predictions: Mapping[str, Sequence[RetrievalResult]],
    k_values: Sequence[int],
) -> dict[str, MetricSummary]:
    """Break metrics out by evidence label category, keeping mixed claims explicit."""

    grouped: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        grouped[evidence_category(claim)].append(claim)
    return {
        category: summarize_metrics(group, predictions, k_values)
        for category, group in sorted(grouped.items())
        if category != "no_evidence"
    }
