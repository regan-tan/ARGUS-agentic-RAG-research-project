"""Deterministic retrieval-result perturbation for SciFact robustness experiments.

This wrapper changes only the documents a system sees, not the claim, prompt, model,
retrieval backend, document contents, or annotation policy. It is claim-scoped so
iterative and multi-agent retrieval calls are all logged independently.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Collection
from dataclasses import dataclass

from argus.data.models import DocumentId
from argus.retrieval.hybrid import Retriever
from argus.retrieval.types import RetrievalResult, canonical_document_id


@dataclass(frozen=True)
class NoiseConfig:
    """Number of top-k slots to replace with lower-ranked, same-query candidates."""

    replacements: int = 0
    seed: int = 42
    candidate_pool: int = 20
    preserve_annotated_gold: bool = True

    def __post_init__(self) -> None:
        if self.replacements < 0:
            raise ValueError("replacements cannot be negative")
        if self.candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")


@dataclass(frozen=True)
class NoiseReplacement:
    """One actual substitution, preserving the original SciFact source document."""

    slot: int
    removed_doc_id: DocumentId
    inserted_doc_id: DocumentId
    inserted_source_rank: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "slot": self.slot,
            "removed_doc_id": self.removed_doc_id,
            "inserted_doc_id": self.inserted_doc_id,
            "inserted_source_rank": self.inserted_source_rank,
        }


@dataclass(frozen=True)
class NoiseEvent:
    """Audit record for one retrieval call; never passed to an LLM."""

    claim_id: DocumentId
    query: str
    call_number: int
    requested_k: int
    requested_replacements: int
    original_doc_ids: tuple[DocumentId, ...]
    returned_doc_ids: tuple[DocumentId, ...]
    replacements: tuple[NoiseReplacement, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "query": self.query,
            "call_number": self.call_number,
            "requested_k": self.requested_k,
            "requested_replacements": self.requested_replacements,
            "actual_replacements": len(self.replacements),
            "original_doc_ids": list(self.original_doc_ids),
            "returned_doc_ids": list(self.returned_doc_ids),
            "replacements": [item.to_dict() for item in self.replacements],
        }


class NoisyRetriever:
    """Wrap any ARGUS retriever using fixed, per-claim hard-negative substitutions.

    Distractors are selected from ranks k+1 through candidate_pool of the SAME
    retrieval query, keeping their genuine SciFact documents and sentence IDs.
    Being unannotated does not prove a candidate is semantically irrelevant;
    manual review is necessary before calling these verified distractors.

    A protected gold-ID set may be used for a labeled evaluation claim solely
    to avoid deleting already retrieved annotated evidence. It is never added
    to the query, result documents, or LLM prompts.
    """

    def __init__(
        self,
        base: Retriever,
        *,
        claim_id: DocumentId,
        config: NoiseConfig,
        protected_doc_ids: Collection[DocumentId] | None = None,
    ) -> None:
        if config.replacements and config.preserve_annotated_gold and protected_doc_ids is None:
            raise ValueError(
                "gold-preserving perturbation requires explicit annotated document IDs"
            )
        self.base = base
        self.claim_id = claim_id
        self.config = config
        self._protected = {
            canonical_document_id(doc_id) for doc_id in (protected_doc_ids or ())
        } if config.preserve_annotated_gold else set()
        self._events: list[NoiseEvent] = []

    @property
    def events(self) -> tuple[NoiseEvent, ...]:
        return tuple(self._events)

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        call_number = len(self._events) + 1
        if self.config.replacements == 0 or not query.strip() or k <= 0:
            # Zero noise is exactly one original retrieval call with the original k.
            original = tuple(self.base.retrieve(query, k))
            self._record(query, k, original, original, (), call_number)
            return original

        ranking = tuple(self.base.retrieve(query, max(k, self.config.candidate_pool)))
        original = ranking[:k]
        original_ids = {canonical_document_id(item.doc_id) for item in original}
        protected_slots = self._protected
        eligible_slots = [
            position
            for position, item in enumerate(original)
            if canonical_document_id(item.doc_id) not in protected_slots
        ]
        seen = set(original_ids)
        candidates: list[RetrievalResult] = []
        for item in ranking[k:]:
            identifier = canonical_document_id(item.doc_id)
            if identifier in seen or identifier in protected_slots:
                continue
            seen.add(identifier)
            candidates.append(item)

        # Do not incorporate noise level into the seed: level 1 is a subset of
        # levels 2 and 3 whenever adequate candidate slots are available.
        key = json.dumps(
            [self.config.seed, str(self.claim_id), query, call_number, k],
            ensure_ascii=False,
        )
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        rng.shuffle(eligible_slots)
        rng.shuffle(candidates)
        count = min(self.config.replacements, len(eligible_slots), len(candidates))

        returned = list(original)
        replacements: list[NoiseReplacement] = []
        for slot, candidate in zip(
            eligible_slots[:count], candidates[:count], strict=True
        ):
            removed = original[slot]
            returned[slot] = RetrievalResult(
                candidate.doc_id,
                candidate.score,
                slot + 1,
                candidate.document,
            )
            replacements.append(
                NoiseReplacement(slot + 1, removed.doc_id, candidate.doc_id, candidate.rank)
            )

        completed = tuple(returned)
        self._record(query, k, original, completed, tuple(replacements), call_number)
        return completed

    def _record(
        self,
        query: str,
        k: int,
        original: tuple[RetrievalResult, ...],
        returned: tuple[RetrievalResult, ...],
        replacements: tuple[NoiseReplacement, ...],
        call_number: int,
    ) -> None:
        self._events.append(
            NoiseEvent(
                claim_id=self.claim_id,
                query=query,
                call_number=call_number,
                requested_k=k,
                requested_replacements=self.config.replacements,
                original_doc_ids=tuple(item.doc_id for item in original),
                returned_doc_ids=tuple(item.doc_id for item in returned),
                replacements=replacements,
            )
        )
