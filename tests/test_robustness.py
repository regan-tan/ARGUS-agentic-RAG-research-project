"""CPU-only, deterministic tests of the retrieval-noise intervention."""

from __future__ import annotations

import pytest

from argus.data.models import CorpusDocument
from argus.evaluation.robustness import NoiseConfig, NoisyRetriever
from argus.retrieval.types import RetrievalResult


def ranking(size: int = 12) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            doc_id=index,
            score=1.0 / index,
            rank=index,
            document=CorpusDocument(index, f"Paper {index}", (f"Sentence {index}",), False),
        )
        for index in range(1, size + 1)
    )


class StaticRetriever:
    def __init__(self, ranked: tuple[RetrievalResult, ...]) -> None:
        self.ranked = ranked
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        self.calls.append((query, k))
        return self.ranked[:k] if query.strip() and k > 0 else ()


def wrapper(
    replacements: int,
    *,
    seed: int = 42,
    protected: tuple[int, ...] = (1, 4),
    size: int = 12,
) -> NoisyRetriever:
    return NoisyRetriever(
        StaticRetriever(ranking(size)),
        claim_id=101,
        config=NoiseConfig(replacements=replacements, seed=seed, candidate_pool=12),
        protected_doc_ids=protected,
    )


def test_clean_condition_is_identical_and_does_not_expand_retrieval() -> None:
    base = StaticRetriever(ranking())
    perturbed = NoisyRetriever(base, claim_id=101, config=NoiseConfig(replacements=0))

    assert perturbed.retrieve("claim", 5) == base.ranked[:5]
    assert base.calls == [("claim", 5)]
    assert perturbed.events[0].replacements == ()
    assert perturbed.events[0].to_dict()["actual_replacements"] == 0


def test_noise_preserves_gold_docs_and_original_sentences() -> None:
    perturbed = wrapper(3)
    results = perturbed.retrieve("claim", 5)
    ids = [result.doc_id for result in results]

    assert len(results) == len(set(ids)) == 5
    assert 1 in ids and 4 in ids
    assert sum(doc_id not in {1, 2, 3, 4, 5} for doc_id in ids) == 3
    assert [result.rank for result in results] == [1, 2, 3, 4, 5]
    assert all(result.document.doc_id == result.doc_id for result in results)
    assert all(result.document.abstract == (f"Sentence {result.doc_id}",) for result in results)
    assert len(perturbed.events[0].replacements) == 3
    assert all(change.inserted_source_rank > 5 for change in perturbed.events[0].replacements)


def test_nested_noise_levels_use_the_same_positions_and_new_documents() -> None:
    one = wrapper(1)
    two = wrapper(2)
    three = wrapper(3)
    one.retrieve("claim", 5)
    two.retrieve("claim", 5)
    three.retrieve("claim", 5)

    assert three.events[0].replacements[:1] == one.events[0].replacements
    assert three.events[0].replacements[:2] == two.events[0].replacements


def test_identical_inputs_produce_reproducible_noise_events() -> None:
    a = wrapper(2, seed=987)
    b = wrapper(2, seed=987)

    assert a.retrieve("same claim", 5) == b.retrieve("same claim", 5)
    assert a.events == b.events
    assert a.events[0].to_dict() == b.events[0].to_dict()


def test_gold_in_candidate_pool_is_not_inserted() -> None:
    perturbed = wrapper(3, protected=(1, 4, 6))
    results = perturbed.retrieve("claim", 5)

    assert 6 not in {result.doc_id for result in results}
    assert 1 in {result.doc_id for result in results}
    assert 4 in {result.doc_id for result in results}


def test_no_false_assumption_of_full_noise_when_pool_is_short() -> None:
    perturbed = wrapper(3, size=6)
    results = perturbed.retrieve("claim", 5)

    assert len(results) == 5
    assert len(perturbed.events[0].replacements) == 1
    assert perturbed.events[0].to_dict()["requested_replacements"] == 3
    assert perturbed.events[0].to_dict()["actual_replacements"] == 1


def test_gold_preservation_requires_explicit_annotation_set() -> None:
    with pytest.raises(ValueError, match="requires explicit annotated document IDs"):
        NoisyRetriever(
            StaticRetriever(ranking()),
            claim_id=1,
            config=NoiseConfig(replacements=1),
        )


def test_empty_annotations_are_valid_for_labeled_no_evidence_claim() -> None:
    perturbed = wrapper(2, protected=())
    results = perturbed.retrieve("claim", 5)

    assert len(perturbed.events[0].replacements) == 2
    assert len(results) == 5


def test_multiple_retrieval_calls_get_separate_trace_events() -> None:
    perturbed = wrapper(1)
    perturbed.retrieve("claim", 5)
    perturbed.retrieve("follow-up query", 5)

    assert [event.call_number for event in perturbed.events] == [1, 2]
    assert [event.query for event in perturbed.events] == ["claim", "follow-up query"]
    assert all(len(event.replacements) == 1 for event in perturbed.events)


def test_blank_or_zero_k_never_injects_noise() -> None:
    perturbed = wrapper(2)

    assert perturbed.retrieve("", 5) == ()
    assert perturbed.retrieve("claim", 0) == ()
    assert all(not event.replacements for event in perturbed.events)


@pytest.mark.parametrize(
    "config",
    [
        {"replacements": -1},
        {"candidate_pool": 0},
    ],
)
def test_invalid_noise_configuration_is_rejected(config: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        NoiseConfig(**config)
