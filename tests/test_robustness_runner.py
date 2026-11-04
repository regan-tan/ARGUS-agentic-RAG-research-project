"""End-to-end noise integration tests using only injected fake retrievers and model calls."""

from __future__ import annotations

import json

import pytest

from argus.data.models import Claim, CorpusDocument, EvidenceAnnotation
from argus.evaluation.robustness import NoiseConfig
from argus.evaluation.robustness_runner import (
    SYSTEMS,
    build_system,
    evaluate_noisy_attempts,
    run_noisy_claim,
    summarize_noise_events,
)
from argus.llm.ollama import GenerationMetadata, OllamaChatResponse
from argus.retrieval.types import RetrievalResult


class RetrieverStub:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []
        self.ranked = tuple(
            RetrievalResult(
                identifier,
                1.0 / identifier,
                identifier,
                CorpusDocument(
                    identifier, f"Paper {identifier}", (f"Sentence {identifier}.",), False
                ),
            )
            for identifier in range(1, 8)
        )

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        self.queries.append((query, k))
        return self.ranked[:k]


class GeneratorStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def chat(
        self, *, system: str, user: str, response_schema: object
    ) -> OllamaChatResponse:
        self.calls.append((system, user))
        if "evidence-sufficiency assessor" in system:
            response: dict[str, object] = {
                "missing_information": "",
                "reason": "There is direct evidence.",
                "sufficient": True,
            }
        elif "independent evidence verifier" in system:
            response = {
                "label": "SUPPORT",
                "rationale": "The abstract supports the claim.",
                "sentence_indices": [0],
            }
        else:
            response = {
                "verdict": "SUPPORT",
                "rationale": "The first article directly supports the claim.",
                "evidence": [{"doc_id": 1, "sentence_indices": [0]}],
            }
        return OllamaChatResponse(
            json.dumps(response),
            GenerationMetadata(10, 10, None, None, None, None, 0.01),
        )


def example_claim() -> Claim:
    return Claim(
        101,
        "A scientific claim",
        (1,),
        {"1": (EvidenceAnnotation("SUPPORT", (0,)),)},
    )


@pytest.mark.parametrize("system", SYSTEMS)
def test_noise_is_passed_through_every_original_pipeline(system: str) -> None:
    base, generator = RetrieverStub(), GeneratorStub()
    result, events = run_noisy_claim(
        system,
        base,
        generator,
        example_claim(),
        NoiseConfig(replacements=1, candidate_pool=7),
        k=2,
    )

    assert result.status == "success"
    assert result.prediction == "SUPPORT"
    assert len(events) >= 1
    assert all(event["actual_replacements"] == 1 for event in events)
    assert all(1 in event["returned_doc_ids"] for event in events)
    assert all(
        event["original_doc_ids"] != event["returned_doc_ids"] for event in events
    )
    assert all(event["requested_k"] == 2 for event in events)
    assert len(generator.calls) >= 1
    metrics = evaluate_noisy_attempts(system, (example_claim(),), {"101": result})
    assert metrics["classification"]["accuracy"] == 1.0


@pytest.mark.parametrize("system", SYSTEMS)
def test_clean_condition_keeps_exact_original_documents(system: str) -> None:
    base, generator = RetrieverStub(), GeneratorStub()
    result, events = run_noisy_claim(
        system, base, generator, example_claim(), NoiseConfig(replacements=0), k=2
    )

    assert result.status == "success"
    assert all(event["original_doc_ids"] == event["returned_doc_ids"] for event in events)
    assert all(not event["replacements"] for event in events)
    assert all(depth == 2 for _, depth in base.queries)


def test_unlabeled_claim_is_rejected_before_any_retrieval() -> None:
    base, generator = RetrieverStub(), GeneratorStub()
    claim = Claim(101, "No gold labels", (), None)

    with pytest.raises(ValueError, match="labeled claim"):
        run_noisy_claim("standard", base, generator, claim, NoiseConfig(replacements=1))
    assert not base.queries
    assert not generator.calls


def test_noise_metrics_report_actual_dose_and_partial_records() -> None:
    base, generator = RetrieverStub(), GeneratorStub()
    result, events = run_noisy_claim(
        "standard",
        base,
        generator,
        example_claim(),
        NoiseConfig(replacements=1, candidate_pool=7),
        k=2,
    )
    record = result.to_dict()
    record["noise_events"] = list(events)
    summary = summarize_noise_events({"101": record})

    assert summary["retrieval_calls"] == 1
    assert summary["actual_replacements"] == 1
    assert summary["requested_replacements"] == 1
    assert summary["underdelivered_retrieval_calls"] == 0
    assert summary["per_claim"]["101"]["retrieval_calls"] == 1


def test_reject_unknown_system_without_mutating_originals() -> None:
    with pytest.raises(ValueError, match="Unknown robustness system"):
        build_system("unknown", RetrieverStub(), GeneratorStub())
