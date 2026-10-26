from __future__ import annotations

import json
from pathlib import Path

import pytest

from argus.data.models import Claim, CorpusDocument, EvidenceAnnotation
from argus.evaluation.verification import (
    derive_gold_verdict,
    evaluate_verification_attempts,
    select_smoke_claims,
)
from argus.llm.ollama import GenerationMetadata, OllamaChatResponse
from argus.rag.artifacts import (
    ArtifactError,
    append_prediction_record,
    ensure_resume_compatible,
    load_prediction_records,
)
from argus.rag.standard import (
    StandardRAGPipeline,
    ValidationError,
    build_messages,
    validate_response,
)
from argus.retrieval.types import RetrievalResult


def document(
    doc_id: int, sentences: tuple[str, ...] = ("First finding.", "Second finding.")
) -> CorpusDocument:
    return CorpusDocument(doc_id, f"Study {doc_id}", sentences, False)


def result(doc: CorpusDocument, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(doc.doc_id, 1.0, rank, doc)


def metadata() -> GenerationMetadata:
    return GenerationMetadata(10, 5, None, None, None, None, 0.25)


def test_prompt_preserves_original_document_ids_and_sentence_indices() -> None:
    system, user = build_messages("A precise scientific claim.", (result(document(17)),))

    assert "scientific claim verifier" in system
    assert "A precise scientific claim." in user
    assert "Document ID: 17" in user
    assert "Title: Study 17" in user
    assert "Sentence [0]: First finding." in user
    assert "Sentence [1]: Second finding." in user


@pytest.mark.parametrize("verdict", ["SUPPORT", "CONTRADICT"])
def test_valid_supported_or_contradicted_response_requires_real_citation(verdict: str) -> None:
    response = validate_response(
        json.dumps(
            {
                "verdict": verdict,
                "rationale": "A cited sentence directly establishes the relationship.",
                "evidence": [{"doc_id": "17", "sentence_indices": [1]}],
            }
        ),
        (result(document(17)),),
    )

    assert response.verdict == verdict
    assert response.evidence[0].doc_id == 17
    assert response.evidence[0].sentence_indices == (1,)


def test_valid_insufficient_can_have_empty_evidence() -> None:
    response = validate_response(
        '{"verdict":"INSUFFICIENT","rationale":"The evidence is indirect.","evidence":[]}',
        (result(document(17)),),
    )
    assert response.verdict == "INSUFFICIENT"
    assert response.evidence == ()


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not json", "not JSON"),
        ('{"verdict":"MAYBE","rationale":"x","evidence":[]}', "verdict"),
        ('{"verdict":"SUPPORT","rationale":"x","evidence":[]}', "requires"),
        (
            '{"verdict":"SUPPORT","rationale":"x","evidence":[{"doc_id":99,"sentence_indices":[0]}]}',
            "not retrieved",
        ),
        (
            '{"verdict":"SUPPORT","rationale":"x","evidence":[{"doc_id":17,"sentence_indices":[4]}]}',
            "out-of-range",
        ),
        ('{"verdict":"SUPPORT","evidence":[]}', "missing"),
    ],
)
def test_invalid_model_responses_are_rejected(payload: str, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        validate_response(payload, (result(document(17)),))


def test_duplicate_evidence_references_are_merged_deterministically() -> None:
    response = validate_response(
        """{
          "verdict": "SUPPORT", "rationale": "Both sentences matter.",
          "evidence": [
            {"doc_id": 17, "sentence_indices": [0, 0]},
            {"doc_id": "17", "sentence_indices": [1, 0]}
          ]
        }""",
        (result(document(17)),),
    )
    assert response.evidence[0].sentence_indices == (0, 1)


class FakeRetriever:
    def __init__(self, ranking: tuple[RetrievalResult, ...]) -> None:
        self.ranking = ranking
        self.calls = 0

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        self.calls += 1
        return self.ranking[:k]


class FakeGenerator:
    def __init__(self, content: str | Exception) -> None:
        self.content = content
        self.calls = 0
        self.last_user = ""

    def chat(self, *, system: str, user: str, response_schema: object) -> OllamaChatResponse:
        self.calls += 1
        self.last_user = user
        if isinstance(self.content, Exception):
            raise self.content
        return OllamaChatResponse(self.content, metadata())


def test_pipeline_uses_one_retrieval_and_one_generation_without_fallback() -> None:
    retriever = FakeRetriever((result(document(17)),))
    generator = FakeGenerator(
        '{"verdict":"SUPPORT","rationale":"The study says so.",'
        '"evidence":[{"doc_id":17,"sentence_indices":[0]}]}'
    )
    outcome = StandardRAGPipeline(retriever, generator, k=5).verify("Claim", claim_id=3)

    assert retriever.calls == 1
    assert generator.calls == 1
    assert outcome.status == "success"
    assert outcome.prediction == "SUPPORT"
    assert "Document ID: 17" in generator.last_user


def test_pipeline_reports_inference_failure_without_fabricating_insufficient() -> None:
    retriever = FakeRetriever((result(document(17)),))
    outcome = StandardRAGPipeline(retriever, FakeGenerator(RuntimeError("offline"))).verify("Claim")

    assert outcome.status == "inference_failure"
    assert outcome.prediction is None


def test_pipeline_rejects_a_response_when_ollama_reports_context_limit_reached() -> None:
    class ContextLimitGenerator(FakeGenerator):
        def chat(self, *, system: str, user: str, response_schema: object) -> OllamaChatResponse:
            self.calls += 1
            return OllamaChatResponse(
                '{"verdict":"SUPPORT","rationale":"Grounded.",'
                '"evidence":[{"doc_id":17,"sentence_indices":[0]}]}',
                GenerationMetadata(8_192, 5, None, None, None, None, 0.25),
            )

    generator = ContextLimitGenerator("unused")
    outcome = StandardRAGPipeline(
        FakeRetriever((result(document(17)),)),
        generator,
        context_window_tokens=8_192,
    ).verify("Claim")

    assert generator.calls == 1
    assert outcome.status == "context_limit_exceeded"
    assert outcome.prediction is None


def claim(claim_id: int, evidence: dict[str, tuple[EvidenceAnnotation, ...]]) -> Claim:
    return Claim(claim_id, f"claim {claim_id}", (), evidence)


def successful_attempt(claim_id: int, verdict: str) -> object:
    retriever = FakeRetriever((result(document(17)),))
    return StandardRAGPipeline(
        retriever,
        FakeGenerator(
            json.dumps(
                {
                    "verdict": verdict,
                    "rationale": "Grounded response.",
                    "evidence": []
                    if verdict == "INSUFFICIENT"
                    else [{"doc_id": 17, "sentence_indices": [0]}],
                }
            )
        ),
    ).verify(f"claim {claim_id}", claim_id=claim_id)


def test_gold_derivation_smoke_selection_and_failure_aware_metrics() -> None:
    support = claim(1, {"17": (EvidenceAnnotation("SUPPORT", (0,)),)})
    contradict = claim(2, {"17": (EvidenceAnnotation("CONTRADICT", (0,)),)})
    insufficient = claim(3, {})
    mixed = claim(
        4,
        {"17": (EvidenceAnnotation("SUPPORT", (0,)), EvidenceAnnotation("CONTRADICT", (1,)))},
    )
    assert derive_gold_verdict(support).verdict == "SUPPORT"
    assert derive_gold_verdict(insufficient).verdict == "INSUFFICIENT"
    assert derive_gold_verdict(mixed).verdict is None
    assert [
        item.claim_id for item in select_smoke_claims((support, contradict, insufficient), limit=3)
    ] == [
        1,
        2,
        3,
    ]

    metrics = evaluate_verification_attempts(
        (support, contradict, insufficient, mixed),
        {"1": successful_attempt(1, "SUPPORT"), "2": successful_attempt(2, "SUPPORT")},
    )
    classification = metrics["classification"]
    assert classification["evaluated_claims"] == 3
    assert classification["excluded_mixed_gold_claims"] == 1
    assert classification["accuracy"] == pytest.approx(1 / 3)
    assert classification["confusion_matrix"]["CONTRADICT"]["SUPPORT"] == 1
    assert classification["confusion_matrix"]["INSUFFICIENT"]["FAILURE"] == 1
    assert metrics["failures"] == {"missing_attempt": 2}


def test_append_only_checkpoint_supports_resume_without_duplicate_claims(tmp_path: Path) -> None:
    configuration = {
        "schema_version": 1,
        "evaluation": {"seed": 42, "selected_claim_ids": ["1", "2", "3"]},
        "generator": {"model_digest": "expected"},
    }
    metadata = {"resume_configuration": configuration}
    predictions = tmp_path / "predictions.jsonl"
    with predictions.open("a", encoding="utf-8") as handle:
        append_prediction_record(handle, {"claim_id": 1, "status": "success"})
        append_prediction_record(handle, {"claim_id": 2, "status": "inference_failure"})

    completed = load_prediction_records(predictions)
    assert set(completed) == {"1", "2"}
    assert [claim_id for claim_id in ("1", "2", "3") if claim_id not in completed] == ["3"]
    assert completed["2"]["status"] == "inference_failure"
    ensure_resume_compatible(metadata, configuration)

    with predictions.open("a", encoding="utf-8") as handle:
        append_prediction_record(handle, {"claim_id": 3, "status": "success"})
    assert set(load_prediction_records(predictions)) == {"1", "2", "3"}


def test_resume_artifacts_reject_duplicate_claims_and_configuration_mismatch(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"claim_id": 1, "status": "success"}\n{"claim_id": "1", "status": "success"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="duplicate"):
        load_prediction_records(predictions)

    with pytest.raises(ArtifactError, match="does not match"):
        ensure_resume_compatible(
            {"resume_configuration": {"generator": {"model_digest": "old"}}},
            {"generator": {"model_digest": "new"}},
        )


def test_resume_recovers_only_an_unterminated_final_jsonl_fragment(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"claim_id": 1, "status": "success"}\n{"claim_id": 2', encoding="utf-8")
    assert set(load_prediction_records(predictions)) == {"1"}


def test_resume_checkpoint_does_not_repeat_completed_stubbed_inference(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    valid_json = (
        '{"verdict":"SUPPORT","rationale":"Grounded.",'
        '"evidence":[{"doc_id":17,"sentence_indices":[0]}]}'
    )
    initial_generator = FakeGenerator(valid_json)
    initial_pipeline = StandardRAGPipeline(
        FakeRetriever((result(document(17)),)), initial_generator
    )
    with predictions.open("a", encoding="utf-8") as handle:
        for claim_id in (1, 2):
            attempt = initial_pipeline.verify(f"claim {claim_id}", claim_id=claim_id)
            append_prediction_record(handle, attempt.to_dict())
    assert initial_generator.calls == 2

    persisted = load_prediction_records(predictions)
    resumed_generator = FakeGenerator(valid_json)
    resumed_pipeline = StandardRAGPipeline(
        FakeRetriever((result(document(17)),)), resumed_generator
    )
    with predictions.open("a", encoding="utf-8") as handle:
        for claim_id in (1, 2, 3):
            if str(claim_id) in persisted:
                continue
            attempt = resumed_pipeline.verify(f"claim {claim_id}", claim_id=claim_id)
            append_prediction_record(handle, attempt.to_dict())

    assert resumed_generator.calls == 1
    assert set(load_prediction_records(predictions)) == {"1", "2", "3"}
