from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType

import pytest

from argus.agents.iterative import (
    IterativeRAGAgent,
    IterativeRAGConfig,
    IterativeRAGResult,
    evaluate_iterative_attempts,
    prompt_fingerprints,
    rewrite_json_schema,
    sufficiency_json_schema,
)
from argus.data.models import Claim, CorpusDocument, EvidenceAnnotation
from argus.evaluation.verification import select_internal_validation_claims, select_smoke_claims
from argus.llm.ollama import GenerationMetadata, OllamaChatResponse
from argus.rag.standard import build_messages
from argus.retrieval.types import RetrievalResult

CLAIM = "Drug X lowers blood pressure in adults."


def document(doc_id: int | str) -> CorpusDocument:
    return CorpusDocument(doc_id, f"Study {doc_id}", ("First finding.", "Second finding."), False)


def ranking(*doc_ids: int | str) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(doc_id, 1.0 - rank / 100, rank, document(doc_id))
        for rank, doc_id in enumerate(doc_ids, start=1)
    )


class FakeRetriever:
    """Returns a preset ranking per query and records every call."""

    def __init__(self, rankings: dict[str, tuple[RetrievalResult, ...] | Exception]) -> None:
        self.rankings = rankings
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        self.calls.append((query, k))
        outcome = self.rankings.get(query, ())
        if isinstance(outcome, Exception):
            raise outcome
        return outcome[:k]


class FakeGenerator:
    """Returns scripted model replies in order and records every prompt."""

    def __init__(self, replies: list[str | Exception | OllamaChatResponse]) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def chat(self, *, system: str, user: str, response_schema: object) -> OllamaChatResponse:
        self.prompts.append((system, user))
        if not self.replies:
            raise AssertionError("The agent made more model calls than the test scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, OllamaChatResponse):
            return reply
        return OllamaChatResponse(reply, GenerationMetadata(10, 5, None, None, None, None, 0.25))


def sufficient(value: bool) -> str:
    return json.dumps(
        {
            "sufficient": value,
            "reason": "Direct evidence found." if value else "Only indirect evidence.",
            "missing_information": "" if value else "A trial measuring blood pressure.",
        }
    )


def rewrite(query: str) -> str:
    return json.dumps({"query": query})


def verdict(label: str = "SUPPORT", doc_id: int | str = 1, sentence: int = 0) -> str:
    evidence = (
        [] if label == "INSUFFICIENT" else [{"doc_id": doc_id, "sentence_indices": [sentence]}]
    )
    return json.dumps({"verdict": label, "rationale": "Grounded rationale.", "evidence": evidence})


def purposes(result: IterativeRAGResult) -> list[str]:
    return [str(event["purpose"]) for event in result.trace if event["type"] == "llm_call"]


def test_stops_immediately_when_initial_evidence_is_sufficient() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1, 2, 3, 4, 5, 6)})
    generator = FakeGenerator([sufficient(True), verdict("SUPPORT", 1)])

    result = IterativeRAGAgent(retriever, generator).verify(CLAIM, claim_id=7)

    assert retriever.calls == [(CLAIM, 5)]  # identical to Standard RAG's initial retrieval
    assert result.status == "success"
    assert result.prediction == "SUPPORT"
    assert result.stop_reason == "evidence_sufficient"
    assert purposes(result) == ["sufficiency", "verdict"]
    assert result.evidence_doc_ids == (1, 2, 3, 4, 5)
    assert not result.budget_exhausted
    assessment_event, verdict_event = result.trace[1], result.trace[2]
    assert assessment_event["parsed"]["sufficient"] is True
    assert verdict_event["parsed"]["verdict"] == "SUPPORT"


def test_insufficient_evidence_rewrites_query_and_retrieves_with_it() -> None:
    retriever = FakeRetriever(
        {CLAIM: ranking(1, 2), "drug X randomized trial blood pressure": ranking(3, 4)}
    )
    generator = FakeGenerator(
        [
            sufficient(False),
            rewrite("drug X randomized trial blood pressure"),
            sufficient(True),
            verdict("SUPPORT", 4, 1),
        ]
    )

    result = IterativeRAGAgent(retriever, generator).verify(CLAIM)

    assert retriever.calls == [(CLAIM, 5), ("drug X randomized trial blood pressure", 5)]
    assert purposes(result) == ["sufficiency", "query_rewrite", "sufficiency", "verdict"]
    assert result.stop_reason == "evidence_sufficient"
    assert result.status == "success"
    # The citation to a second-round document is accepted with its original ID and offset.
    assert result.response is not None
    assert result.response.evidence[0].doc_id == 4
    assert result.response.evidence[0].sentence_indices == (1,)
    rewrite_prompt = generator.prompts[1][1]
    assert "Only indirect evidence." in rewrite_prompt
    assert "A trial measuring blood pressure." in rewrite_prompt


def test_accumulated_evidence_is_deduplicated_by_source_id_with_provenance() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1, 2), "follow-up": ranking("2", 3)})
    generator = FakeGenerator(
        [sufficient(False), rewrite("follow-up"), sufficient(True), verdict("SUPPORT", 3)]
    )

    result = IterativeRAGAgent(retriever, generator).verify(CLAIM)

    assert result.evidence_doc_ids == (1, 2, 3)
    second_round = [event for event in result.trace if event["type"] == "retrieval"][1]
    assert second_round["duplicate_doc_ids"] == ["2"]
    assert second_round["admitted_doc_ids"] == [3]
    verdict_prompt = generator.prompts[-1][1]
    assert verdict_prompt.count("Document ID: 2") == 1
    assert "Document ID: 3" in verdict_prompt


def test_stops_at_max_retrieval_rounds_with_exact_call_and_token_counts() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1), "q2": ranking(2), "q3": ranking(3)})
    generator = FakeGenerator(
        [
            sufficient(False),
            rewrite("q2"),
            sufficient(False),
            rewrite("q3"),
            sufficient(False),
            verdict("INSUFFICIENT"),
        ]
    )
    config = IterativeRAGConfig(max_retrieval_rounds=3, max_llm_calls=6)

    result = IterativeRAGAgent(retriever, generator, config).verify(CLAIM)

    assert result.stop_reason == "max_retrieval_rounds"
    assert result.budget_exhausted
    assert result.prediction == "INSUFFICIENT"
    assert len(retriever.calls) == result.retrieval_calls == 3
    assert len(generator.prompts) == result.llm_calls == 6
    assert result.llm_calls_by_purpose == {"sufficiency": 3, "query_rewrite": 2, "verdict": 1}
    assert result.token_usage() == {
        "prompt_tokens": 60,
        "prompt_token_measurements": 6,
        "completion_tokens": 30,
        "completion_token_measurements": 6,
    }


def test_llm_call_budget_always_reserves_the_final_verdict_call() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1), "q2": ranking(2)})
    generator = FakeGenerator([sufficient(False), rewrite("q2"), verdict("INSUFFICIENT")])
    config = IterativeRAGConfig(max_retrieval_rounds=3, max_llm_calls=3)

    result = IterativeRAGAgent(retriever, generator, config).verify(CLAIM)

    assert result.stop_reason == "max_llm_calls"
    assert purposes(result) == ["sufficiency", "query_rewrite", "verdict"]
    assert result.retrieval_calls == 2
    assert result.status == "success"


def test_follow_up_with_only_duplicate_documents_stops_as_no_new_evidence() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1, 2), "same papers": ranking(2, 1)})
    generator = FakeGenerator([sufficient(False), rewrite("same papers"), verdict("INSUFFICIENT")])

    result = IterativeRAGAgent(retriever, generator).verify(CLAIM)

    assert result.stop_reason == "no_new_evidence"
    assert purposes(result) == ["sufficiency", "query_rewrite", "verdict"]
    assert not result.budget_exhausted


def test_repeated_query_stops_without_another_retrieval() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1)})
    generator = FakeGenerator(
        [sufficient(False), rewrite(f"  {CLAIM.upper()} "), verdict("INSUFFICIENT")]
    )

    result = IterativeRAGAgent(retriever, generator).verify(CLAIM)

    assert result.stop_reason == "repeated_query"
    assert retriever.calls == [(CLAIM, 5)]


def test_evidence_document_budget_is_recorded_and_excluded_documents_cannot_be_cited() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1, 2), "more": ranking(3, 4, 5)})
    generator = FakeGenerator(
        [sufficient(False), rewrite("more"), sufficient(True), verdict("SUPPORT", 5)]
    )
    config = IterativeRAGConfig(initial_k=2, followup_k=3, max_evidence_documents=3)

    result = IterativeRAGAgent(retriever, generator, config).verify(CLAIM)

    second_round = [event for event in result.trace if event["type"] == "retrieval"][1]
    assert second_round["admitted_doc_ids"] == [3]
    assert second_round["excluded_by_evidence_budget_doc_ids"] == [4, 5]
    assert result.evidence_doc_ids == (1, 2, 3)
    assert result.status == "invalid_response"
    assert result.prediction is None
    assert "not retrieved" in str(result.failure_message)


@pytest.mark.parametrize(
    ("retrieval", "replies", "status", "llm_calls"),
    [
        (RuntimeError("index offline"), [], "retrieval_failure", 0),
        (None, [RuntimeError("ollama down")], "inference_failure", 1),
        (None, ["not json"], "invalid_json", 1),
        (
            None,
            ['{"sufficient": "yes", "reason": "x", "missing_information": ""}'],
            "invalid_response",
            1,
        ),
        (
            None,
            [
                OllamaChatResponse(
                    sufficient(True), GenerationMetadata(8_192, 5, None, None, None, None, 0.1)
                )
            ],
            "context_limit_exceeded",
            1,
        ),
        (None, [sufficient(True), verdict("SUPPORT", 99)], "invalid_response", 2),
    ],
)
def test_failures_are_recorded_without_a_fabricated_prediction(
    retrieval: Exception | None,
    replies: list[str | Exception | OllamaChatResponse],
    status: str,
    llm_calls: int,
) -> None:
    retriever = FakeRetriever({CLAIM: retrieval if retrieval is not None else ranking(1)})
    generator = FakeGenerator(replies)
    config = IterativeRAGConfig(context_window_tokens=8_192)

    result = IterativeRAGAgent(retriever, generator, config).verify(CLAIM)

    assert result.status == status
    assert result.prediction is None
    assert result.response is None
    assert result.failure_message
    assert result.llm_calls == llm_calls
    assert result.retrieval_calls == 1


def test_second_round_retrieval_failure_keeps_counts_and_reports_failure() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1), "q2": RuntimeError("timeout")})
    generator = FakeGenerator([sufficient(False), rewrite("q2")])

    result = IterativeRAGAgent(retriever, generator).verify(CLAIM)

    assert result.status == "retrieval_failure"
    assert result.stop_reason == "failure"
    assert result.retrieval_calls == 2
    assert result.llm_calls == 2


def test_agent_never_receives_gold_labels_or_annotated_evidence_ids() -> None:
    labeled = Claim(
        11,
        CLAIM,
        ("GOLD-ONLY-777",),
        {"GOLD-ONLY-777": (EvidenceAnnotation("CONTRADICT", (0,)),)},
    )
    parameters = list(inspect.signature(IterativeRAGAgent.verify).parameters)
    assert parameters == ["self", "claim", "claim_id"]

    retriever = FakeRetriever({CLAIM: ranking(1), "q2": ranking(2)})
    generator = FakeGenerator(
        [sufficient(False), rewrite("q2"), sufficient(True), verdict("SUPPORT", 2)]
    )
    result = IterativeRAGAgent(retriever, generator).verify(
        labeled.claim, claim_id=labeled.claim_id
    )

    sent = [query for query, _ in retriever.calls] + [
        text for prompt in generator.prompts for text in prompt
    ]
    assert all("GOLD-ONLY-777" not in text for text in sent)
    assert all("CONTRADICT" not in user for _, user in generator.prompts)
    assert result.status == "success"


def test_verdict_prompt_is_the_unchanged_standard_rag_prompt_over_all_evidence() -> None:
    retriever = FakeRetriever({CLAIM: ranking(1), "q2": ranking(2)})
    generator = FakeGenerator(
        [sufficient(False), rewrite("q2"), sufficient(True), verdict("SUPPORT", 1)]
    )
    result = IterativeRAGAgent(retriever, generator).verify(CLAIM)

    expected_system, expected_user = build_messages(CLAIM, ranking(1) + ranking(2))
    assert generator.prompts[-1] == (expected_system, expected_user)
    assert "Document ID: 1" in expected_user and "Document ID: 2" in expected_user
    standard_hash = hashlib.sha256(expected_system.encode("utf-8")).hexdigest()
    assert prompt_fingerprints()["verdict"]["system_prompt_sha256"] == standard_hash
    assert result.status == "success"


def test_metrics_report_every_agent_call_not_one_per_claim() -> None:
    one_round = IterativeRAGAgent(
        FakeRetriever({"claim 1": ranking(1)}),
        FakeGenerator([sufficient(True), verdict("SUPPORT", 1)]),
    ).verify("claim 1", claim_id=1)
    three_rounds = IterativeRAGAgent(
        FakeRetriever({"claim 2": ranking(1), "q2": ranking(2), "q3": ranking(3)}),
        FakeGenerator(
            [
                sufficient(False),
                rewrite("q2"),
                sufficient(False),
                rewrite("q3"),
                sufficient(False),
                verdict("INSUFFICIENT"),
            ]
        ),
    ).verify("claim 2", claim_id=2)
    claims = (
        Claim(1, "claim 1", (), {"1": (EvidenceAnnotation("SUPPORT", (0,)),)}),
        Claim(2, "claim 2", (), {}),
    )

    metrics = evaluate_iterative_attempts(claims, {"1": one_round, "2": three_rounds})

    efficiency = metrics["efficiency"]
    assert efficiency["retrieval_calls"] == 4
    assert efficiency["llm_calls"] == 8
    assert efficiency["llm_calls_by_purpose"] == {
        "sufficiency": 4,
        "query_rewrite": 2,
        "verdict": 2,
    }
    assert efficiency["retrieval_rounds_per_claim"]["distribution"] == {"1": 1, "3": 1}
    assert efficiency["prompt_tokens_measured"] == 80
    assert efficiency["stop_reasons"] == {"evidence_sufficient": 1, "max_retrieval_rounds": 1}
    assert efficiency["budget_exhaustion"]["count"] == 1
    assert efficiency["distinct_evidence_documents"]["total_presented"] == 4
    assert metrics["classification"]["accuracy"] == 1.0
    assert metrics["citations"]["gold_document_matches"] == 1


def test_result_round_trips_through_a_resume_record() -> None:
    result = IterativeRAGAgent(
        FakeRetriever({CLAIM: ranking(1), "q2": ranking(2)}),
        FakeGenerator([sufficient(False), rewrite("q2"), sufficient(True), verdict("SUPPORT", 2)]),
    ).verify(CLAIM, claim_id="5")

    restored = IterativeRAGResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored.to_dict() == json.loads(json.dumps(result.to_dict()))
    assert restored.llm_calls == 4
    assert restored.retrieval_calls == 2
    assert restored.prediction == "SUPPORT"


def test_agent_output_schemas_bound_free_text_and_reason_before_deciding() -> None:
    sufficiency = sufficiency_json_schema()
    properties = sufficiency["properties"]
    # Ollama emits keys alphabetically: reasoning first, then the length-bounded-safe decision.
    assert sorted(properties) == ["missing_information", "reason", "sufficient"]
    assert properties["reason"]["maxLength"] <= 300
    assert properties["missing_information"]["maxLength"] <= 300
    assert rewrite_json_schema()["properties"]["query"]["maxLength"] <= 300


def test_invalid_budget_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        IterativeRAGConfig(max_retrieval_rounds=0)
    with pytest.raises(ValueError):
        IterativeRAGConfig(max_llm_calls=0)
    with pytest.raises(ValueError):
        IterativeRAGConfig(initial_k=5, max_evidence_documents=4)


def _load_cli_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_iterative_rag.py"
    spec = importlib.util.spec_from_file_location("run_iterative_rag", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_selects_the_same_claims_as_standard_rag() -> None:
    labels = ("SUPPORT", "CONTRADICT", None)
    claims = tuple(
        Claim(
            index,
            f"claim {index}",
            (),
            {}
            if labels[index % 3] is None
            else {str(index): (EvidenceAnnotation(labels[index % 3], (0,)),)},
        )
        for index in range(1, 31)
    )
    validation = select_internal_validation_claims(claims, seed=42)
    cli = _load_cli_module()

    full, full_mode = cli.select_claims(validation, smoke_test=False, limit=None)
    smoke, smoke_mode = cli.select_claims(validation, smoke_test=True, limit=None)
    partial, partial_mode = cli.select_claims(validation, smoke_test=False, limit=2)

    assert [claim.claim_id for claim in full] == [claim.claim_id for claim in validation]
    assert smoke == select_smoke_claims(validation, limit=5)
    assert partial == validation[:2]
    assert (full_mode, smoke_mode, partial_mode) == (
        "internal_validation",
        "smoke_test",
        "partial_internal_validation",
    )
