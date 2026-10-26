"""Focused contract tests for the Experiment 3 ARGUS graph."""

from __future__ import annotations

import json

import pytest

from argus.agents.argus import ArgusAgent, ArgusConfig, evaluate_argus_attempts, merge_evidence
from argus.data.models import Claim, CorpusDocument, EvidenceAnnotation
from argus.llm.ollama import GenerationMetadata, OllamaChatResponse
from argus.retrieval.types import RetrievalResult


def item(doc_id: int) -> RetrievalResult:
    document = CorpusDocument(doc_id, f"Paper {doc_id}", ("Evidence sentence.",), False)
    return RetrievalResult(doc_id, 1.0, 1, document)


class RetrieverStub:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        self.queries.append(query)
        return (item(1),) if len(self.queries) == 1 else (item(1), item(2))


class GeneratorStub:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = iter(responses)
        self.users: list[str] = []

    def chat(self, *, system: str, user: str, response_schema: object) -> OllamaChatResponse:
        self.users.append(user)
        return OllamaChatResponse(
            json.dumps(next(self.responses)),
            GenerationMetadata(10, 5, None, None, None, None, 0.01),
        )


def test_merge_preserves_cross_agent_provenance_and_is_deterministic() -> None:
    merged = merge_evidence((item(1),), (item(1), item(2)), 2)
    assert [(value.result.doc_id, value.discovered_by) for value in merged] == [
        (1, ("support", "challenge")),
        (2, ("challenge",)),
    ]


def test_full_graph_keeps_provenance_out_of_verifier_and_validates_judge_citations() -> None:
    generator = GeneratorStub(
        [
            {"label": "SUPPORT", "rationale": "Direct.", "sentence_indices": [0]},
            {"label": "CONTRADICT", "rationale": "Direct.", "sentence_indices": [0]},
            {
                "verdict": "SUPPORT",
                "rationale": "Weighed evidence.",
                "evidence": [{"doc_id": 1, "sentence_indices": [0]}],
            },
        ]
    )
    result = ArgusAgent(RetrieverStub(), generator).verify("A scientific claim", claim_id=7)
    assert result.status == "success"
    assert result.prediction == "SUPPORT"
    assert result.retrieval_calls == 2
    assert result.llm_calls == 3
    assert "discovered_by" not in generator.users[0]
    assert "discovered_by" not in generator.users[1]
    assert "exceptions limitations" in result.trace[2]["query"]


def test_no_challenge_performs_no_challenge_retrieval_or_inference() -> None:
    retriever = RetrieverStub()
    generator = GeneratorStub(
        [
            {"label": "INSUFFICIENT", "rationale": "Irrelevant.", "sentence_indices": []},
            {"verdict": "INSUFFICIENT", "rationale": "Not enough.", "evidence": []},
        ]
    )
    result = ArgusAgent(retriever, generator, ArgusConfig(challenge_enabled=False)).verify("claim")
    assert len(retriever.queries) == 1
    assert result.retrieval_calls == 1
    assert all(event.get("agent") != "challenge" for event in result.trace)


@pytest.mark.parametrize(
    "verification, status",
    [
        ({"label": "SUPPORT", "rationale": "x", "sentence_indices": []}, "invalid_response"),
        ({"label": "SUPPORT", "rationale": "x", "sentence_indices": [9]}, "invalid_response"),
    ],
)
def test_invalid_verifier_citations_are_recorded_as_failures(
    verification: dict[str, object], status: str
) -> None:
    result = ArgusAgent(
        RetrieverStub(), GeneratorStub([verification]), ArgusConfig(challenge_enabled=False)
    ).verify("claim")
    assert result.status == status
    assert result.prediction is None


def test_argus_metrics_keep_gold_access_post_inference_and_separate_challenge_contribution() -> (
    None
):
    result = ArgusAgent(
        RetrieverStub(),
        GeneratorStub(
            [
                {"label": "SUPPORT", "rationale": "Direct.", "sentence_indices": [0]},
                {"label": "CONTRADICT", "rationale": "Direct.", "sentence_indices": [0]},
                {
                    "verdict": "SUPPORT",
                    "rationale": "Grounded.",
                    "evidence": [{"doc_id": 1, "sentence_indices": [0]}],
                },
            ]
        ),
    ).verify("claim", claim_id=1)
    claim = Claim(
        1,
        "claim",
        (),
        {
            "1": (EvidenceAnnotation("SUPPORT", (0,)),),
            "2": (EvidenceAnnotation("CONTRADICT", (0,)),),
        },
    )
    metrics = evaluate_argus_attempts((claim,), {"1": result})
    retrieval = metrics["argus"]["retrieval"]
    assert retrieval["all_rounds_gold_document_recall"] == 1.0
    assert retrieval["challenge_unique_gold_documents"] == 1
    assert metrics["argus"]["evidence"]["verified_labels"] == {
        "CONTRADICT": 1,
        "INSUFFICIENT": 0,
        "SUPPORT": 1,
    }
    assert metrics["argus"]["efficiency"]["llm_calls_by_component"] == {
        "evidence_verifier": 2,
        "judge": 1,
    }
