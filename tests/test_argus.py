"""Focused contract tests for the Experiment 3 ARGUS graph."""

from __future__ import annotations

import json

import pytest

from argus.agents.argus import (
    CONCISE_REASONING_INSTRUCTION,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_V4_SYSTEM_PROMPT,
    JUDGE_V4_VERSION,
    MAX_JUDGE_EVIDENCE_REFERENCES,
    MAX_JUDGE_V4_EVIDENCE_REFERENCES,
    MAX_JUDGE_V4_SENTENCE_INDICES,
    MAX_RATIONALE_CHARS,
    PROMPT_VERSION,
    VERIFIER_SYSTEM_PROMPT,
    ArgusAgent,
    ArgusConfig,
    ArgusResult,
    _parse_verification,
    evaluate_argus_attempts,
    judge_json_schema,
    merge_evidence,
    prompt_fingerprints,
    verifier_json_schema,
)
from argus.data.models import Claim, CorpusDocument, EvidenceAnnotation
from argus.llm.ollama import GenerationMetadata, OllamaChatResponse
from argus.rag.standard import response_json_schema
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


class EmptyRetrieverStub:
    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        return ()


class GeneratorStub:
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.users: list[str] = []
        self.systems: list[str] = []
        self.schemas: list[object] = []

    def chat(self, *, system: str, user: str, response_schema: object) -> OllamaChatResponse:
        self.users.append(user)
        self.systems.append(system)
        self.schemas.append(response_schema)
        response = next(self.responses)
        return OllamaChatResponse(
            response if isinstance(response, str) else json.dumps(response),
            GenerationMetadata(10, 5, None, None, None, None, 0.01),
        )


def test_argus_v3_schemas_bound_rationales_without_mutating_standard_rag_schema() -> None:
    verifier_schema = verifier_json_schema()
    judge_schema = judge_json_schema()

    assert verifier_schema["properties"]["rationale"]["maxLength"] == MAX_RATIONALE_CHARS
    assert judge_schema["properties"]["rationale"]["maxLength"] == MAX_RATIONALE_CHARS
    assert judge_schema["properties"]["evidence"]["maxItems"] == MAX_JUDGE_EVIDENCE_REFERENCES
    assert judge_schema["required"] == ["verdict", "rationale", "evidence"]
    assert "maxLength" not in response_json_schema()["properties"]["rationale"]


def test_argus_v3_prompts_and_fingerprints_record_citation_requirement() -> None:
    assert PROMPT_VERSION == "argus-v3"
    assert CONCISE_REASONING_INSTRUCTION in VERIFIER_SYSTEM_PROMPT
    assert CONCISE_REASONING_INSTRUCTION in JUDGE_SYSTEM_PROMPT
    assert "Do not return SUPPORT or CONTRADICT with an empty" in VERIFIER_SYSTEM_PROMPT
    fingerprints = prompt_fingerprints()
    assert fingerprints["evidence_verifier"]["version"] == "argus-v3"
    assert fingerprints["judge"]["version"] == "argus-v3"
    assert (
        fingerprints["evidence_verifier"]["system_prompt_sha256"]
        != fingerprints["judge"]["system_prompt_sha256"]
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


def test_no_retrieved_abstracts_are_distinct_from_invalid_verifier_responses() -> None:
    generator = GeneratorStub([])
    result = ArgusAgent(
        EmptyRetrieverStub(), generator, ArgusConfig(challenge_enabled=False)
    ).verify("claim")

    assert result.status == "retrieval_failure"
    assert result.failure_message == "no abstracts were retrieved for evidence verification"
    assert not generator.users
    assert result.trace[-1]["outcome"] == "no_abstracts_retrieved"


@pytest.mark.parametrize(
    "payload, valid",
    [
        ({"label": "SUPPORT", "rationale": "x", "sentence_indices": []}, False),
        ({"label": "CONTRADICT", "rationale": "x", "sentence_indices": []}, False),
        ({"label": "INSUFFICIENT", "rationale": "x", "sentence_indices": []}, True),
        ({"label": "SUPPORT", "rationale": "x", "sentence_indices": [0]}, True),
        ({"label": "CONTRADICT", "rationale": "x", "sentence_indices": [0]}, True),
    ],
)
def test_verifier_citation_rules_remain_strict(payload: dict[str, object], valid: bool) -> None:
    raw = json.dumps(payload)
    evidence = merge_evidence((item(1),), (), 1)[0]
    if valid:
        assert _parse_verification(raw, evidence)[0] == payload["label"]
    else:
        with pytest.raises(ValueError, match="require a cited sentence"):
            _parse_verification(raw, evidence)


def test_invalid_verifier_document_is_excluded_and_valid_document_reaches_judge() -> None:
    generator = GeneratorStub(
        [
            {"label": "SUPPORT", "rationale": "x", "sentence_indices": []},
            {"label": "CONTRADICT", "rationale": "Direct.", "sentence_indices": [0]},
            {
                "verdict": "CONTRADICT",
                "rationale": "Grounded.",
                "evidence": [{"doc_id": 2, "sentence_indices": [0]}],
            },
        ]
    )
    result = ArgusAgent(RetrieverStub(), generator).verify("claim", claim_id=9)

    assert result.status == "success"
    assert [item.evidence.result.doc_id for item in result.verified_evidence] == [2]
    verifier_events = [
        event for event in result.trace if event.get("purpose") == "evidence_verifier"
    ]
    assert [
        (event["doc_id"], event["validation_status"], event["admitted"])
        for event in verifier_events
    ] == [
        (1, "invalid", False),
        (2, "valid", True),
    ]
    assert verifier_events[0]["raw_response"]
    assert "Document ID: 2" in generator.users[-1]
    assert "Document ID: 1" not in generator.users[-1]
    assert result.llm_calls == 3


def test_valid_verifier_document_survives_a_later_invalid_document() -> None:
    generator = GeneratorStub(
        [
            {"label": "SUPPORT", "rationale": "Direct.", "sentence_indices": [0]},
            {"label": "CONTRADICT", "rationale": "x", "sentence_indices": []},
            {
                "verdict": "SUPPORT",
                "rationale": "Grounded.",
                "evidence": [{"doc_id": 1, "sentence_indices": [0]}],
            },
        ]
    )
    result = ArgusAgent(RetrieverStub(), generator).verify("claim")

    assert result.status == "success"
    assert [item.evidence.result.doc_id for item in result.verified_evidence] == [1]
    assert "Document ID: 1" in generator.users[-1]
    assert "Document ID: 2" not in generator.users[-1]


def test_all_invalid_verifier_responses_fail_without_calling_judge_and_preserve_raw_trace() -> None:
    generator = GeneratorStub(
        [
            "not json",
            {"label": "CONTRADICT", "rationale": "x", "sentence_indices": [9]},
        ]
    )
    result = ArgusAgent(RetrieverStub(), generator).verify("claim")

    assert result.status == "invalid_response"
    assert result.prediction is None
    assert len(generator.users) == 2
    verifier_events = [
        event for event in result.trace if event.get("purpose") == "evidence_verifier"
    ]
    assert [event["doc_id"] for event in verifier_events] == [1, 2]
    assert [event["error_category"] for event in verifier_events] == [
        "invalid_json",
        "invalid_response",
    ]
    assert result.trace[-1]["outcome"] == "all_verifier_responses_invalid"


def test_argus_result_rehydration_preserves_verifier_failure_trace() -> None:
    result = ArgusAgent(RetrieverStub(), GeneratorStub(["not json", "not json"])).verify(
        "claim", claim_id=9
    )
    rehydrated = ArgusResult.from_dict(result.to_dict())

    assert rehydrated.trace == result.trace


def test_argus_judge_keeps_shared_citation_validation() -> None:
    result = ArgusAgent(
        RetrieverStub(),
        GeneratorStub(
            [
                {"label": "SUPPORT", "rationale": "Direct.", "sentence_indices": [0]},
                {"label": "CONTRADICT", "rationale": "Direct.", "sentence_indices": [0]},
                {
                    "verdict": "SUPPORT",
                    "rationale": "Grounded.",
                    "evidence": [{"doc_id": 99, "sentence_indices": [0]}],
                },
            ]
        ),
    ).verify("claim")

    assert result.status == "invalid_response"
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
    evidence = metrics["argus"]["evidence"]
    assert evidence["verification_attempts"] == 2
    assert evidence["successful_verifier_classifications"] == 2
    assert evidence["overall_verifier_valid_response_rate"] == 1.0


def test_v4_schema_bounds_citations_without_changing_v3_or_standard() -> None:
    v3 = judge_json_schema()
    v4 = judge_json_schema(version="v4")
    v4_evidence = v4["properties"]["evidence"]
    v4_indices = v4_evidence["items"]["properties"]["sentence_indices"]

    assert v3["properties"]["evidence"]["maxItems"] == MAX_JUDGE_EVIDENCE_REFERENCES
    assert "maxItems" not in v3["properties"]["evidence"]["items"]["properties"]["sentence_indices"]
    assert v4_evidence["maxItems"] == MAX_JUDGE_V4_EVIDENCE_REFERENCES
    assert v4_indices["minItems"] == 1
    assert v4_indices["maxItems"] == MAX_JUDGE_V4_SENTENCE_INDICES
    assert v4_indices["uniqueItems"] is True
    assert "maxItems" not in response_json_schema()["properties"]["evidence"]


def test_v4_changes_only_judge_prompt_fingerprint() -> None:
    v3 = prompt_fingerprints()
    v4 = prompt_fingerprints("v4")

    assert v4["evidence_verifier"] == v3["evidence_verifier"]
    assert v4["judge"]["version"] == JUDGE_V4_VERSION
    assert v4["judge"]["system_prompt_sha256"] != v3["judge"]["system_prompt_sha256"]
    assert "For INSUFFICIENT with no decisive evidence" in JUDGE_V4_SYSTEM_PROMPT


def test_v4_judge_allows_empty_evidence_for_insufficient() -> None:
    generator = GeneratorStub(
        [
            {"label": "INSUFFICIENT", "rationale": "Not direct evidence.", "sentence_indices": []},
            {"verdict": "INSUFFICIENT", "rationale": "Evidence is inconclusive.", "evidence": []},
        ]
    )
    result = ArgusAgent(
        RetrieverStub(), generator, ArgusConfig(challenge_enabled=False, judge_version="v4")
    ).verify("claim")

    assert result.status == "success"
    assert result.response is not None
    assert result.response.evidence == ()
    assert generator.systems[0] == VERIFIER_SYSTEM_PROMPT
    assert generator.systems[-1] == JUDGE_V4_SYSTEM_PROMPT
    assert generator.schemas[-1] == judge_json_schema(version="v4")
    judge_event = next(event for event in result.trace if event.get("purpose") == "judge")
    assert judge_event["prompt_version"] == JUDGE_V4_VERSION


@pytest.mark.parametrize(
    "bad_evidence",
    [
        [{"doc_id": 1, "sentence_indices": []}],
        [{"doc_id": 1, "sentence_indices": [9]}],
    ],
)
def test_v4_judge_keeps_strict_citation_validation(
    bad_evidence: list[dict[str, object]],
) -> None:
    result = ArgusAgent(
        RetrieverStub(),
        GeneratorStub(
            [
                {"label": "SUPPORT", "rationale": "Direct.", "sentence_indices": [0]},
                {
                    "verdict": "SUPPORT",
                    "rationale": "Grounded.",
                    "evidence": bad_evidence,
                },
            ]
        ),
        ArgusConfig(challenge_enabled=False, judge_version="v4"),
    ).verify("claim")

    assert result.status == "invalid_response"
    assert result.prediction is None
    assert result.raw_response is not None
