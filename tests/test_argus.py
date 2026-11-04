"""Focused contract tests for the Experiment 3 ARGUS graph."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from argus.agents.argus import (
    CHALLENGE_QUERY_SUFFIX,
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
    exclude_documents,
    judge_json_schema,
    merge_evidence,
    prompt_fingerprints,
    verifier_json_schema,
)
from argus.data.models import Claim, CorpusDocument, EvidenceAnnotation
from argus.llm.ollama import GenerationMetadata, OllamaChatResponse
from argus.rag.standard import response_json_schema
from argus.retrieval import BM25Retriever
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


class RecordingRetriever:
    def __init__(self, *doc_ids: int) -> None:
        self.doc_ids = doc_ids
        self.queries: list[str] = []
        self.requested_k: list[int] = []

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        self.queries.append(query)
        self.requested_k.append(k)
        return tuple(item(doc_id) for doc_id in self.doc_ids[:k])


class FailingRetriever:
    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        raise OSError("index unavailable")


def insufficient_run(document_count: int) -> GeneratorStub:
    verifications: list[object] = [
        {"label": "INSUFFICIENT", "rationale": "Not direct.", "sentence_indices": []}
    ] * document_count
    return GeneratorStub(
        [*verifications, {"verdict": "INSUFFICIENT", "rationale": "Not enough.", "evidence": []}]
    )


def test_default_challenge_configuration_keeps_recorded_dense_suffix_behavior() -> None:
    retriever = RecordingRetriever(1, 2)
    result = ArgusAgent(retriever, insufficient_run(2)).verify("A claim")
    assert retriever.queries == [
        "A claim",
        "A claim exceptions limitations conflicting findings alternative outcomes",
    ]
    assert ArgusConfig().variant() == "full"
    assert ArgusConfig(challenge_enabled=False).variant() == "no_challenge"
    assert result.status == "success"
    assert retriever.requested_k == [5, 5]
    challenge = next(event for event in result.trace if event.get("agent") == "challenge")
    assert "excluded_doc_ids" not in challenge
    assert "requested_k" not in challenge


def test_bm25_challenge_searches_claim_while_support_stays_on_dense() -> None:
    dense, bm25 = RecordingRetriever(1, 2), RecordingRetriever(2, 3)
    config = ArgusConfig(challenge_retrieval="bm25", challenge_query="claim")
    generator = insufficient_run(3)
    result = ArgusAgent(dense, generator, config, challenge_retriever=bm25).verify("A claim")
    assert dense.queries == ["A claim"]
    assert bm25.queries == ["A claim"]
    assert [(entry.result.doc_id, entry.discovered_by) for entry in result.evidence] == [
        (1, ("support",)),
        (2, ("support", "challenge")),
        (3, ("challenge",)),
    ]
    assert result.llm_calls == 4
    assert config.variant() == "bm25_claim"
    assert all("bm25" not in user.lower() and "challenge" not in user for user in generator.users)


def test_challenge_query_choice_is_independent_of_the_retriever() -> None:
    dense, bm25 = RecordingRetriever(1), RecordingRetriever(2)
    config = ArgusConfig(challenge_retrieval="bm25")
    ArgusAgent(dense, insufficient_run(2), config, challenge_retriever=bm25).verify("A claim")
    assert bm25.queries == [f"A claim {CHALLENGE_QUERY_SUFFIX}"]
    assert config.variant() == "bm25_suffix"
    assert ArgusConfig(challenge_query="claim").variant() == "dense_claim"


def test_real_bm25_retriever_satisfies_the_challenge_contract() -> None:
    corpus = (
        CorpusDocument(1, "Zebrafish fins", ("Regeneration of fins.",), False),
        CorpusDocument(2, "Radioiodine goitre", ("Radioiodine reduces goitre volume.",), False),
        CorpusDocument(3, "Mouse sleep", ("Sleep in mice.",), False),
    )
    config = ArgusConfig(retrieval_k=1, challenge_retrieval="bm25", challenge_query="claim")
    agent = ArgusAgent(
        RecordingRetriever(1),
        insufficient_run(2),
        config,
        challenge_retriever=BM25Retriever(corpus),
    )
    result = agent.verify("Radioiodine changes goitre volume")
    challenge = next(event for event in result.trace if event.get("agent") == "challenge")
    assert challenge["retrieved_doc_ids"] == [2]
    assert [entry.result.doc_id for entry in result.evidence] == [1, 2]


def test_bm25_challenge_failure_is_recorded_without_a_verdict() -> None:
    config = ArgusConfig(challenge_retrieval="bm25", challenge_query="claim")
    generator = GeneratorStub([])
    result = ArgusAgent(
        RecordingRetriever(1), generator, config, challenge_retriever=FailingRetriever()
    ).verify("A claim")
    assert result.status == "retrieval_failure"
    assert result.prediction is None
    assert "challenge retrieval failed: index unavailable" in str(result.failure_message)
    assert generator.users == []


def test_challenge_retriever_must_match_the_recorded_configuration() -> None:
    with pytest.raises(ValueError, match="challenge_retriever"):
        ArgusAgent(
            RecordingRetriever(1), GeneratorStub([]), ArgusConfig(challenge_retrieval="bm25")
        )
    with pytest.raises(ValueError, match="challenge_retriever"):
        ArgusAgent(
            RecordingRetriever(1), GeneratorStub([]), challenge_retriever=RecordingRetriever(2)
        )


def ranked(*doc_ids: int | str) -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            doc_id,
            1.0 / rank,
            rank,
            CorpusDocument(doc_id, f"Paper {doc_id}", ("Evidence sentence.",), False),
        )
        for rank, doc_id in enumerate(doc_ids, start=1)
    )


def test_exclude_documents_keeps_order_scores_and_renumbers_ranks() -> None:
    results = ranked(1, 2, 3, 4)
    kept = exclude_documents(results, ranked(1, 3), 5)
    assert [(entry.doc_id, entry.rank, entry.score) for entry in kept] == [
        (2, 1, 0.5),
        (4, 2, 0.25),
    ]
    assert [entry.rank for entry in results] == [1, 2, 3, 4]


def test_exclude_documents_applies_limit_and_canonical_ids() -> None:
    kept = exclude_documents(ranked(1, "2", 3, 4), ranked("1", 2), 1)
    assert [(entry.doc_id, entry.rank) for entry in kept] == [(3, 1)]


def test_exclude_support_fetches_enough_and_returns_only_new_documents() -> None:
    dense = RecordingRetriever(1, 2, 3, 4, 5, 6, 7)
    config = ArgusConfig(retrieval_k=2, challenge_query="claim", challenge_exclude_support=True)
    result = ArgusAgent(dense, insufficient_run(4), config).verify("A claim")
    assert dense.queries == ["A claim", "A claim"]
    assert dense.requested_k == [2, 4]
    challenge = next(event for event in result.trace if event.get("agent") == "challenge")
    assert challenge["retrieved_doc_ids"] == [3, 4]
    assert challenge["k"] == 2
    assert challenge["requested_k"] == 4
    assert challenge["excluded_doc_ids"] == [1, 2]
    assert [(entry.result.doc_id, entry.discovered_by) for entry in result.evidence] == [
        (1, ("support",)),
        (2, ("support",)),
        (3, ("challenge",)),
        (4, ("challenge",)),
    ]
    assert [entry.result.rank for entry in result.evidence[2:]] == [1, 2]
    assert config.variant() == "dense_claim_exclude"


def test_exclude_support_full_evidence_load_stays_within_the_llm_budget() -> None:
    dense = RecordingRetriever(*range(1, 11))
    config = ArgusConfig(challenge_query="claim", challenge_exclude_support=True)
    generator = insufficient_run(10)
    result = ArgusAgent(dense, generator, config).verify("A claim")
    assert len(result.evidence) == 10
    assert result.llm_calls == 11
    assert result.status == "success"


def test_exclude_support_records_outcome_when_nothing_new_exists() -> None:
    config = ArgusConfig(challenge_query="claim", challenge_exclude_support=True)
    result = ArgusAgent(RecordingRetriever(1), insufficient_run(1), config).verify("A claim")
    challenge = next(event for event in result.trace if event.get("agent") == "challenge")
    assert challenge["retrieved_doc_ids"] == []
    assert challenge["outcome"] == "no_reliable_counter_evidence_found"
    assert result.status == "success"


def test_exclude_support_works_with_a_separate_challenge_retriever() -> None:
    dense, bm25 = RecordingRetriever(1, 2), RecordingRetriever(2, 3, 4)
    config = ArgusConfig(
        retrieval_k=2,
        challenge_retrieval="bm25",
        challenge_query="claim",
        challenge_exclude_support=True,
    )
    result = ArgusAgent(dense, insufficient_run(4), config, challenge_retriever=bm25).verify(
        "A claim"
    )
    assert dense.requested_k == [2]
    assert bm25.requested_k == [4]
    challenge = next(event for event in result.trace if event.get("agent") == "challenge")
    assert challenge["retrieved_doc_ids"] == [3, 4]
    assert config.variant() == "bm25_claim_exclude"


def test_exclude_support_with_default_query_is_not_labelled_full() -> None:
    assert ArgusConfig(challenge_exclude_support=True).variant() == "dense_suffix_exclude"


@pytest.mark.parametrize(
    "options",
    [
        {"challenge_retrieval": "bm25"},
        {"challenge_query": "claim"},
        {"challenge_exclude_support": True},
    ],
)
def test_challenge_options_are_rejected_without_the_challenge_agent(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="require the Challenge agent"):
        ArgusConfig(challenge_enabled=False, **options)


@pytest.mark.parametrize("options", [{"challenge_retrieval": "faiss"}, {"challenge_query": "not"}])
def test_unknown_challenge_options_are_rejected(options: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="must be"):
        ArgusConfig(**options)


def _load_cli_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_argus.py"
    spec = importlib.util.spec_from_file_location("run_argus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeOllamaClient:
    def __init__(self, **_: object) -> None:
        pass

    def model_identity(self) -> SimpleNamespace:
        return SimpleNamespace(digest="fake-digest", to_dict=lambda: {"digest": "fake-digest"})

    def chat(self, *, system: str, user: str, response_schema: object) -> OllamaChatResponse:
        payload: dict[str, object] = (
            {"verdict": "INSUFFICIENT", "rationale": "Not enough.", "evidence": []}
            if system == JUDGE_SYSTEM_PROMPT
            else {"label": "INSUFFICIENT", "rationale": "Not direct.", "sentence_indices": []}
        )
        return OllamaChatResponse(
            json.dumps(payload), GenerationMetadata(10, 5, None, None, None, None, 0.01)
        )


def run_cli(
    monkeypatch: pytest.MonkeyPatch, output_dir: Path, *flags: str
) -> tuple[int, ModuleType]:
    module = _load_cli_module()
    corpus = (
        CorpusDocument(1, "Zebrafish fins", ("Regeneration of fins.",), False),
        CorpusDocument(2, "Radioiodine goitre", ("Radioiodine reduces goitre volume.",), False),
        CorpusDocument(3, "Mouse sleep", ("Sleep in mice.",), False),
    )
    monkeypatch.setattr(module, "OllamaChatClient", FakeOllamaClient)
    monkeypatch.setattr(module, "load_scifact_dataset", lambda _: SimpleNamespace(corpus=corpus))
    monkeypatch.setattr(module, "validate_scifact_dataset", lambda _: None)
    monkeypatch.setattr(
        module,
        "build_dense_retriever",
        lambda *_, **__: (RecordingRetriever(1), SimpleNamespace(corpus_fingerprint="fingerprint")),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_argus.py", "--claim", "Radioiodine goitre", "--output-dir", str(output_dir), *flags],
    )
    return module.main(), module


def read_cli_metadata(output_dir: Path) -> dict[str, object]:
    run_dir = next(output_dir.iterdir())
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def test_cli_records_the_bm25_claim_arm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exit_code, module = run_cli(
        monkeypatch,
        tmp_path,
        "--k",
        "1",
        "--challenge-retrieval",
        "bm25",
        "--challenge-query",
        "claim",
    )
    assert exit_code == 0
    metadata = read_cli_metadata(tmp_path)
    assert metadata["variant"] == "bm25_claim"
    assert metadata["implementation_version"] == "argus-v3-challenge-retrieval"
    assert metadata["retrieval"]["method"] == "dense BGE-small plus FAISS IndexFlatIP"
    assert metadata["retrieval"]["challenge"] == {
        "method": module.CHALLENGE_METHODS["bm25"],
        "query": "claim",
        "exclude_support": False,
    }
    run_dir = next(tmp_path.iterdir())
    record = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    challenge = next(event for event in record["trace"] if event.get("agent") == "challenge")
    assert challenge["query"] == "Radioiodine goitre"
    assert challenge["retrieved_doc_ids"] == [2]


def test_cli_without_challenge_flags_keeps_original_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exit_code, module = run_cli(monkeypatch, tmp_path)
    assert exit_code == 0
    metadata = read_cli_metadata(tmp_path)
    assert metadata["variant"] == "full"
    assert metadata["implementation_version"] == "argus-v3"
    assert metadata["retrieval"]["challenge"] == {
        "method": module.CHALLENGE_METHODS["dense"],
        "query": "suffix",
        "exclude_support": False,
    }


def test_cli_records_the_exclude_support_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exit_code, module = run_cli(
        monkeypatch, tmp_path, "--challenge-query", "claim", "--challenge-exclude-support"
    )
    assert exit_code == 0
    metadata = read_cli_metadata(tmp_path)
    assert metadata["variant"] == "dense_claim_exclude"
    assert metadata["implementation_version"] == "argus-v3-challenge-retrieval"
    assert metadata["retrieval"]["challenge"] == {
        "method": module.CHALLENGE_METHODS["dense"],
        "query": "claim",
        "exclude_support": True,
    }


def test_cli_rejects_exclude_support_with_no_challenge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exit_code, _ = run_cli(monkeypatch, tmp_path, "--no-challenge", "--challenge-exclude-support")
    assert exit_code == 2
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_challenge_flags_with_no_challenge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exit_code, _ = run_cli(monkeypatch, tmp_path, "--no-challenge", "--challenge-query", "claim")
    assert exit_code == 2
    assert list(tmp_path.iterdir()) == []


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
    assert "uniqueItems" not in v4_indices
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
        [{"doc_id": 1, "sentence_indices": [0, 0, 0, 0]}],
        [
            {"doc_id": 1, "sentence_indices": [0]},
            {"doc_id": 1, "sentence_indices": [0]},
        ],
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


def test_v4_judge_truncated_json_is_recorded_without_retry() -> None:
    generator = GeneratorStub(
        [
            {"label": "SUPPORT", "rationale": "Direct.", "sentence_indices": [0]},
            '{"evidence": [{"doc_id": 1, "sentence_indices": [0]}], "rationale":',
        ]
    )
    result = ArgusAgent(
        RetrieverStub(), generator, ArgusConfig(challenge_enabled=False, judge_version="v4")
    ).verify("claim")

    assert result.status == "invalid_json"
    assert result.prediction is None
    assert result.llm_calls == 2
    assert result.raw_response is not None
    assert result.trace[-1]["purpose"] == "judge"
