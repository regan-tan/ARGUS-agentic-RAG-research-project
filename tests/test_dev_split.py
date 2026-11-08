"""The opt-in official dev split (`--split dev`) for the three claim-verification CLIs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from argus.data.models import Claim, CorpusDocument, EvidenceAnnotation
from argus.evaluation.verification import OFFICIAL_DEV_SOURCE, select_official_dev_claims
from argus.llm.ollama import GenerationMetadata, OllamaChatResponse
from argus.retrieval.types import RetrievalResult

SCRIPTS = ("run_standard_rag.py", "run_iterative_rag.py", "run_argus.py")


def dev_claims(total: int = 300, evidence_bearing: int = 188) -> tuple[Claim, ...]:
    """Synthetic labeled dev claims with the official split's shape."""

    claims = []
    for index in range(1, total + 1):
        label = "SUPPORT" if index % 2 else "CONTRADICT"
        evidence = {"1": (EvidenceAnnotation(label, (0,)),)} if index <= evidence_bearing else {}
        claims.append(Claim(index, f"dev claim {index}", (1,), evidence))
    return tuple(claims)


def test_official_dev_selection_is_complete_ordered_and_guarded() -> None:
    claims = dev_claims()
    selected, mode = select_official_dev_claims(tuple(reversed(claims)))
    assert mode == "official_dev"
    assert len(selected) == 300
    assert [claim.claim_id for claim in selected] == sorted(
        (claim.claim_id for claim in claims), key=str
    )

    partial, partial_mode = select_official_dev_claims(claims, limit=3)
    assert partial_mode == "partial_official_dev"
    assert [claim.claim_id for claim in partial] == [1, 10, 100]

    with pytest.raises(ValueError, match="300 claims"):
        select_official_dev_claims(claims[:299])
    with pytest.raises(ValueError, match="188 evidence-bearing"):
        select_official_dev_claims(dev_claims(evidence_bearing=187))
    with pytest.raises(ValueError, match="labeled"):
        select_official_dev_claims((*claims[:-1], Claim(300, "unlabeled", (), None)))
    with pytest.raises(ValueError, match="positive"):
        select_official_dev_claims(claims, limit=0)


def _load(script: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    spec = importlib.util.spec_from_file_location(script.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Client:
    """Answers every call with an empty INSUFFICIENT verdict; no network."""

    def __init__(self, **_: object) -> None:
        pass

    def model_identity(self) -> SimpleNamespace:
        return SimpleNamespace(digest="fake-digest", to_dict=lambda: {"digest": "fake-digest"})

    def chat(self, **_: object) -> OllamaChatResponse:
        payload = {"verdict": "INSUFFICIENT", "rationale": "Not enough.", "evidence": []}
        return OllamaChatResponse(
            json.dumps(payload), GenerationMetadata(10, 5, None, None, None, None, 0.01)
        )


class _Retriever:
    def __init__(self, document: CorpusDocument) -> None:
        self.document = document

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        return (RetrievalResult(self.document.doc_id, 1.0, 1, self.document),)


def _run(monkeypatch: pytest.MonkeyPatch, script: str, output_dir: Path, *flags: str) -> int:
    module = _load(script)
    document = CorpusDocument(1, "Paper", ("A finding.",), False)
    dataset = SimpleNamespace(corpus=(document,), train_claims=(), dev_claims=dev_claims())
    build_info = SimpleNamespace(
        cache_used=True,
        embedding_dimension=4,
        max_sequence_length=512,
        truncated_document_count=0,
        corpus_fingerprint="fingerprint",
    )
    monkeypatch.setattr(module, "OllamaChatClient", _Client)
    monkeypatch.setattr(module, "load_scifact_dataset", lambda _: dataset)
    monkeypatch.setattr(module, "validate_scifact_dataset", lambda _: None)
    monkeypatch.setattr(
        module, "build_dense_retriever", lambda *_, **__: (_Retriever(document), build_info)
    )
    monkeypatch.setattr("sys.argv", [script, "--output-dir", str(output_dir), *flags])
    return module.main()


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_evaluates_a_prefix_of_the_official_dev_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str
) -> None:
    exit_code = _run(monkeypatch, script, tmp_path, "--evaluate", "--split", "dev", "--limit", "3")

    assert exit_code == 0
    run_dir = next(tmp_path.iterdir())
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [str(record["claim_id"]) for record in records] == ["1", "10", "100"]
    assert [record["gold_label"] for record in records] == [
        "SUPPORT",
        "CONTRADICT",
        "CONTRADICT",
    ]
    assert metadata["resume_configuration"]["evaluation"]["mode"] == "partial_official_dev"
    assert metadata["resume_configuration"]["evaluation"]["selected_claim_ids"] == [
        "1",
        "10",
        "100",
    ]
    if script == "run_argus.py":
        assert metadata["evaluation"]["source"] == OFFICIAL_DEV_SOURCE
    else:
        assert metadata["evaluation"]["split"] == {"source": OFFICIAL_DEV_SOURCE}
        assert metadata["evaluation"]["full_official_dev_claim_count"] == 300
        assert "full_validation_claim_count" not in metadata["evaluation"]
        assert metadata["run"]["full_official_dev_claim_count"] == 300
        assert metadata["run"]["is_full_official_dev"] is False
        assert "is_full_internal_validation" not in metadata["run"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_rejects_dev_split_outside_evaluate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str
) -> None:
    try:
        exit_code = _run(monkeypatch, script, tmp_path, "--smoke-test", "--split", "dev")
    except SystemExit as stop:  # argparse rejection in the RAG baselines
        exit_code = stop.code
    assert exit_code == 2
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
