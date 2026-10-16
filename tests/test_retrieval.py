from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from argus.data.models import Claim, CorpusDocument, EvidenceAnnotation
from argus.evaluation.retrieval import make_internal_split, summarize_metrics
from argus.retrieval.bm25 import BM25Retriever, tokenize
from argus.retrieval.dense import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DenseRetriever,
    _cache_metadata,
    _load_cached_embeddings,
    build_dense_retriever,
)
from argus.retrieval.hybrid import HybridRetriever
from argus.retrieval.types import RetrievalResult


def document(doc_id: int, text: str) -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id, title=f"Title {doc_id}", abstract=(text,), structured=False
    )


def result(document_: CorpusDocument, rank: int) -> RetrievalResult:
    return RetrievalResult(document_.doc_id, 1.0 / rank, rank, document_)


def test_bm25_tokenization_ranking_and_top_k_behavior() -> None:
    documents = (
        document(2, "Cellular mitochondria generate energy"),
        document(1, "A bird sings"),
        document(3, "A fish swims"),
    )
    retriever = BM25Retriever(documents)

    assert tokenize("MiTo-CHONDRIA!") == ["mito", "chondria"]
    assert retriever.retrieve("MITOCHONDRIA", 10)[0].doc_id == 2
    assert [item.rank for item in retriever.retrieve("mitochondria", 10)] == [1, 2, 3]
    assert retriever.retrieve("   ", 3) == ()
    assert retriever.retrieve("mitochondria", 0) == ()


def test_bm25_rejects_duplicate_document_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever((document(1, "one"), document(1, "two")))


def test_bm25_ranking_is_independent_of_claim_gold_annotations() -> None:
    documents = (document(1, "mitochondria make energy"), document(2, "birds sing"))
    retriever = BM25Retriever(documents)
    plain_claim = Claim(1, "mitochondria energy", (), {})
    annotated_claim = Claim(
        2,
        "mitochondria energy",
        (2,),
        {"2": (EvidenceAnnotation("CONTRADICT", (0,)),)},
    )

    assert retriever.retrieve(plain_claim.claim, 2) == retriever.retrieve(annotated_claim.claim, 2)


def test_dense_retrieval_normalizes_vectors_and_preserves_document_mapping() -> None:
    documents = (document(31, "first"), document(7, "second"), document(22, "third"))

    def encode(_: str) -> np.ndarray:
        return np.array([3.0, 0.0], dtype=np.float32)

    retriever = DenseRetriever.from_embeddings(
        documents,
        np.array([[2.0, 0.0], [0.0, 7.0], [1.0, 1.0]], dtype=np.float32),
        encode,
    )
    results = retriever.retrieve("query", 10)

    assert retriever.dimension == 2
    assert [item.doc_id for item in results] == [31, 22, 7]
    assert results[0].score == pytest.approx(1.0)
    assert retriever.retrieve("", 2) == ()


def test_dense_retrieval_validates_dimensions() -> None:
    documents = (document(1, "one"),)
    with pytest.raises(ValueError, match="match"):
        DenseRetriever.from_embeddings(
            documents, np.ones((2, 2), dtype=np.float32), lambda _: np.ones(2)
        )

    retriever = DenseRetriever.from_embeddings(
        documents, np.ones((1, 2), dtype=np.float32), lambda _: np.ones(3)
    )
    with pytest.raises(ValueError, match="shape"):
        retriever.retrieve("query", 1)


def test_embedding_cache_rejects_incompatible_metadata(tmp_path: Path) -> None:
    documents = (document(1, "one"), document(2, "two"))
    metadata = _cache_metadata(
        documents,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
        "prefix",
        dimension=2,
        max_sequence_length=512,
    )
    vectors = tmp_path / "vectors.npy"
    metadata_path = tmp_path / "metadata.json"
    np.save(vectors, np.ones((2, 2), dtype=np.float32), allow_pickle=False)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert _load_cached_embeddings(
        vectors,
        metadata_path,
        _cache_metadata(
            documents, DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, "prefix", max_sequence_length=512
        ),
    )
    assert (
        _load_cached_embeddings(
            vectors,
            metadata_path,
            _cache_metadata(
                documents, "other", DEFAULT_MODEL_REVISION, "prefix", max_sequence_length=512
            ),
        )
        is None
    )
    assert (
        _load_cached_embeddings(
            vectors,
            metadata_path,
            _cache_metadata(
                documents,
                DEFAULT_MODEL_NAME,
                DEFAULT_MODEL_REVISION,
                "prefix",
                max_sequence_length=256,
            ),
        )
        is None
    )


def test_embedding_cache_rejects_missing_revision(tmp_path: Path) -> None:
    documents = (document(1, "one"),)
    metadata = _cache_metadata(
        documents, DEFAULT_MODEL_NAME, None, "prefix", dimension=2, max_sequence_length=512
    )
    vectors = tmp_path / "vectors.npy"
    metadata_path = tmp_path / "metadata.json"
    np.save(vectors, np.ones((1, 2), dtype=np.float32), allow_pickle=False)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert (
        _load_cached_embeddings(
            vectors,
            metadata_path,
            _cache_metadata(
                documents,
                DEFAULT_MODEL_NAME,
                DEFAULT_MODEL_REVISION,
                "prefix",
                max_sequence_length=512,
            ),
        )
        is None
    )


def test_dense_builder_passes_pinned_revision_to_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = (document(1, "one"),)
    metadata = _cache_metadata(
        documents,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
        "prefix",
        dimension=2,
        max_sequence_length=512,
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    vector_path = cache_dir / f"bge_embeddings_{metadata['corpus_fingerprint'][:16]}.npy"
    metadata_path = cache_dir / f"bge_embeddings_{metadata['corpus_fingerprint'][:16]}.json"
    np.save(vector_path, np.ones((1, 2), dtype=np.float32), allow_pickle=False)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls: list[tuple[str, str, str]] = []

    class FakeSentenceTransformer:
        max_seq_length = 512

        def __init__(self, model_name: str, *, revision: str, device: str) -> None:
            calls.append((model_name, revision, device))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    _, info = build_dense_retriever(documents, cache_dir=cache_dir, query_prefix="prefix")

    assert info.cache_used
    assert calls == [(DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, "cpu")]


def test_dense_builder_writes_pinned_revision_to_new_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = (document(1, "one"),)

    class FakeSentenceTransformer:
        max_seq_length = 512

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def tokenizer(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
            return {"input_ids": [[1] for _ in texts]}

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    _, info = build_dense_retriever(documents, cache_dir=tmp_path, query_prefix="prefix")
    metadata_path = tmp_path / f"bge_embeddings_{info.corpus_fingerprint[:16]}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert not info.cache_used
    assert metadata["model_revision"] == DEFAULT_MODEL_REVISION


class StaticRetriever:
    def __init__(self, ranking: tuple[RetrievalResult, ...]) -> None:
        self.ranking = ranking

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        return self.ranking[:k] if query else ()


def test_hybrid_rrf_deduplicates_and_uses_stable_ties() -> None:
    one, two, three = document(1, "one"), document(2, "two"), document(3, "three")
    retriever = HybridRetriever(
        StaticRetriever((result(one, 1), result(two, 2))),
        StaticRetriever((result(two, 1), result(three, 2))),
        candidate_pool=2,
        rrf_constant=60,
    )

    results = retriever.retrieve("claim", 10)
    assert [item.doc_id for item in results] == [2, 1, 3]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert retriever.retrieve("", 3) == ()


def claim(claim_id: int, evidence: dict[str, tuple[EvidenceAnnotation, ...]]) -> Claim:
    return Claim(claim_id, f"claim {claim_id}", (), evidence)


def test_metrics_deduplicate_gold_documents_and_exclude_no_evidence() -> None:
    first = claim(1, {"1": (EvidenceAnnotation("SUPPORT", (0,)),), "2": ()})
    second = claim(2, {})
    predictions = {"1": (result(document(2, "two"), 1), result(document(3, "three"), 2))}

    summary = summarize_metrics((first, second), predictions, (1, 2))
    assert summary.eligible_claims == 1
    assert summary.excluded_no_evidence_claims == 1
    assert summary.recall_at_k == {1: 0.5, 2: 0.5}
    assert summary.hit_at_k == {1: 1.0, 2: 1.0}


def test_internal_split_is_reproducible_and_stratified() -> None:
    claims = (
        claim(1, {"1": (EvidenceAnnotation("SUPPORT", (0,)),)}),
        claim(2, {"2": (EvidenceAnnotation("SUPPORT", (0,)),)}),
        claim(3, {"3": (EvidenceAnnotation("CONTRADICT", (0,)),)}),
        claim(4, {"4": (EvidenceAnnotation("CONTRADICT", (0,)),)}),
        claim(5, {}),
    )
    first = make_internal_split(claims, seed=42, validation_fraction=0.4)
    second = make_internal_split(tuple(reversed(claims)), seed=42, validation_fraction=0.4)

    assert first == second
    assert len(first.validation_claim_ids) == 2
    assert set(first.validation_claim_ids).isdisjoint(first.development_claim_ids)
