"""CPU FAISS dense retrieval and compatibility-checked embedding caching."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from argus.data.models import CorpusDocument

from .types import RetrievalResult, canonical_document_id, document_text

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_FULL_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class DenseIndexBuildInfo:
    """One-time construction information, separate from warm-query latency."""

    cache_used: bool
    elapsed_seconds: float
    document_encoding_seconds: float
    index_construction_seconds: float
    embedding_dimension: int
    max_sequence_length: int | None
    truncated_document_count: int | None
    corpus_fingerprint: str


def _corpus_fingerprint(documents: Sequence[CorpusDocument]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(canonical_document_id(document.doc_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(document_text(document).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embeddings must not contain zero vectors")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class DenseRetriever:
    """Exact cosine-similarity search using normalized vectors and IndexFlatIP."""

    def __init__(
        self,
        documents: Sequence[CorpusDocument],
        embeddings: np.ndarray,
        query_encoder: Callable[[str], np.ndarray],
    ) -> None:
        self.documents = tuple(documents)
        document_ids = [canonical_document_id(document.doc_id) for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Dense documents must have unique document IDs")
        normalized = _normalize(embeddings)
        if len(self.documents) != normalized.shape[0]:
            raise ValueError("Embedding rows must match the number of documents")
        self._query_encoder = query_encoder
        self.dimension = int(normalized.shape[1])
        self._index = faiss.IndexFlatIP(self.dimension)
        self._index.add(normalized)

    @classmethod
    def from_embeddings(
        cls,
        documents: Sequence[CorpusDocument],
        embeddings: np.ndarray,
        query_encoder: Callable[[str], np.ndarray],
    ) -> DenseRetriever:
        """Build a retriever from supplied vectors, mainly for deterministic tests."""

        return cls(documents, embeddings, query_encoder)

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        """Encode a claim and return exact cosine-similarity results."""

        if k <= 0 or not query.strip() or not self.documents:
            return ()
        vector = np.asarray(self._query_encoder(query), dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (1, self.dimension):
            raise ValueError(
                f"Query embedding must have shape (1, {self.dimension}), got {vector.shape}"
            )
        vector = _normalize(vector)
        scores, indices = self._index.search(vector, min(k, len(self.documents)))
        results = [
            RetrievalResult(
                doc_id=self.documents[int(index)].doc_id,
                score=float(score),
                rank=rank,
                document=self.documents[int(index)],
            )
            for rank, (score, index) in enumerate(zip(scores[0], indices[0], strict=True), start=1)
            if index >= 0
        ]
        results.sort(key=lambda result: (-result.score, canonical_document_id(result.doc_id)))
        return tuple(
            RetrievalResult(result.doc_id, result.score, rank, result.document)
            for rank, result in enumerate(results, start=1)
        )


def _cache_paths(cache_dir: Path, fingerprint: str) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bge_embeddings_{fingerprint[:16]}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _cache_metadata(
    documents: Sequence[CorpusDocument],
    model_name: str,
    model_revision: str | None,
    query_prefix: str,
    dimension: int | None = None,
    max_sequence_length: int | None = None,
    truncated_document_count: int | None = None,
) -> dict[str, Any]:
    return {
        "cache_schema_version": 1,
        "corpus_fingerprint": _corpus_fingerprint(documents),
        "document_ids": [document.doc_id for document in documents],
        "model_name": model_name,
        "model_revision": model_revision,
        "query_prefix": query_prefix,
        "document_normalized": True,
        "dtype": "float32",
        "embedding_dimension": dimension,
        "max_sequence_length": max_sequence_length,
        "truncated_document_count": truncated_document_count,
    }


def _is_commit_sha(value: object) -> bool:
    return isinstance(value, str) and bool(_FULL_COMMIT_SHA.fullmatch(value))


def _load_cached_embeddings(
    vector_path: Path, metadata_path: Path, expected: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not vector_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not _is_commit_sha(expected.get("model_revision")):
            return None
        if not _is_commit_sha(metadata.get("model_revision")):
            return None
        compatibility_keys = (
            "cache_schema_version",
            "corpus_fingerprint",
            "document_ids",
            "model_name",
            "model_revision",
            "query_prefix",
            "document_normalized",
            "dtype",
            "max_sequence_length",
        )
        if any(metadata.get(key) != expected[key] for key in compatibility_keys):
            return None
        vectors = np.load(vector_path, allow_pickle=False)
        if vectors.dtype != np.float32 or vectors.shape != (
            len(expected["document_ids"]),
            metadata.get("embedding_dimension"),
        ):
            return None
        return vectors, metadata
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def build_dense_retriever(
    documents: Sequence[CorpusDocument],
    *,
    cache_dir: Path,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
    batch_size: int = 32,
    rebuild_cache: bool = False,
) -> tuple[DenseRetriever, DenseIndexBuildInfo]:
    """Load compatible cached document vectors or encode them once on CPU.

    The cache stores vectors and compatibility metadata only. Live corpus records remain
    authoritative for result metadata and ID mapping.
    """

    if not documents:
        raise ValueError("Cannot build dense retrieval for an empty corpus")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    started = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DEFAULT_MODEL_NAME, revision=DEFAULT_MODEL_REVISION, device="cpu")
    max_sequence_length = int(model.max_seq_length) if model.max_seq_length else None
    expected = _cache_metadata(
        documents,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
        query_prefix,
        max_sequence_length=max_sequence_length,
    )
    vector_path, metadata_path = _cache_paths(cache_dir, expected["corpus_fingerprint"])
    cached = (
        None if rebuild_cache else _load_cached_embeddings(vector_path, metadata_path, expected)
    )

    def encode_query(query: str) -> np.ndarray:
        return np.asarray(
            model.encode(
                [f"{query_prefix}{query}"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    if cached is not None:
        vectors, metadata = cached
        index_started = time.perf_counter()
        retriever = DenseRetriever(documents, vectors, encode_query)
        index_seconds = time.perf_counter() - index_started
        return retriever, DenseIndexBuildInfo(
            cache_used=True,
            elapsed_seconds=time.perf_counter() - started,
            document_encoding_seconds=0.0,
            index_construction_seconds=index_seconds,
            embedding_dimension=retriever.dimension,
            max_sequence_length=metadata.get("max_sequence_length"),
            truncated_document_count=metadata.get("truncated_document_count"),
            corpus_fingerprint=expected["corpus_fingerprint"],
        )

    texts = [document_text(document) for document in documents]
    tokenized = model.tokenizer(texts, add_special_tokens=True, truncation=False)
    truncated_document_count = (
        sum(len(ids) > max_sequence_length for ids in tokenized["input_ids"])
        if max_sequence_length is not None
        else None
    )
    encoding_started = time.perf_counter()
    vectors = np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    encoding_seconds = time.perf_counter() - encoding_started
    index_started = time.perf_counter()
    retriever = DenseRetriever(documents, vectors, encode_query)
    index_seconds = time.perf_counter() - index_started
    metadata = _cache_metadata(
        documents,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
        query_prefix,
        dimension=retriever.dimension,
        max_sequence_length=max_sequence_length,
        truncated_document_count=truncated_document_count,
    )
    np.save(vector_path, _normalize(vectors), allow_pickle=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return retriever, DenseIndexBuildInfo(
        cache_used=False,
        elapsed_seconds=time.perf_counter() - started,
        document_encoding_seconds=encoding_seconds,
        index_construction_seconds=index_seconds,
        embedding_dimension=retriever.dimension,
        max_sequence_length=max_sequence_length,
        truncated_document_count=truncated_document_count,
        corpus_fingerprint=expected["corpus_fingerprint"],
    )
