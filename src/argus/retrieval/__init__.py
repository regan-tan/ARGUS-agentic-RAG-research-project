"""Reusable document retrieval backends for ARGUS."""

from .bm25 import BM25Retriever, tokenize
from .dense import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_QUERY_PREFIX,
    DenseIndexBuildInfo,
    DenseRetriever,
    build_dense_retriever,
)
from .hybrid import HybridRetriever
from .types import RetrievalResult, document_text

__all__ = [
    "BM25Retriever",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_QUERY_PREFIX",
    "DenseIndexBuildInfo",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "build_dense_retriever",
    "document_text",
    "tokenize",
]
