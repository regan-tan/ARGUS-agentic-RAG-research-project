"""Benchmark BM25, dense FAISS, and hybrid RRF retrieval on internal SciFact validation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from argus.data.scifact import SciFactDataError, load_scifact_dataset
from argus.data.validation import DatasetValidationError, validate_scifact_dataset
from argus.evaluation.benchmark import (
    per_claim_payload,
    qualitative_examples,
    retrieve_and_measure,
    summarize_method,
)
from argus.evaluation.retrieval import make_internal_split
from argus.retrieval import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_QUERY_PREFIX,
    BM25Retriever,
    HybridRetriever,
    build_dense_retriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("bm25", "dense", "hybrid"),
        default=("bm25", "dense", "hybrid"),
        help="Retrieval methods to evaluate (default: all).",
    )
    parser.add_argument("--k", nargs="+", type=int, default=(3, 5, 10))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/retrieval_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rrf-candidate-pool", type=int, default=50)
    parser.add_argument("--rrf-constant", type=int, default=60)
    return parser.parse_args()


def package_versions() -> dict[str, str | None]:
    packages = ("faiss-cpu", "numpy", "rank-bm25", "sentence-transformers")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def write_json(path: Path, content: object) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.k or any(k <= 0 for k in args.k):
        print("--k must contain positive values", file=sys.stderr)
        return 2
    try:
        dataset = load_scifact_dataset(args.data_dir)
        validate_scifact_dataset(dataset)
    except (SciFactDataError, DatasetValidationError) as error:
        print(f"SciFact loading or validation failed: {error}", file=sys.stderr)
        return 1

    split = make_internal_split(dataset.train_claims, seed=args.seed)
    validation_ids = {str(claim_id) for claim_id in split.validation_claim_ids}
    claims = tuple(claim for claim in dataset.train_claims if str(claim.claim_id) in validation_ids)
    max_k = max(args.k)
    requested = tuple(dict.fromkeys(args.methods))
    retrievers: dict[str, object] = {}
    construction: dict[str, object] = {}

    bm25_started = time.perf_counter()
    bm25 = BM25Retriever(dataset.corpus)
    construction["bm25"] = {"elapsed_seconds": time.perf_counter() - bm25_started}
    if "bm25" in requested or "hybrid" in requested:
        retrievers["bm25"] = bm25

    dense = None
    if "dense" in requested or "hybrid" in requested:
        try:
            dense, dense_info = build_dense_retriever(
                dataset.corpus,
                cache_dir=args.cache_dir,
                query_prefix=DEFAULT_QUERY_PREFIX,
                batch_size=args.batch_size,
                rebuild_cache=args.rebuild_cache,
            )
        except (
            Exception
        ) as error:  # Preserve the precise external model/download error for the user.
            print(f"Dense retriever construction failed: {error}", file=sys.stderr)
            return 1
        construction["dense"] = {
            "cache_used": dense_info.cache_used,
            "elapsed_seconds": dense_info.elapsed_seconds,
            "document_encoding_seconds": dense_info.document_encoding_seconds,
            "index_construction_seconds": dense_info.index_construction_seconds,
            "embedding_dimension": dense_info.embedding_dimension,
            "max_sequence_length": dense_info.max_sequence_length,
            "truncated_document_count": dense_info.truncated_document_count,
            "corpus_fingerprint": dense_info.corpus_fingerprint,
        }
        if "dense" in requested:
            retrievers["dense"] = dense
    if "hybrid" in requested:
        if dense is None:
            raise AssertionError("Dense retriever is required for hybrid retrieval")
        retrievers["hybrid"] = HybridRetriever(
            bm25,
            dense,
            candidate_pool=args.rrf_candidate_pool,
            rrf_constant=args.rrf_constant,
        )

    predictions: dict[str, object] = {}
    metrics: dict[str, object] = {}
    latency: dict[str, object] = {}
    for method, retriever in retrievers.items():
        method_predictions, method_latency = retrieve_and_measure(retriever, claims, max_k)
        predictions[method] = method_predictions
        metrics[method] = summarize_method(claims, method_predictions, args.k)
        latency[method] = method_latency

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": "AllenAI SciFact official release (local corpus.jsonl)",
        "corpus_fingerprint": construction.get("dense", {}).get("corpus_fingerprint"),
        "corpus_abstract_count": len(dataset.corpus),
        "train_claim_count": len(dataset.train_claims),
        "internal_validation_claim_count": len(claims),
        "internal_development_claim_count": len(split.development_claim_ids),
        "validation_claim_ids": list(split.validation_claim_ids),
        "split": {"seed": split.seed, "validation_fraction": split.validation_fraction},
        "retrieval": {
            "document_representation": "title plus full ordered abstract",
            "bm25": {"tokenizer": "lowercase regex \\b\\w+\\b", "algorithm": "BM25Okapi"},
            "dense": {
                "model": DEFAULT_MODEL_NAME,
                "model_revision": DEFAULT_MODEL_REVISION,
                "query_prefix": DEFAULT_QUERY_PREFIX,
                "index": "faiss.IndexFlatIP with normalized float32 embeddings",
            },
            "hybrid": {
                "candidate_pool": args.rrf_candidate_pool,
                "rrf_constant": args.rrf_constant,
            },
            "k_values": sorted(set(args.k)),
        },
        "construction": construction,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": package_versions(),
        },
    }
    write_json(run_dir / "metrics.json", {"metrics": metrics, "latency_ms": latency})
    write_json(run_dir / "metadata.json", metadata)
    for method, method_predictions in predictions.items():
        records = per_claim_payload(claims, method_predictions, args.k)
        with (run_dir / f"predictions_{method}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if {"bm25", "dense", "hybrid"}.issubset(predictions):
        write_json(
            run_dir / "qualitative_examples.json",
            qualitative_examples(
                claims,
                predictions,
                {str(document.doc_id): document.title for document in dataset.corpus},
                max_k,
            ),
        )

    print(f"Results written to {run_dir}")
    print("Method   Recall@3  Recall@5  Recall@10  Hit@5  Median latency (ms)")
    for method in requested:
        overall = metrics[method]["overall"]
        recall = overall["recall_at_k"]
        hit = overall["hit_at_k"]
        print(
            f"{method:<8} {recall.get('3', 0.0):>8.3f}  {recall.get('5', 0.0):>8.3f}"
            f"  {recall.get('10', 0.0):>9.3f}  {hit.get('5', 0.0):>5.3f}"
            f"  {latency[method]['median_ms']:>19.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
