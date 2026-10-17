"""Run or safely resume the one-pass Qwen Standard RAG baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from argus.data.models import Claim, DocumentId
from argus.data.scifact import SciFactDataError, load_scifact_dataset
from argus.data.validation import DatasetValidationError, validate_scifact_dataset
from argus.evaluation.verification import (
    assert_expected_internal_validation,
    derive_gold_verdict,
    evaluate_verification_attempts,
    select_internal_validation_claims,
    select_smoke_claims,
)
from argus.llm import (
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_QWEN_MODEL,
    GenerationMetadata,
    OllamaChatClient,
    OllamaError,
)
from argus.rag import PROMPT_VERSION, StandardRAGPipeline, build_messages
from argus.rag.artifacts import (
    ArtifactError,
    append_prediction_record,
    ensure_resume_compatible,
    load_json_object,
    load_prediction_records,
)
from argus.rag.standard import EvidenceReference, StandardRAGResult, VerificationResponse
from argus.retrieval import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_QUERY_PREFIX,
    build_dense_retriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--claim", help="Run one unlabelled scientific claim.")
    mode.add_argument(
        "--evaluate", action="store_true", help="Evaluate internal validation claims."
    )
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a deterministic five-claim, label-stratified internal smoke test.",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum evaluation claims; use only for a partial run."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted evaluation directory after strict configuration validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/retrieval_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rag"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=8_192)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _run_directory(base: Path) -> tuple[str, Path]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = base / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return run_id, directory


def _artifact_record(
    result: StandardRAGResult, gold_label: object, gold_policy: str | None
) -> dict[str, object]:
    record = result.to_dict()
    record["gold_label"] = gold_label
    record["gold_label_policy"] = gold_policy
    record["structural_response_valid"] = result.status == "success"
    record["structural_citations_valid"] = result.status == "success"
    return record


def _print_result(result: StandardRAGResult) -> None:
    print(f"claim_id={result.claim_id!r} status={result.status} prediction={result.prediction}")
    if result.response:
        print(f"rationale: {result.response.rationale}")
        print(
            "evidence: "
            + json.dumps([reference.to_dict() for reference in result.response.evidence])
        )
    if result.failure_message:
        print(f"failure: {result.failure_message}", file=sys.stderr)
    retrieval_ms = (
        None
        if result.retrieval_latency_seconds is None
        else result.retrieval_latency_seconds * 1_000
    )
    print(f"retrieval_ms={retrieval_ms}")
    print(f"end_to_end_ms={result.end_to_end_latency_seconds * 1_000:.2f}")
    if result.generation:
        print(f"generation_metadata={json.dumps(result.generation.to_dict(), sort_keys=True)}")


def _optional_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    raise ArtifactError("Prediction record has a non-numeric timing or token field")


def _record_to_attempt(record: Mapping[str, Any]) -> StandardRAGResult:
    """Rehydrate the fields evaluation needs from a previously flushed prediction record."""

    claim_id = record.get("claim_id")
    claim = record.get("claim")
    status = record.get("status")
    if (
        not isinstance(claim_id, (int, str))
        or isinstance(claim_id, bool)
        or not isinstance(claim, str)
        or not isinstance(status, str)
    ):
        raise ArtifactError("Prediction record is missing a valid claim_id, claim, or status")
    raw_generation = record.get("generation")
    generation = None
    if raw_generation is not None:
        if not isinstance(raw_generation, dict):
            raise ArtifactError("Prediction record generation must be an object or null")
        generation = GenerationMetadata(
            prompt_tokens=cast(int | None, _optional_number(raw_generation.get("prompt_tokens"))),
            completion_tokens=cast(
                int | None, _optional_number(raw_generation.get("completion_tokens"))
            ),
            total_duration_ns=cast(
                int | None, _optional_number(raw_generation.get("total_duration_ns"))
            ),
            load_duration_ns=cast(
                int | None, _optional_number(raw_generation.get("load_duration_ns"))
            ),
            prompt_eval_duration_ns=cast(
                int | None, _optional_number(raw_generation.get("prompt_eval_duration_ns"))
            ),
            eval_duration_ns=cast(
                int | None, _optional_number(raw_generation.get("eval_duration_ns"))
            ),
            wall_clock_seconds=float(
                _optional_number(raw_generation.get("wall_clock_seconds")) or 0
            ),
        )
    response = None
    raw_response = record.get("response")
    if raw_response is not None:
        if not isinstance(raw_response, dict):
            raise ArtifactError("Prediction record response must be an object or null")
        verdict = raw_response.get("verdict")
        rationale = raw_response.get("rationale")
        evidence = raw_response.get("evidence")
        if verdict not in {"SUPPORT", "CONTRADICT", "INSUFFICIENT"} or not isinstance(
            rationale, str
        ):
            raise ArtifactError("Prediction record has an invalid structured verdict")
        if not isinstance(evidence, list):
            raise ArtifactError("Prediction record evidence must be a list")
        references: list[EvidenceReference] = []
        for item in evidence:
            if not isinstance(item, dict):
                raise ArtifactError("Prediction record evidence item must be an object")
            doc_id = item.get("doc_id")
            indices = item.get("sentence_indices")
            if (
                not isinstance(doc_id, (int, str))
                or isinstance(doc_id, bool)
                or not isinstance(indices, list)
                or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
            ):
                raise ArtifactError("Prediction record has invalid evidence references")
            references.append(EvidenceReference(cast(DocumentId, doc_id), tuple(indices)))
        response = VerificationResponse(cast(Any, verdict), rationale, tuple(references))
    retrieval_latency = _optional_number(record.get("retrieval_latency_seconds"))
    end_to_end = _optional_number(record.get("end_to_end_latency_seconds"))
    if end_to_end is None:
        raise ArtifactError("Prediction record has no end_to_end_latency_seconds")
    return StandardRAGResult(
        claim_id=claim_id,
        claim=claim,
        retrieved=(),
        retrieval_latency_seconds=float(retrieval_latency)
        if retrieval_latency is not None
        else None,
        generation=generation,
        end_to_end_latency_seconds=float(end_to_end),
        status=cast(Any, status),
        response=response,
        raw_response=record.get("raw_response")
        if isinstance(record.get("raw_response"), str)
        else None,
        failure_message=(
            record.get("failure_message")
            if isinstance(record.get("failure_message"), str)
            else None
        ),
        retrieval_call_attempted=bool(
            record.get("retrieval_call_attempted", generation is not None)
        ),
        model_call_attempted=bool(record.get("model_call_attempted", generation is not None)),
    )


def _resume_configuration(
    *,
    mode: str,
    claims: Sequence[Claim],
    seed: int,
    retrieval: Mapping[str, object],
    generator: Mapping[str, object],
) -> dict[str, object]:
    ollama = generator["ollama"]
    if not isinstance(ollama, Mapping):
        raise AssertionError("Ollama identity must be a mapping")
    return {
        "schema_version": 1,
        "evaluation": {
            "mode": mode,
            "seed": seed,
            "selected_claim_ids": [str(claim.claim_id) for claim in claims],
        },
        "retrieval": {
            key: retrieval[key]
            for key in (
                "method",
                "k",
                "model",
                "model_revision",
                "query_prefix",
                "corpus_fingerprint",
            )
        },
        "generator": {
            "model": generator["model"],
            "model_digest": ollama.get("digest"),
            "temperature": generator["temperature"],
            "top_k": generator["top_k"],
            "num_predict": generator["num_predict"],
            "num_ctx": generator["num_ctx"],
            "prompt_version": generator["prompt_version"],
            "system_prompt_sha256": generator["system_prompt_sha256"],
        },
    }


def _update_run_state(
    metadata: dict[str, object],
    run_dir: Path,
    claims: Sequence[Claim],
    attempts: Mapping[str, StandardRAGResult],
    *,
    status: str,
    full_validation_count: int,
) -> dict[str, object]:
    state: dict[str, object] = {
        "status": status,
        "selected_claim_count": len(claims),
        "completed_claim_count": len(attempts),
        "completed_claim_ids": [
            claim.claim_id for claim in claims if str(claim.claim_id) in attempts
        ],
        "remaining_claim_ids": [
            claim.claim_id for claim in claims if str(claim.claim_id) not in attempts
        ],
        "full_internal_validation_claim_count": full_validation_count,
        "is_full_internal_validation": len(claims) == full_validation_count,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    metadata["run"] = state
    _write_json(run_dir / "metadata.json", metadata)
    return state


def _write_metrics(
    run_dir: Path,
    claims: Sequence[Claim],
    attempts: Mapping[str, StandardRAGResult],
    state: Mapping[str, object],
) -> dict[str, object]:
    attempted_claims = tuple(claim for claim in claims if str(claim.claim_id) in attempts)
    metrics = evaluate_verification_attempts(attempted_claims, attempts)
    metrics["run"] = {
        **state,
        "metrics_claim_count": len(attempted_claims),
        "note": (
            "Metrics cover persisted attempts only. Recorded failures remain in those "
            "denominators; "
            "claims not yet attempted are reported through run coverage, not scored as failures."
        ),
    }
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 2
    if args.k <= 0:
        print("--k must be positive", file=sys.stderr)
        return 2
    if args.smoke_test and args.limit is not None:
        print("--smoke-test always uses five claims; do not pass --limit", file=sys.stderr)
        return 2
    if args.resume is not None and args.claim is not None:
        print("--resume is available only for evaluation and smoke-test runs", file=sys.stderr)
        return 2

    client = OllamaChatClient(
        model=args.model,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        top_k=args.top_k,
    )
    try:
        model_identity = client.model_identity()
    except OllamaError as error:
        print(f"Ollama preflight failed: {error}", file=sys.stderr)
        return 1
    try:
        dataset = load_scifact_dataset(args.data_dir)
        validate_scifact_dataset(dataset)
    except (SciFactDataError, DatasetValidationError) as error:
        print(f"SciFact loading or validation failed: {error}", file=sys.stderr)
        return 1
    try:
        retriever, build_info = build_dense_retriever(
            dataset.corpus,
            cache_dir=args.cache_dir,
            query_prefix=DEFAULT_QUERY_PREFIX,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"Dense retriever construction failed: {error}", file=sys.stderr)
        return 1

    pipeline = StandardRAGPipeline(
        retriever,
        client,
        k=args.k,
        context_window_tokens=args.num_ctx,
    )
    sample_system, _ = build_messages("prompt fingerprint sample", ())
    prompt_hash = hashlib.sha256(sample_system.encode("utf-8")).hexdigest()
    metadata: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": "AllenAI SciFact official release (local corpus.jsonl)",
        "dataset_counts": {
            "corpus_abstracts": len(dataset.corpus),
            "train_claims": len(dataset.train_claims),
        },
        "retrieval": {
            "method": "dense BGE-small plus FAISS IndexFlatIP",
            "k": args.k,
            "model": DEFAULT_MODEL_NAME,
            "model_revision": DEFAULT_MODEL_REVISION,
            "query_prefix": DEFAULT_QUERY_PREFIX,
            "embedding_cache_used": build_info.cache_used,
            "embedding_dimension": build_info.embedding_dimension,
            "embedding_max_sequence_length": build_info.max_sequence_length,
            "embedding_truncated_document_count": build_info.truncated_document_count,
            "corpus_fingerprint": build_info.corpus_fingerprint,
        },
        "generator": {
            "model": args.model,
            "ollama": model_identity.to_dict(),
            "quantization": "Q4_K_M (local Ollama model tag; not numerically identical to BF16)",
            "temperature": 0.0,
            "top_k": args.top_k,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "stream": False,
            "prompt_version": PROMPT_VERSION,
            "system_prompt_sha256": prompt_hash,
            "context_policy": (
                "Complete retrieved SciFact abstracts; no application-side evidence truncation."
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
    }

    if args.claim is not None:
        run_id, run_dir = _run_directory(args.output_dir)
        metadata["run_id"] = run_id
        _write_json(run_dir / "metadata.json", metadata)
        result = pipeline.verify(args.claim)
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            append_prediction_record(handle, _artifact_record(result, None, None))
        _write_json(
            run_dir / "metrics.json",
            {
                "mode": "single_claim",
                "status": result.status,
                "note": "No gold label is available for an arbitrary single claim.",
            },
        )
        _print_result(result)
        print(f"artifacts={run_dir}")
        return 0 if result.status == "success" else 1

    validation_claims = select_internal_validation_claims(dataset.train_claims, seed=args.seed)
    try:
        assert_expected_internal_validation(validation_claims)
    except ValueError as error:
        print(f"Internal validation configuration failed: {error}", file=sys.stderr)
        return 1
    if args.smoke_test:
        claims = select_smoke_claims(validation_claims, limit=5)
        mode = "smoke_test"
    else:
        claims = validation_claims[: args.limit] if args.limit is not None else validation_claims
        mode = "internal_validation" if args.limit is None else "partial_internal_validation"
    metadata["evaluation"] = {
        "mode": mode,
        "split": {"source": "Milestone 2 train-derived validation", "seed": args.seed},
        "full_validation_claim_count": len(validation_claims),
        "selected_claim_ids": [claim.claim_id for claim in claims],
        "mixed_label_policy": (
            "Exclude mixed evidence labels from three-class metrics; preserve them in artifacts."
        ),
    }
    expected_configuration = _resume_configuration(
        mode=mode,
        claims=claims,
        seed=args.seed,
        retrieval=cast(Mapping[str, object], metadata["retrieval"]),
        generator=cast(Mapping[str, object], metadata["generator"]),
    )

    attempts: dict[str, StandardRAGResult] = {}
    if args.resume is not None:
        run_dir = args.resume
        try:
            metadata = load_json_object(run_dir / "metadata.json")
            ensure_resume_compatible(metadata, expected_configuration)
            records = load_prediction_records(run_dir / "predictions.jsonl")
            expected_ids = {str(claim.claim_id) for claim in claims}
            unexpected = sorted(set(records).difference(expected_ids))
            if unexpected:
                raise ArtifactError(
                    f"Prediction artifact contains claim IDs outside this run: {unexpected}"
                )
            attempts = {
                claim_id: _record_to_attempt(record) for claim_id, record in records.items()
            }
        except ArtifactError as error:
            print(f"Cannot resume {run_dir}: {error}", file=sys.stderr)
            return 1
        history = metadata.setdefault("resume_events", [])
        if isinstance(history, list):
            history.append({"resumed_at_utc": datetime.now(UTC).isoformat()})
        print(f"resuming={run_dir} completed_claims={len(attempts)}/{len(claims)}")
    else:
        run_id, run_dir = _run_directory(args.output_dir)
        metadata["run_id"] = run_id
        metadata["resume_configuration"] = expected_configuration
        _update_run_state(
            metadata,
            run_dir,
            claims,
            attempts,
            status="in_progress",
            full_validation_count=len(validation_claims),
        )

    predictions_path = run_dir / "predictions.jsonl"
    try:
        with predictions_path.open("a", encoding="utf-8") as handle:
            for claim in claims:
                if str(claim.claim_id) in attempts:
                    continue
                result = pipeline.verify(claim.claim, claim_id=claim.claim_id)
                attempts[str(claim.claim_id)] = result
                gold = derive_gold_verdict(claim)
                append_prediction_record(
                    handle, _artifact_record(result, gold.verdict, gold.policy)
                )
                _update_run_state(
                    metadata,
                    run_dir,
                    claims,
                    attempts,
                    status="in_progress",
                    full_validation_count=len(validation_claims),
                )
                print(
                    f"[{len(attempts)}/{len(claims)}] claim_id={claim.claim_id} "
                    f"gold={gold.verdict} predicted={result.prediction} status={result.status} "
                    f"end_to_end_s={result.end_to_end_latency_seconds:.2f}"
                )
    except KeyboardInterrupt:
        state = _update_run_state(
            metadata,
            run_dir,
            claims,
            attempts,
            status="partial",
            full_validation_count=len(validation_claims),
        )
        _write_metrics(run_dir, claims, attempts, state)
        print(f"interrupted; resume with --resume {run_dir}", file=sys.stderr)
        return 130

    state = _update_run_state(
        metadata,
        run_dir,
        claims,
        attempts,
        status="completed",
        full_validation_count=len(validation_claims),
    )
    metrics = _write_metrics(run_dir, claims, attempts, state)
    classification = cast(Mapping[str, object], metrics["classification"])
    efficiency = cast(Mapping[str, object], metrics["efficiency"])
    citations = cast(Mapping[str, object], metrics["citations"])
    print(
        f"accuracy={float(classification['accuracy']):.3f} "
        f"macro_f1={float(classification['macro_f1']):.3f} "
        f"valid_responses={citations['valid_structured_responses']}/{len(attempts)}"
    )
    latency = cast(Mapping[str, object], efficiency["end_to_end_latency"])
    print(f"mean_end_to_end_ms={latency['mean_ms']}")
    print(f"artifacts={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
