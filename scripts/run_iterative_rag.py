"""Run or safely resume the bounded single-agent iterative RAG experiment (Milestone 4)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from argus.agents import (
    PROMPT_VERSION,
    IterativeRAGAgent,
    IterativeRAGConfig,
    IterativeRAGResult,
    evaluate_iterative_attempts,
    prompt_fingerprints,
    summarize_agent_costs,
)
from argus.data.models import Claim
from argus.data.scifact import SciFactDataError, load_scifact_dataset
from argus.data.validation import DatasetValidationError, validate_scifact_dataset
from argus.evaluation.verification import (
    assert_expected_internal_validation,
    derive_gold_verdict,
    select_internal_validation_claims,
    select_smoke_claims,
)
from argus.llm import DEFAULT_OLLAMA_ENDPOINT, DEFAULT_QWEN_MODEL, OllamaChatClient, OllamaError
from argus.rag.artifacts import (
    ArtifactError,
    append_prediction_record,
    ensure_resume_compatible,
    load_json_object,
    load_prediction_records,
)
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
        help="Run the deterministic five-claim, label-stratified internal smoke test.",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum evaluation claims; use only for a partial run."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an interrupted run directory after strict configuration validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5, help="Initial retrieval depth.")
    parser.add_argument("--followup-k", type=int, default=5, help="Depth of each later round.")
    parser.add_argument("--max-retrieval-rounds", type=int, default=3)
    parser.add_argument("--max-llm-calls", type=int, default=6)
    parser.add_argument("--max-evidence-documents", type=int, default=15)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/retrieval_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/iterative_rag"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=8_192)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def select_claims(
    validation_claims: Sequence[Claim], *, smoke_test: bool, limit: int | None
) -> tuple[tuple[Claim, ...], str]:
    """Select exactly the claims Standard RAG uses for the same mode and limit."""

    if smoke_test:
        return select_smoke_claims(validation_claims, limit=5), "smoke_test"
    if limit is not None:
        return tuple(validation_claims[:limit]), "partial_internal_validation"
    return tuple(validation_claims), "internal_validation"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _run_directory(base: Path) -> tuple[str, Path]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = base / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return run_id, directory


def _artifact_record(
    result: IterativeRAGResult, gold_label: object, gold_policy: str | None
) -> dict[str, object]:
    """Gold fields are attached after the agent finishes, for scoring only."""

    record = result.to_dict()
    record["gold_label"] = gold_label
    record["gold_label_policy"] = gold_policy
    return record


def _print_result(result: IterativeRAGResult) -> None:
    print(f"claim_id={result.claim_id!r} status={result.status} prediction={result.prediction}")
    print(
        f"stop_reason={result.stop_reason} retrieval_calls={result.retrieval_calls} "
        f"llm_calls={result.llm_calls} evidence_documents={len(result.evidence_doc_ids)}"
    )
    for event in result.trace:
        if event.get("type") == "retrieval":
            print(
                f"  round {event.get('round')}: query={event.get('query')!r} "
                f"admitted={event.get('admitted_doc_ids')} "
                f"duplicates={event.get('duplicate_doc_ids')}"
            )
        elif event.get("purpose") == "verdict":
            parsed = event.get("parsed")
            outcome = (
                f"verdict={parsed.get('verdict')} citations={len(parsed.get('evidence', []))}"
                if isinstance(parsed, dict)
                else f"rejected ({event.get('error') or 'no valid response'})"
            )
            print(f"  llm verdict: {outcome}")
        else:
            parsed = event.get("parsed")
            if parsed is None:
                parsed = {"rejected": event.get("error") or "no valid response"}
            print(f"  llm {event.get('purpose')}: {json.dumps(parsed)}")
    if result.response:
        print(f"rationale: {result.response.rationale}")
        print(
            "evidence: "
            + json.dumps([reference.to_dict() for reference in result.response.evidence])
        )
    if result.failure_message:
        print(f"failure: {result.failure_message}", file=sys.stderr)
    print(f"tokens={json.dumps(result.token_usage(), sort_keys=True)}")
    print(f"end_to_end_ms={result.end_to_end_latency_seconds * 1_000:.2f}")


def _resume_configuration(
    *,
    mode: str,
    claims: Sequence[Claim],
    seed: int,
    retrieval: Mapping[str, object],
    generator: Mapping[str, object],
    agent: Mapping[str, object],
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
                "initial_k",
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
        },
        "agent": dict(agent),
    }


def _update_run_state(
    metadata: dict[str, object],
    run_dir: Path,
    claims: Sequence[Claim],
    attempts: Mapping[str, IterativeRAGResult],
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
    attempts: Mapping[str, IterativeRAGResult],
    state: Mapping[str, object],
) -> dict[str, object]:
    attempted_claims = tuple(claim for claim in claims if str(claim.claim_id) in attempts)
    metrics = evaluate_iterative_attempts(attempted_claims, attempts)
    metrics["run"] = {
        **state,
        "metrics_claim_count": len(attempted_claims),
        "note": (
            "Metrics cover persisted attempts only. Recorded failures remain in those "
            "denominators; claims not yet attempted are reported through run coverage, "
            "not scored as failures."
        ),
    }
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 2
    if args.smoke_test and args.limit is not None:
        print("--smoke-test always uses five claims; do not pass --limit", file=sys.stderr)
        return 2
    if args.resume is not None and args.claim is not None:
        print("--resume is available only for evaluation and smoke-test runs", file=sys.stderr)
        return 2
    try:
        config = IterativeRAGConfig(
            initial_k=args.k,
            followup_k=args.followup_k,
            max_retrieval_rounds=args.max_retrieval_rounds,
            max_llm_calls=args.max_llm_calls,
            max_evidence_documents=args.max_evidence_documents,
            context_window_tokens=args.num_ctx,
        )
    except ValueError as error:
        print(f"Invalid agent configuration: {error}", file=sys.stderr)
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
    except Exception as error:  # CLI boundary: report construction failure and exit
        print(f"Dense retriever construction failed: {error}", file=sys.stderr)
        return 1

    agent = IterativeRAGAgent(retriever, client, config)
    agent_configuration: dict[str, object] = {
        "workflow": "single-agent iterative RAG (LangGraph)",
        "budgets": config.to_dict(),
        "prompts": prompt_fingerprints(),
        "prompt_version": PROMPT_VERSION,
        "deduplication": "by canonical original SciFact document ID",
        "context_policy": (
            "Complete SciFact abstracts; no application-side truncation. A call whose prompt "
            "reaches num_ctx is rejected as context_limit_exceeded."
        ),
    }
    metadata: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": "AllenAI SciFact official release (local corpus.jsonl)",
        "dataset_counts": {
            "corpus_abstracts": len(dataset.corpus),
            "train_claims": len(dataset.train_claims),
        },
        "retrieval": {
            "method": "dense BGE-small plus FAISS IndexFlatIP",
            "initial_k": args.k,
            "followup_k": args.followup_k,
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
        },
        "agent": agent_configuration,
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
        result = agent.verify(args.claim)
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            append_prediction_record(handle, _artifact_record(result, None, None))
        _write_json(
            run_dir / "metrics.json",
            {
                "mode": "single_claim",
                "status": result.status,
                "efficiency": summarize_agent_costs((result,)),
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
    claims, mode = select_claims(validation_claims, smoke_test=args.smoke_test, limit=args.limit)
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
        agent=agent_configuration,
    )

    attempts: dict[str, IterativeRAGResult] = {}
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
                claim_id: IterativeRAGResult.from_dict(record)
                for claim_id, record in records.items()
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
                result = agent.verify(claim.claim, claim_id=claim.claim_id)
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
                    f"stop={result.stop_reason} rounds={result.retrieval_calls} "
                    f"llm_calls={result.llm_calls} "
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
        f"accuracy={float(cast(float, classification['accuracy'])):.3f} "
        f"macro_f1={float(cast(float, classification['macro_f1'])):.3f} "
        f"valid_responses={citations['valid_structured_responses']}/{len(attempts)}"
    )
    print(
        f"retrieval_calls={efficiency['retrieval_calls']} llm_calls={efficiency['llm_calls']} "
        f"stop_reasons={json.dumps(efficiency['stop_reasons'])}"
    )
    print(f"artifacts={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
