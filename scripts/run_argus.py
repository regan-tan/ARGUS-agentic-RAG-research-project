"""Run bounded Experiment 3 ARGUS or its controlled no-Challenge ablation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from argus.agents.argus import (
    ArgusAgent,
    ArgusConfig,
    ArgusResult,
    evaluate_argus_attempts,
    prompt_fingerprints,
)
from argus.data.scifact import load_scifact_dataset
from argus.data.validation import validate_scifact_dataset
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
from argus.retrieval import DEFAULT_QUERY_PREFIX, build_dense_retriever


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--claim")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-challenge", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-evidence-documents", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=12)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/retrieval_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/argus"))
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def update_run_state(
    metadata: dict[str, object],
    run_dir: Path,
    claim_ids: list[int | str],
    attempts: dict[str, ArgusResult],
    status: str,
) -> None:
    metadata["run"] = {
        "status": status,
        "selected_claim_count": len(claim_ids),
        "completed_claim_count": len(attempts),
        "completed_claim_ids": [claim_id for claim_id in claim_ids if str(claim_id) in attempts],
        "remaining_claim_ids": [
            claim_id for claim_id in claim_ids if str(claim_id) not in attempts
        ],
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    write_json(run_dir / "metadata.json", metadata)


def main() -> int:
    args = arguments()
    if args.limit is not None and args.limit <= 0:
        print("--limit must be positive", file=sys.stderr)
        return 2
    config = ArgusConfig(
        retrieval_k=args.k,
        max_evidence_documents=args.max_evidence_documents,
        max_llm_calls=args.max_llm_calls,
        challenge_enabled=not args.no_challenge,
        context_window_tokens=args.num_ctx,
    )
    client = OllamaChatClient(
        model=args.model,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        top_k=args.top_k,
    )
    try:
        identity = client.model_identity()
        dataset = load_scifact_dataset(args.data_dir)
        validate_scifact_dataset(dataset)
        retriever, build_info = build_dense_retriever(
            dataset.corpus, cache_dir=args.cache_dir, query_prefix=DEFAULT_QUERY_PREFIX
        )
    except (OllamaError, ValueError, OSError) as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 1
    agent = ArgusAgent(retriever, client, config)
    metadata = {
        "experiment": "Experiment 3 ARGUS",
        "variant": "full" if config.challenge_enabled else "no_challenge",
        "seed": args.seed,
        "agent": config.to_dict(),
        "prompts": prompt_fingerprints(),
        "retrieval": {
            "method": "dense BGE-small plus FAISS IndexFlatIP",
            "k": args.k,
            "query_prefix": DEFAULT_QUERY_PREFIX,
            "corpus_fingerprint": build_info.corpus_fingerprint,
        },
        "generator": {
            "model": args.model,
            "ollama": identity.to_dict(),
            "temperature": 0,
            "top_k": args.top_k,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
        "context_policy": (
            "Complete abstracts only; calls reported at or above num_ctx are rejected."
        ),
    }
    if args.claim:
        if args.resume is not None:
            print("--resume is available only with --evaluate", file=sys.stderr)
            return 2
        run_dir = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        result = agent.verify(args.claim)
        with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            append_prediction_record(handle, result.to_dict())
        metadata["mode"] = "single_claim"
        write_json(run_dir / "metadata.json", metadata)
        print(f"status={result.status} prediction={result.prediction} artifacts={run_dir}")
        return 0 if result.status == "success" else 1
    validation_claims = select_internal_validation_claims(dataset.train_claims, seed=args.seed)
    try:
        assert_expected_internal_validation(validation_claims)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.smoke_test:
        if args.limit is not None:
            print("--smoke-test selects exactly five claims; do not pass --limit", file=sys.stderr)
            return 2
        validation_ids = {str(claim.claim_id) for claim in validation_claims}
        development_claims = tuple(
            claim for claim in dataset.train_claims if str(claim.claim_id) not in validation_ids
        )
        selected = select_smoke_claims(development_claims, limit=5)
        evaluation_mode = "internal_development_smoke_test"
    else:
        selected = validation_claims[: args.limit] if args.limit else validation_claims
        evaluation_mode = "partial_internal_validation" if args.limit else "internal_validation"
    claim_ids = [claim.claim_id for claim in selected]
    expected_configuration = {
        "schema_version": 1,
        "variant": metadata["variant"],
        "evaluation": {
            "mode": evaluation_mode,
            "seed": args.seed,
            "selected_claim_ids": [str(value) for value in claim_ids],
        },
        "agent": metadata["agent"],
        "prompts": metadata["prompts"],
        "retrieval": metadata["retrieval"],
        "generator": {
            "model": args.model,
            "model_digest": identity.digest,
            "top_k": args.top_k,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
    }
    attempts: dict[str, ArgusResult] = {}
    if args.resume is not None:
        run_dir = args.resume
        try:
            metadata = load_json_object(run_dir / "metadata.json")
            ensure_resume_compatible(metadata, expected_configuration)
            records = load_prediction_records(run_dir / "predictions.jsonl")
            expected_ids = {str(value) for value in claim_ids}
            unexpected = sorted(set(records).difference(expected_ids))
            if unexpected:
                raise ArtifactError(
                    f"Prediction records contain unexpected claim IDs: {unexpected}"
                )
            attempts = {key: ArgusResult.from_dict(value) for key, value in records.items()}
        except (ArtifactError, ValueError) as error:
            print(f"Cannot resume {run_dir}: {error}", file=sys.stderr)
            return 1
        events = metadata.setdefault("resume_events", [])
        if isinstance(events, list):
            events.append({"resumed_at_utc": datetime.now(UTC).isoformat()})
    else:
        run_dir = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata["selected_claim_ids"] = claim_ids
        metadata["evaluation"] = {
            "mode": evaluation_mode,
            "source": "train-derived internal development"
            if args.smoke_test
            else "seed-42 validation",
        }
        metadata["resume_configuration"] = expected_configuration
        update_run_state(metadata, run_dir, claim_ids, attempts, "in_progress")
    predictions_path = run_dir / "predictions.jsonl"
    try:
        with predictions_path.open("a", encoding="utf-8") as handle:
            for claim in selected:
                if str(claim.claim_id) in attempts:
                    continue
                result = agent.verify(claim.claim, claim_id=claim.claim_id)
                attempts[str(claim.claim_id)] = result
                record = result.to_dict()
                gold = derive_gold_verdict(claim)
                record.update(gold_label=gold.verdict, gold_label_policy=gold.policy)
                append_prediction_record(handle, record)
                update_run_state(metadata, run_dir, claim_ids, attempts, "in_progress")
    except KeyboardInterrupt:
        update_run_state(metadata, run_dir, claim_ids, attempts, "partial")
        write_json(run_dir / "metrics.json", evaluate_argus_attempts(selected, attempts))
        print(f"interrupted; resume with --resume {run_dir}", file=sys.stderr)
        return 130
    update_run_state(metadata, run_dir, claim_ids, attempts, "completed")
    write_json(run_dir / "metrics.json", evaluate_argus_attempts(selected, attempts))
    print(f"completed={len(selected)} artifacts={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
