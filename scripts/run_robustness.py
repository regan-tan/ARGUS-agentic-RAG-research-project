"""Run failure-aware, controlled retrieval-noise comparisons on frozen RAG systems.

Run each system/noise level separately, in a GPU allocation with Ollama available.
Use --pilot N on training-derived development claims to validate the protocol.
Only use --evaluate (162 validation claims) after the protocol is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.agents.argus import ArgusResult
from argus.agents.argus import prompt_fingerprints as argus_prompts
from argus.agents.iterative import IterativeRAGResult
from argus.agents.iterative import prompt_fingerprints as iterative_prompts
from argus.data.scifact import load_scifact_dataset
from argus.data.validation import validate_scifact_dataset
from argus.evaluation.robustness import NoiseConfig
from argus.evaluation.robustness_runner import (
    SYSTEMS,
    Attempt,
    SystemName,
    evaluate_noisy_attempts,
    run_noisy_claim,
    summarize_noise_events,
)
from argus.evaluation.verification import (
    assert_expected_internal_validation,
    derive_gold_verdict,
    select_internal_validation_claims,
    select_smoke_claims,
)
from argus.llm import DEFAULT_OLLAMA_ENDPOINT, DEFAULT_QWEN_MODEL, OllamaChatClient
from argus.rag.artifacts import (
    ArtifactError,
    append_prediction_record,
    ensure_resume_compatible,
    load_json_object,
    load_prediction_records,
)
from argus.rag.standard import build_messages
from argus.retrieval import DEFAULT_QUERY_PREFIX, build_dense_retriever

# The Standard RAG script already implements strict decoding of append-only records.
# Reuse it rather than silently dropping token/timing fields during resume.
from run_standard_rag import _record_to_attempt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", type=int, metavar="N")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--noise-level", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/retrieval_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robustness/runs"))
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def commit_identity() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def fingerprints(system: SystemName) -> dict[str, object]:
    if system == "standard":
        prompt = build_messages("claim", ())[0]
        return {"verdict": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
    if system == "iterative":
        return iterative_prompts()
    return argus_prompts()


def select_claims(
    train_claims: Any, *, seed: int, pilot: int | None
) -> tuple[tuple[Any, ...], str]:
    validation = select_internal_validation_claims(train_claims, seed=seed)
    assert_expected_internal_validation(validation)
    if pilot is None:
        return validation, "internal_validation_162"
    if pilot < 3:
        raise ValueError("--pilot must select at least 3 claims for label stratification")
    validation_ids = {str(claim.claim_id) for claim in validation}
    development = tuple(
        claim for claim in train_claims if str(claim.claim_id) not in validation_ids
    )
    if pilot > len(development):
        raise ValueError(f"--pilot exceeds available development claims ({len(development)})")
    return select_smoke_claims(development, limit=pilot), "train_development_pilot"


def decode_attempt(system: SystemName, record: dict[str, Any]) -> Attempt:
    if system == "standard":
        return _record_to_attempt(record)
    if system == "iterative":
        return IterativeRAGResult.from_dict(record)
    return ArgusResult.from_dict(record)


def score(
    system: SystemName,
    claims: tuple[Any, ...],
    records: dict[str, dict[str, Any]],
) -> dict[str, object]:
    attempts = {key: decode_attempt(system, value) for key, value in records.items()}
    metrics = evaluate_noisy_attempts(system, claims, attempts)
    metrics["retrieval_noise"] = summarize_noise_events(records)
    metrics["run_coverage"] = {
        "selected_claims": len(claims),
        "completed_claims": len(records),
        "is_complete": len(records) == len(claims),
    }
    return metrics


def main() -> int:
    args = parse_args()
    if args.k <= 0 or args.candidate_pool <= args.k:
        print("Require positive --k and --candidate-pool greater than --k", file=sys.stderr)
        return 2
    if args.pilot is not None and args.pilot < 3:
        print("--pilot must be at least 3", file=sys.stderr)
        return 2
    system: SystemName = args.system
    noise = NoiseConfig(
        replacements=args.noise_level,
        seed=args.seed,
        candidate_pool=args.candidate_pool,
    )

    # The same Ollama model digest, CPU FAISS index and claim IDs should be recorded
    # for all conditions. Do not substitute the official dev labels here.
    try:
        client = OllamaChatClient(
            model=args.model,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
            num_predict=args.num_predict,
            num_ctx=args.num_ctx,
            top_k=args.top_k,
        )
        identity = client.model_identity()
        dataset = load_scifact_dataset(args.data_dir)
        validate_scifact_dataset(dataset)
        claims, mode = select_claims(
            dataset.train_claims, seed=args.seed, pilot=args.pilot
        )
        retriever, build_info = build_dense_retriever(
            dataset.corpus, cache_dir=args.cache_dir, query_prefix=DEFAULT_QUERY_PREFIX
        )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Robustness preflight failed: {error}", file=sys.stderr)
        return 1

    commit = commit_identity()
    config = {
        "schema_version": 1,
        "experiment": "argus-retrieval-noise-v1",
        "system": system,
        "dataset": "AllenAI SciFact official release; train-derived only",
        "mode": mode,
        "seed": args.seed,
        "claim_ids": [str(claim.claim_id) for claim in claims],
        "git_commit": commit,
        "noise": {
            "replacements_per_retrieval_call": args.noise_level,
            "candidate_pool": args.candidate_pool,
            "preserve_annotated_gold": True,
            "selection": "ranked same-query lower candidates, deterministic per claim/query/call",
        },
        "retrieval": {
            "method": "dense BGE-small plus FAISS IndexFlatIP",
            "k": args.k,
            "query_prefix": DEFAULT_QUERY_PREFIX,
            "corpus_fingerprint": build_info.corpus_fingerprint,
        },
        "generation": {
            "model": args.model,
            "model_digest": identity.digest,
            "ollama": identity.to_dict(),
            "temperature": 0,
            "top_k": args.top_k,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
        "prompts": fingerprints(system),
        "agent_budgets": (
            {"llm_calls": 1, "retrieval_calls": 1}
            if system == "standard"
            else {
                "max_retrieval_rounds": 3,
                "max_llm_calls": 6,
                "max_evidence_documents": max(15, args.k),
                "followup_k": args.k,
            }
            if system == "iterative"
            else {
                "max_llm_calls": 12,
                "max_evidence_documents": max(10, args.k),
                "challenge_enabled": system == "argus",
            }
        ),
    }
    metadata = {
        **config,
        "started_utc": datetime.now(UTC).isoformat(),
        "resume_configuration": config,
        "status": "in_progress",
        "completed_claim_count": 0,
    }
    if args.resume is not None:
        run_dir = args.resume
        try:
            previous = load_json_object(run_dir / "metadata.json")
            ensure_resume_compatible(previous, config)
            records = load_prediction_records(run_dir / "predictions.jsonl")
            allowed = set(config["claim_ids"])
            if not set(records).issubset(allowed):
                raise ArtifactError("Resume contains claim IDs absent from this run")
            # Rehydrate all historical rows before appending.
            for record in records.values():
                decode_attempt(system, record)
            metadata = previous
            metadata["resumed_utc"] = datetime.now(UTC).isoformat()
        except (ArtifactError, ValueError) as error:
            print(f"Refusing incompatible resume: {error}", file=sys.stderr)
            return 1
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = args.output_dir / f"{system}__noise{args.noise_level}__{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        records = {}
        write_json(run_dir / "metadata.json", metadata)

    def save_state(status: str) -> None:
        metadata["status"] = status
        metadata["completed_claim_count"] = len(records)
        metadata["updated_utc"] = datetime.now(UTC).isoformat()
        write_json(run_dir / "metadata.json", metadata)
        write_json(run_dir / "metrics.json", score(system, claims, records))

    try:
        with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            for claim in claims:
                key = str(claim.claim_id)
                if key in records:
                    continue
                attempt, events = run_noisy_claim(
                    system, retriever, client, claim, noise, k=args.k, num_ctx=args.num_ctx
                )
                record = attempt.to_dict()
                gold = derive_gold_verdict(claim)
                record["gold_label"] = gold.verdict
                record["gold_label_policy"] = gold.policy
                record["noise_events"] = list(events)
                append_prediction_record(handle, record)
                records[key] = record
                metadata["completed_claim_count"] = len(records)
                write_json(run_dir / "metadata.json", metadata)
                print(
                    f"{len(records)}/{len(claims)} claim={key} "
                    f"status={attempt.status} label={attempt.prediction}",
                    flush=True,
                )
    except KeyboardInterrupt:
        save_state("partial")
        print(f"Interrupted; resume with --resume {run_dir}", file=sys.stderr)
        return 130
    save_state("completed")
    print(f"completed={len(records)} system={system} noise={args.noise_level}")
    print(f"artifacts={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
