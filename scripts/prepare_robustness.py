"""Prepare and audit reproducible retrieval-noise manifests without any LLM calls.

This is a CPU preflight for the robustness protocol, NOT an evaluation of verdicts.
Only training-derived development claims are selected; internal validation and the
official labeled development set are left unused for model/prompt tuning.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from argus.data.scifact import load_scifact_dataset
from argus.data.validation import validate_scifact_dataset
from argus.evaluation.robustness import NoiseConfig, NoisyRetriever
from argus.evaluation.verification import (
    select_internal_validation_claims,
    select_smoke_claims,
)
from argus.retrieval import DEFAULT_QUERY_PREFIX, build_dense_retriever


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--noise-levels", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/retrieval_cache"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/robustness/preflight")
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.limit <= 0 or args.k <= 0 or args.candidate_pool <= args.k:
        raise ValueError("--limit and --k must be positive; candidate pool must exceed k")
    levels = sorted(set(args.noise_levels))
    if not levels or levels[0] != 0 or levels[-1] > args.k or levels[0] < 0:
        raise ValueError("--noise-levels must include 0 and remain between 0 and k")

    dataset = load_scifact_dataset(args.data_dir)
    validate_scifact_dataset(dataset)
    internal_validation = select_internal_validation_claims(dataset.train_claims, seed=args.seed)
    excluded_ids = {str(claim.claim_id) for claim in internal_validation}
    development = tuple(
        claim for claim in dataset.train_claims if str(claim.claim_id) not in excluded_ids
    )
    selected = select_smoke_claims(development, limit=args.limit)
    retriever, build_info = build_dense_retriever(
        dataset.corpus, cache_dir=args.cache_dir, query_prefix=DEFAULT_QUERY_PREFIX
    )

    records: list[dict[str, object]] = []
    summary: dict[str, dict[str, int]] = {}
    for level in levels:
        delivered = underdelivered = 0
        for claim in selected:
            assert claim.evidence is not None, "development split cannot include unlabeled claims"
            noisy = NoisyRetriever(
                retriever,
                claim_id=claim.claim_id,
                config=NoiseConfig(
                    replacements=level, seed=args.seed, candidate_pool=args.candidate_pool
                ),
                protected_doc_ids=tuple(claim.evidence),
            )
            actual = noisy.retrieve(claim.claim, args.k)
            event = noisy.events[0]
            if level == 0 and actual != tuple(retriever.retrieve(claim.claim, args.k)):
                raise RuntimeError("Clean control no longer equals the original retriever")
            delivered += len(event.replacements)
            underdelivered += len(event.replacements) < level
            records.append({"noise_level": level, **event.to_dict()})
        summary[str(level)] = {
            "claims": len(selected),
            "requested_replacements": level * len(selected),
            "realized_replacements": delivered,
            "claims_below_requested_noise": underdelivered,
        }

    output = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "perturbations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "experiment": "retrieval-noise CPU preflight only",
        "claim_source": "training-derived development, excluding internal validation",
        "selected_claim_ids": [str(claim.claim_id) for claim in selected],
        "seed": args.seed,
        "k": args.k,
        "candidate_pool": args.candidate_pool,
        "noise_levels": levels,
        "noise_policy": (
            "Replace non-annotated top-k slots with lower-ranked same-query SciFact documents; "
            "protect already retrieved annotated documents. No semantic negative labels asserted."
        ),
        "retrieval_backend": "BGE-small/FAISS dense, unmodified",
        "corpus_fingerprint": build_info.corpus_fingerprint,
        "summary_by_level": summary,
        "no_llm_inference": True,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(records)} retrieval events without LLM calls at {output}")
    for level, info in summary.items():
        print(
            f"noise={level}: {info['realized_replacements']}/"
            f"{info['requested_replacements']} replacements; "
            f"{info['claims_below_requested_noise']} under-delivered claims"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
