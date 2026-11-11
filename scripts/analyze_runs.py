"""Offline analysis of completed claim-verification runs: trajectory, cost, ranking, failures.

Reads existing ``predictions.jsonl`` files only; it makes no model or retrieval calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from argus.data.scifact import SciFactDataError, load_claims
from argus.evaluation.run_analysis import (
    FAILURE_CATEGORIES,
    SCORING_RULES,
    RunAnalysisError,
    analyze_run,
)

SPLIT_FILES = {"dev": "claims_dev.jsonl", "train": "claims_train.jsonl"}


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--run must look like NAME=RUN_DIRECTORY")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=parse_run, action="append", required=True)
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="dev")
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    return parser.parse_args()


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_tables(results: dict[str, dict], k: int) -> None:
    print("\nTrajectory")
    print(
        "| System | Steps/claim | LLM calls/claim | Retrievals/claim | LLM call success "
        "| Budget hit |"
    )
    for name, result in results.items():
        t = result["trajectory"]
        print(
            f"| {name} | {_fmt(t['mean_steps'], 2)} | {_fmt(t['mean_llm_calls'], 2)} | "
            f"{_fmt(t['mean_retrieval_calls'], 2)} | {_fmt(t['llm_call_success_rate'])} | "
            f"{_fmt(t['budget_exhausted_claims'])} |"
        )
    print("\nCost per correct claim")
    print("| System | Correct | Seconds/correct | Tokens/correct | LLM calls/correct |")
    for name, result in results.items():
        c = result["cost"]
        print(
            f"| {name} | {c['correct']}/{c['claims']} | {_fmt(c['seconds_per_correct'], 1)} | "
            f"{_fmt(c['tokens_per_correct'], 0)} | {_fmt(c['llm_calls_per_correct'], 2)} |"
        )
    print(f"\nRetrieval (first retrieval, k={k}; evidence-bearing claims)")
    print(f"| System | MRR@{k} | nDCG@{k} | Recall@{k} | Hit@{k} | Pool recall | Pool hit |")
    for name, result in results.items():
        r = result["retrieval"]
        print(
            f"| {name} | {_fmt(r[f'mrr_at_{k}'])} | {_fmt(r[f'ndcg_at_{k}'])} | "
            f"{_fmt(r[f'recall_at_{k}'])} | {_fmt(r[f'hit_at_{k}'])} | "
            f"{_fmt(r['evidence_pool_recall'])} | {_fmt(r['evidence_pool_hit'])} |"
        )
    print("\nFailure attribution (wrong claims)")
    print("| System | Wrong | " + " | ".join(FAILURE_CATEGORIES) + " |")
    for name, result in results.items():
        f = result["failures"]
        counts = " | ".join(str(f["by_category"][category]) for category in FAILURE_CATEGORIES)
        print(f"| {name} | {f['wrong_claims']} | {counts} |")

    print("\nVerifier on gold-annotated abstracts")
    print(
        "| System | Gold abstracts | Accuracy | SUPPORT recall | CONTRADICT recall "
        "| Direction flips |"
    )
    for name, result in results.items():
        v = result["verifier"]
        if v is None:
            continue
        recall = v["recall_on_gold_abstracts"]
        print(
            f"| {name} | {v['gold_abstracts']} | {_fmt(v['accuracy_on_gold_abstracts'])} | "
            f"{_fmt(recall['SUPPORT'])} | {_fmt(recall['CONTRADICT'])} | "
            f"{v['direction_flips_on_gold_abstracts']} |"
        )

    print("\nAccuracy under alternative scoring rules")
    print("| System | " + " | ".join(SCORING_RULES) + " |")
    for name, result in results.items():
        cells = " | ".join(
            _fmt(result["scoring_rules"][rule]["accuracy"]) for rule in SCORING_RULES
        )
        print(f"| {name} | {cells} |")

    print("\nChallenge-only abstracts")
    print("| System | Claims | Abstracts | Gold | Only Challenge found gold | Verifier labels |")
    for name, result in results.items():
        c = result["challenge_evidence"]
        if c is None:
            continue
        print(
            f"| {name} | {c['claims_with_challenge_only_abstracts']} | "
            f"{c['challenge_only_abstracts']} | {c['challenge_only_gold_abstracts']} | "
            f"{c['claims_where_only_challenge_found_gold']} | "
            f"{c['verifier_labels_on_challenge_only_abstracts']} |"
        )


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        print("--k must be positive", file=sys.stderr)
        return 2
    try:
        claims = load_claims(args.data_dir / SPLIT_FILES[args.split], labels_required=True)
        claims_by_id = {str(claim.claim_id): claim for claim in claims}
        results = {}
        sources = {}
        for name, run_dir in args.run:
            path = run_dir / "predictions.jsonl"
            with path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            results[name] = analyze_run(records, claims_by_id, k=args.k)
            sources[name] = str(path)
    except (OSError, json.JSONDecodeError, SciFactDataError, RunAnalysisError) as error:
        print(f"Run analysis failed: {error}", file=sys.stderr)
        return 1

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "run_id": run_id,
        "split": args.split,
        "k": args.k,
        "sources": sources,
        "note": "Offline analysis of saved predictions; no model or retrieval calls were made.",
        "results": results,
    }
    (output_dir / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_tables(results, args.k)
    print(f"\nWrote {output_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
