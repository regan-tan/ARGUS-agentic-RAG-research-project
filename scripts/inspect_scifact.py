"""Inspect and validate a local official SciFact dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from argus.data.scifact import SciFactDataError, load_scifact_dataset
from argus.data.validation import (
    DatasetValidationError,
    training_statistics,
    validate_scifact_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and inspect official SciFact JSONL files."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset = load_scifact_dataset(args.data_dir)
        report = validate_scifact_dataset(dataset)
    except (SciFactDataError, DatasetValidationError) as error:
        print(f"SciFact validation failed: {error}", file=sys.stderr)
        return 1

    print("SciFact validation passed.")
    print(f"Corpus abstracts: {len(dataset.corpus)}")
    print(f"Training claims: {len(dataset.train_claims)}")
    print(f"Development claims: {len(dataset.dev_claims)}")
    print(f"Test claims (unlabeled): {len(dataset.test_claims)}")
    print("Training evidence statistics:")
    for name, value in training_statistics(dataset.train_claims).items():
        print(f"  {name.replace('_', ' ')}: {value}")
    print(f"Evidence label conflicts retained for review: {len(report.label_conflicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
