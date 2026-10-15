# ARGUS

ARGUS (Adversarial Retrieval and Grounded Understanding System) is an SMU IS469 research project investigating whether adversarial multi-agent RAG improves scientific claim verification over standard and single-agent RAG.

This milestone establishes the reproducible SciFact data pipeline only. It does not implement retrieval, embeddings, agents, LLM calls, fine-tuning, APIs, or a frontend.

## Planned comparisons

- **A — Standard RAG:** retrieve once, then produce an evidence-grounded verdict.
- **B — Single-agent RAG:** iteratively retrieve, rewrite queries, inspect evidence, and stop conditionally.
- **C — ARGUS:** Support and Challenge agents, followed by an Evidence Verifier and Judge; a Challenge-agent ablation is planned.
- **D — ARGUS + LoRA/QLoRA:** optional fine-tuned evidence verifier.

The present stack is Python 3.12, uv, pytest, Ruff, and the standard library. Planned later components include BM25, dense retrieval, FAISS, LangGraph, and optional Hugging Face tooling.

## Prerequisites and installation (Windows CMD)

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then from the repository root run:

```cmd
uv sync
```

`uv` creates the local environment, resolves the development tools, and uses the committed `uv.lock` for reproducibility.

## SciFact data

ARGUS uses the original [AllenAI SciFact repository](https://github.com/allenai/scifact), its [official data schema](https://github.com/allenai/scifact/blob/master/doc/data.md), and the official archive:

```text
https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
```

Download and safely extract the archive into `data/scifact/`:

```cmd
uv run python scripts/download_scifact.py
```

The script does not re-download a valid existing dataset. It writes `data/scifact/PROVENANCE.json` with the source URL, UTC download time, and SHA-256 of the downloaded archive. The source URL intentionally points to `latest`, so it is not an immutable dataset version; retain that provenance file for a particular experimental run. Raw data and archives are ignored by Git.

Expected original JSONL files and split roles are:

| File | Expected records | Intended use |
| --- | ---: | --- |
| `corpus.jsonl` | 5,183 abstracts | Retrieval corpus for all experiments |
| `claims_train.jsonl` | 809 claims | Future development/training only |
| `claims_dev.jsonl` | 300 claims | Final evaluation; do not use its labels for tuning |
| `claims_test.jsonl` | 300 claims | Unlabeled held-out claims |

The loader preserves document and claim IDs, ordered abstract sentences, all evidence documents, and all rationale annotations. An empty training/development evidence mapping means no annotated evidence for that claim; it does not prove that no relevant scientific evidence exists. The test split is represented as unlabeled, never with fabricated labels.

## Inspect and validate data

```cmd
uv run python scripts/inspect_scifact.py
uv run python scripts/inspect_scifact.py --data-dir data\scifact
```

The inspection command checks official split sizes, IDs, references, labels, and rationale sentence indices. Its routine statistics intentionally use the training split only, not development-set label distributions.

## Quality checks

```cmd
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

## Repository layout

```text
configs/                 Future experiment configuration files
data/scifact/            Local official SciFact JSONL files (Git-ignored)
scripts/                 Dataset acquisition and inspection commands
src/argus/data/          SciFact models, loading, and validation
tests/                   Fast synthetic-fixture tests
outputs/                 Generated experiment artifacts (Git-ignored)
```

Future `src/argus/retrieval`, `agents`, `verifier`, and `evaluation` modules will be added only when their implementations begin.

## Status and next milestone

Implemented: project tooling, safe official-data acquisition, lossless SciFact loading, dataset validation, inspection, and automated tests.

Next: benchmark BM25 and dense retrieval over the corpus, reporting Recall@k before adding RAG or agent orchestration.
