# ARGUS

ARGUS (Adversarial Retrieval and Grounded Understanding System) is an SMU IS469 research project investigating whether adversarial multi-agent RAG improves scientific claim verification over standard and single-agent RAG.

Milestone 2 provides the shared, reproducible scientific-evidence retrieval foundation. It does not implement LLM verdict generation, agents, fine-tuning, APIs, or a frontend.

## Retrieval backends

Each SciFact title plus its complete ordered abstract is one searchable document. This preserves original document IDs, titles, sentence order, and provenance while keeping document-level evidence evaluation straightforward. Abstracts are not chunked in this milestone.

- **BM25:** `rank-bm25` `BM25Okapi` with a lowercase deterministic regex tokenizer. It is a fast lexical baseline.
- **Dense:** `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` embeds documents and claims on CPU. Normalized float32 vectors are searched exactly with `faiss.IndexFlatIP`, so the score is cosine similarity. Claims receive BGE's retrieval query prefix; documents do not.
- **Hybrid:** BM25 and dense candidate rankings are fused with Reciprocal Rank Fusion (RRF), not raw-score addition. The fixed defaults are a top-50 candidate pool and `c=60`.

All backends expose `retrieve(query, k)` and return ranked `RetrievalResult` records containing the source document and its original ID. Empty queries and nonpositive `k` return no results; requests larger than the corpus are safely capped.

## Setup

ARGUS targets Python 3.12 on Windows and uses `uv`. Dense retrieval is CPU-compatible, but the initial model download and corpus embedding are materially slower than BM25.

```cmd
uv sync
```

## SciFact data

ARGUS uses the original [AllenAI SciFact repository](https://github.com/allenai/scifact), its [official data schema](https://github.com/allenai/scifact/blob/master/doc/data.md), and the official archive. Download it once into the ignored local directory:

```cmd
uv run python scripts/download_scifact.py
uv run python scripts/inspect_scifact.py
```

Expected data are 5,183 corpus abstracts, 809 training claims, 300 labeled development claims, and 300 unlabeled test claims. The labeled official development set remains reserved for later end-to-end evaluation; this retrieval benchmark does not use its labels for selection or tuning.

## Run the benchmark

```cmd
uv run python scripts/benchmark_retrieval.py --methods bm25 dense hybrid --k 3 5 10 --seed 42
```

Useful options are `--rebuild-cache`, `--batch-size 32`, `--cache-dir data/retrieval_cache`, and `--output-dir outputs/retrieval`. Run just the lexical baseline without downloading a model:

```cmd
uv run python scripts/benchmark_retrieval.py --methods bm25
```

Dense corpus embeddings are cached as a NumPy matrix plus JSON metadata in `data/retrieval_cache/` (ignored by Git). Cache compatibility checks include the corpus content fingerprint and ordered IDs, pinned model revision, query prefix, normalization, dtype, sequence limit, and cache schema. The original SciFact corpus remains authoritative for document metadata. To intentionally update the model, replace the single `DEFAULT_MODEL_REVISION` commit SHA in `src/argus/retrieval/dense.py` and rebuild the cache with `--rebuild-cache`.

The benchmark measures the number of input tokens versus the model's configured maximum sequence length and saves the count of documents that are truncated. It does not add chunking merely to avoid those truncations.

## Evaluation design

The 809 official training claims are deterministically split into approximately 80/20 internal-development/internal-validation subsets with seed 42. Claims are sorted by stable ID before seeded selection and stratified into support-only, contradict-only, mixed, and no-evidence categories. The full 5,183-document corpus remains searchable for every method.

For evidence-bearing validation claims, the benchmark reports:

- `Recall@k`: unique annotated gold documents in the top `k`, divided by unique annotated gold documents for that claim, averaged across eligible claims.
- `Hit@k`: whether any annotated gold document occurs in the top `k`, averaged across eligible claims.

Claims without annotated evidence are excluded from both denominators. This does not mean every retrieved non-gold abstract is false or irrelevant. Retrieval latency is measured after a warm call and includes dense query encoding plus FAISS search; document embedding/index construction is reported separately.

Each run creates an ignored timestamped directory under `outputs/retrieval/` containing `metrics.json`, `metadata.json`, method-specific per-claim JSONL rankings, and (when all three backends run) selected qualitative success/failure examples. Gold IDs occur only in these evaluation outputs, never in queries, indexes, or rankings.

## Quality checks

```cmd
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run python scripts/inspect_scifact.py
```

## Limitations and next milestone

This is a document-retrieval comparison on a modest internal validation subset, not a claim-verification result. SciFact annotations are not exhaustive relevance judgments; BGE input truncation and CPU model inference are practical trade-offs; and cache reproducibility depends on recording the chosen model identifier/revision. The next milestone is a Standard RAG baseline using the selected shared retrieval backend.
