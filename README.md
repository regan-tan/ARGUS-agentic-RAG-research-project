# ARGUS

ARGUS (Adversarial Retrieval and Grounded Understanding System) is an SMU IS469 research project investigating whether adversarial multi-agent RAG improves scientific claim verification over standard and single-agent RAG.

Milestone 3 adds the reproducible Standard RAG baseline: one dense retrieval over SciFact, one local
Qwen inference call, and strict structured-output/citation validation. It does not implement agents,
iterative retrieval, fine-tuning, APIs, or a frontend.

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

## Standard RAG baseline

The Standard RAG path is deliberately non-agentic:

```text
claim -> one dense BGE/FAISS retrieval (k=5) -> one Qwen/Ollama call -> validation
```

It uses the local Ollama tag `qwen3:4b-instruct-2507-q4_K_M`, the Q4_K_M hardware-oriented
quantization of `Qwen/Qwen3-4B-Instruct-2507`. The quantized local model is recorded with its
available Ollama digest and runtime version in each run's metadata; it is not claimed to be
numerically identical to the BF16 checkpoint. Default decoding is fixed at temperature 0, top-k 40,
512 generated tokens, and an 8,192-token context window. The application passes complete original
abstracts for all five retrieved documents without evidence-text truncation.

The local Ollama version used here does not expose a standalone tokenizer endpoint. After each call,
ARGUS records Ollama's measured prompt-token count and rejects a result if it reaches the configured
context window, rather than accepting a potentially truncated-evidence verdict. This is a conservative
overflow guard; it does not truncate abstracts or issue an additional model call.

Install and start Ollama separately, then check the exact local tag before running:

```cmd
ollama list
uv run python scripts/run_standard_rag.py --claim "Example scientific claim"
```

The command verifies the local service and exact model tag before loading data. A successful result
contains one of `SUPPORT`, `CONTRADICT`, or `INSUFFICIENT`, a concise rationale, and citations of the
form `{"doc_id": 123, "sentence_indices": [0, 2]}`. IDs must be among the retrieved documents and
sentence indices are zero-based. A structurally valid citation only establishes that the sentence
exists; it is not by itself proof of scientific support.

Run the deterministic, label-stratified five-claim smoke test (selected from the existing seed-42
internal validation subset without using model predictions):

```cmd
uv run python scripts/run_standard_rag.py --smoke-test
```

Run a small controlled partial evaluation or the entire 162-claim internal validation set explicitly:

```cmd
uv run python scripts/run_standard_rag.py --evaluate --limit 5
uv run python scripts/run_standard_rag.py --evaluate
```

Evaluation predictions are append-only, flushed to disk after each completed claim, and new runs
never overwrite an older experiment. If an evaluation is interrupted, resume its exact run directory:

```cmd
uv run python scripts/run_standard_rag.py --evaluate --resume outputs/rag/<run-id>
```

Resume is rejected unless the model digest, generation settings, prompt hash, retrieval configuration,
validation seed, and selected claim IDs exactly match. Existing success and failure records are both
preserved and never rerun automatically; this avoids duplicate predictions and prevents silently
discarding failed attempts. Artifact metadata exposes run status and completed/remaining claim IDs;
partial-run metrics cover only persisted attempts and report coverage separately.

Every run writes a new ignored directory under `outputs/rag/` containing incrementally written
per-claim predictions, aggregate accuracy/macro-F1/per-class metrics, a confusion matrix, citation
overlap statistics, failure counts, measured latency/token usage, and reproducibility metadata.
Malformed output, invalid citations, retrieval errors, and inference errors are retained as failures;
they are never converted into `INSUFFICIENT`. The internal split is for development only; the official
300-claim development set remains reserved for final evaluation.

The preserved historical retrieval results are summarized in
[`reports/retrieval/milestone2-summary.md`](reports/retrieval/milestone2-summary.md).

## Limitations and next milestone

CPU-only Qwen inference can be slow, local quantization can affect output, a fixed context window can
become limiting for unusually long retrieved evidence, and retrieval misses remain an upper bound on
grounded verdict quality. SciFact's evidence annotations are benchmark labels rather than exhaustive
scientific truth. Temperature zero and pinned settings improve reproducibility but do not guarantee
identical output across Ollama/runtime/hardware versions. The next milestone is Single-agent
Iterative RAG, which must be evaluated against this one-pass baseline rather than altering it.
