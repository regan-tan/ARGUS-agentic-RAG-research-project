# ARGUS

ARGUS (Adversarial Retrieval and Grounded Understanding System) is an SMU IS469 research project investigating whether adversarial multi-agent RAG improves scientific claim verification over standard and single-agent RAG.

Milestone 3 adds the reproducible Standard RAG baseline: one dense retrieval over SciFact, one local
Qwen inference call, and strict structured-output/citation validation. Milestone 4 adds a separate,
bounded Single-agent Iterative RAG workflow on the same retrieval, model, and evaluation contracts,
leaving the baseline unchanged. Experiment 3 adds the bounded multi-agent ARGUS
Support/Challenge/Verifier/Judge system; fine-tuning, APIs, and a frontend are not implemented.

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

## Single-agent Iterative RAG

Milestone 4 adds one LangGraph agent (`src/argus/agents/iterative.py`) that can search again when its
evidence is inadequate:

```text
claim -> dense BGE/FAISS retrieval (k=5, identical to Standard RAG)
      -> Qwen evidence-sufficiency assessment
      -> if insufficient and budget remains: Qwen query rewrite -> retrieve again -> reassess
      -> stop when sufficient or a budget is exhausted
      -> Qwen verdict over all accumulated evidence -> validation
```

It reuses the same SciFact corpus, pinned BGE-small model, exact FAISS index, Ollama Qwen tag and
decoding settings, internal seed-42 validation claims, gold-label policy, and classification/citation
metric definitions as Standard RAG. The final verdict call reuses the Standard RAG prompt
(`standard-rag-v1`), JSON schema, and citation validator unchanged; citations are validated against
every document actually shown to that call. Only the sufficiency and query-rewrite prompts are new
(`iterative-rag-v3`). All prompt versions and system-prompt hashes are recorded in run metadata.
Prompt history from live smoke tests: v1 produced a verbose sufficiency answer that exceeded the
512-token output limit; v2 bounded answer length but asked for the decision before the reasoning,
and the assessor then never judged evidence sufficient (even when its own reason said the evidence
directly contradicted the claim). v3 keeps the length bounds, has the model reason before deciding,
and states explicitly that direct contradiction is sufficient evidence. These revisions were
informed by the five-claim `--smoke-test` claims, which belong to the 162-claim internal validation
set; no other evaluation claims and no official development labels were inspected.

Evidence accumulates across rounds, deduplicated by original SciFact document ID. Sufficiency is
judged by Qwen from abstract text only; retrieval scores are never shown to it. The agent receives
only claim text, never gold labels or annotated evidence IDs. One model call is always reserved for
the final verdict. Default budgets, all configurable:

| Option | Default | Meaning |
| --- | --- | --- |
| `--k` | 5 | Initial retrieval depth (same as Standard RAG) |
| `--followup-k` | 5 | Retrieval depth of each later round |
| `--max-retrieval-rounds` | 3 | Maximum retrieval calls per claim |
| `--max-llm-calls` | 6 | Maximum Qwen calls per claim, including sufficiency, rewrite, and verdict |
| `--max-evidence-documents` | 15 | Maximum distinct abstracts presented; excluded IDs are logged |

Each claim records a stop reason: `evidence_sufficient`, `max_retrieval_rounds`, `max_llm_calls`,
`max_evidence_documents` (the last three count as budget exhaustion), `no_new_evidence` (a follow-up
search found only already-seen documents), `repeated_query`, or `failure`.

LangGraph is the only added dependency (`uv add langgraph`, recorded in `pyproject.toml` and
`uv.lock`); it provides the stateful conditional routing and bounded loop. `uv sync` installs it.

Run one claim, the five-claim smoke test, a partial run, or the full 162-claim internal validation set:

```cmd
uv run python scripts/run_iterative_rag.py --claim "Example scientific claim"
uv run python scripts/run_iterative_rag.py --smoke-test
uv run python scripts/run_iterative_rag.py --evaluate --limit 5
uv run python scripts/run_iterative_rag.py --evaluate
```

Each run writes a new ignored directory under `outputs/iterative_rag/` (never `outputs/rag/`). Every
completed claim is appended to `predictions.jsonl` with its full agent trace: each retrieval round's
query, retrieved/admitted/duplicate/excluded document IDs and latency, and each model call's purpose,
prompt version, raw response, parsed decision, token counts, and latency. `metrics.json` contains the
shared classification, citation, and failure metrics plus agent costs computed from those traces:
retrieval rounds per claim, total retrieval and LLM calls (by purpose), prompt/completion tokens,
distinct evidence documents, latency, stop reasons, and budget-exhaustion frequency. Interrupted runs
resume exactly as Standard RAG does:

```cmd
uv run python scripts/run_iterative_rag.py --evaluate --resume outputs/iterative_rag/<run-id>
```

Resume is rejected unless the model digest, generation settings, retrieval configuration, agent
budgets, prompt hashes, seed, and selected claim IDs exactly match. Failed sufficiency, rewrite, or
verdict calls are recorded as failures without a prediction; they are never converted into
`INSUFFICIENT`.

Limitations: the agent makes up to six Qwen calls per claim, so CPU runs are several times slower
than Standard RAG and a full 162-claim evaluation is a multi-hour job. Accumulating up to 15 complete
abstracts can reach the 8,192-token context window; such calls are rejected as
`context_limit_exceeded` rather than truncated. Sufficiency judgments come from the same small
quantized model and are not calibrated confidence. No claim that iterative RAG outperforms Standard
RAG is made until both are evaluated on the same full claim set.

## Limitations and next milestone

## Experiment 3: ARGUS Support/Challenge verification

Experiment 3 adds a bounded LangGraph workflow for comparing full ARGUS with a controlled
no-Challenge ablation. It keeps the SciFact corpus, pinned dense BGE/FAISS retriever, Qwen/Ollama
model, seed-42 internal validation IDs, and failure-aware claim metrics used by the earlier runs.

```text
claim -> deterministic retrieval objective -> Support retrieval -> Challenge retrieval
      -> provenance-preserving merge -> independent per-document verifier -> Judge verdict
```

The verifier receives the claim and one complete abstract, but never the Support/Challenge source.
The judge receives only independently verified evidence and must pass the existing structured
citation validator. Retrieved papers are treated as untrusted data throughout. Evidence is merged
by original SciFact document ID, retaining `discovered_by` provenance for documents found by both
agents. The deterministic Challenge query explicitly targets exceptions, limitations, conflicting
findings, and alternative outcomes; it does not itself label a document as contradicting.

The checked-in [`configs/argus-v2.json`](configs/argus-v2.json) records conservative defaults for
the 8,192-token context window. Complete abstracts are never application-truncated. Ollama calls
reported at or above the configured context window are rejected, as are malformed verifier output,
out-of-range verifier citations, malformed judge output, and judge citations outside the verified
document set. These conditions remain recorded failures rather than being converted to an
`INSUFFICIENT` verdict.

ARGUS v2 addresses an A40/Ollama smoke-test failure in which Qwen produced repetitive verifier
rationale text until the fixed 512-token output budget ended before the JSON response completed.
Both ARGUS rationale fields now use a 300-character JSON Schema limit and explicitly request one
concise, non-repetitive sentence of at most 40 words. The ARGUS-only Judge schema also caps the
evidence array at the configured verified-document budget (10 by default), preventing repeated
citations from consuming the response budget; shared Standard RAG schemas and citation validation
remain unchanged. The model, temperature, `num_predict=512`, and `num_ctx=8192` are unchanged.

After preparing the local dataset, BGE cache, and Ollama model, run a single claim or a controlled
partial/full seed-42 evaluation:

```cmd
uv run python scripts/run_argus.py --claim "Example scientific claim"
uv run python scripts/run_argus.py --smoke-test
uv run python scripts/run_argus.py --smoke-test --no-challenge
uv run python scripts/run_argus.py --evaluate --limit 5
uv run python scripts/run_argus.py --evaluate
uv run python scripts/run_argus.py --evaluate --no-challenge
```

For the A40 preflight after this v2 update, do not resume an `argus-v1` directory. Run the two
five-claim variants into new output roots:

```cmd
uv run --locked python scripts/run_argus.py --smoke-test --output-dir outputs/argus-v2-smoke-full
uv run --locked python scripts/run_argus.py --smoke-test --no-challenge --output-dir outputs/argus-v2-smoke-no-challenge
```

`--smoke-test` always selects five label-stratified claims from the seed-42 train-derived
development pool, explicitly excluding all frozen 162 internal-validation claim IDs. It is the
only multi-claim command intended for real-model preflight before the controlled evaluation.

Evaluation predictions are append-only and fsync after every claim. Resume only the exact run
directory; the command validates the selected IDs, seed, ARGUS variant, model digest and decoding
settings, retriever/corpus fingerprint, agent budgets, and prompt hashes before accepting it:

```cmd
uv run python scripts/run_argus.py --evaluate --resume outputs/argus/<run-id>
```

Completed successes and failures are never rerun. Duplicate claim IDs and incompatible metadata are
rejected. A single unterminated final JSONL fragment (for example, from an abrupt process exit) is
ignored safely; any earlier malformed line is treated as corrupt and stops resume. `metrics.json` is
recomputed from the authoritative saved rows after a completion or interruption.

In addition to the unchanged failure-aware claim and citation metrics, ARGUS records post-inference
gold-document recall/hit rate after Support retrieval and after all agent retrieval, unique Challenge
documents and unique Challenge gold documents, verifier-label counts, merged-provenance counts,
per-agent retrieval calls, per-component LLM calls/tokens/latency, and failure categories. These
gold-based measurements are evaluation-only: neither retrieval agents, verifier nor Judge receive
gold labels or annotations. Verifier classification is currently whole-abstract with sentence
citations, not a sentence-rationale classifier; this limitation must be reported with the A40 run.

Each invocation creates a new ignored `outputs/argus/<UTC-run-id>/` directory with metadata,
per-claim traces/predictions, and evaluation metrics. Full GPU execution remains an explicit
cluster run: configure the existing Ollama endpoint and exact local Qwen tag on the A40, then use
the final command above for each variant. No GPU experiment is run by these repository checks.

CPU-only Qwen inference can be slow, local quantization can affect output, a fixed context window can
become limiting for unusually long retrieved evidence, and retrieval misses remain an upper bound on
grounded verdict quality. SciFact's evidence annotations are benchmark labels rather than exhaustive
scientific truth. Temperature zero and pinned settings improve reproducibility but do not guarantee
identical output across Ollama/runtime/hardware versions. Iterative RAG must be evaluated against the
one-pass baseline rather than altering it. The next milestone is the multi-agent ARGUS system
(Support/Challenge agents, Evidence Verifier, and Judge).
