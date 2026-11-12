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

The checked-in [`configs/argus-v3.json`](configs/argus-v3.json) records conservative defaults for
the 8,192-token context window. Complete abstracts are never application-truncated. Ollama calls
reported at or above the configured context window are rejected, as are malformed verifier output,
out-of-range verifier citations, malformed judge output, and judge citations outside the verified
document set. These conditions remain recorded failures rather than being converted to an
`INSUFFICIENT` verdict.

ARGUS v3 retains v2's 300-character, one-sentence rationale limits and adds an explicit verifier
citation instruction: SUPPORT and CONTRADICT each require at least one valid zero-based sentence
reference; otherwise the verifier must use INSUFFICIENT only when direct evidence is absent. The
application remains authoritative: malformed JSON, unsupported labels, missing required citations,
and invalid sentence offsets are rejected.

Unlike v2, a validation-only failure for one abstract excludes only that abstract and records its
document ID, raw response, error category/message, token usage, latency, and admission decision in
the persisted trace. ARGUS continues verifying the remaining abstracts and the Judge receives only
valid `VerifiedEvidence`. A partially verified claim may therefore produce a valid verdict; if all
attempted verifier responses are invalid, no Judge call or fabricated verdict is produced. A run
with no retrieved abstracts is recorded separately as a retrieval failure. Service errors,
context-window failures, and exhausted LLM-call budgets remain claim-fatal. Existing Standard RAG
and Iterative RAG validation behavior is unchanged.

ARGUS metrics now additionally report verifier attempts, successful classifications, invalid
responses, citation/missing-reference failures, other validation failures, excluded documents,
partial/complete/all-invalid claim counts, each claim's verifier success ratio, and the overall
valid-response rate. All are derived from saved traces and are available for both full and
no-Challenge variants.

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

For the A40 preflight after this v3 update, do not resume an `argus-v1` or `argus-v2` directory.
Run the two five-claim variants into new output roots:

```cmd
uv run --locked python scripts/run_argus.py --smoke-test --output-dir outputs/argus-v3-smoke-full
uv run --locked python scripts/run_argus.py --smoke-test --no-challenge --output-dir outputs/argus-v3-smoke-no-challenge
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

## Challenge retrieval ablation (opt-in)

By default the Challenge agent searches dense retrieval with the claim plus a fixed suffix. These
options change only that search; Support stays on dense retrieval, and the Verifier and Judge never
see which agent found a document:

| Option | Values | Default |
| --- | --- | --- |
| `--challenge-retrieval` | `dense`, `bm25`, `hybrid` | `dense` |
| `--challenge-query` | `suffix`, `claim`, `counter_hypothesis` | `suffix` |
| `--challenge-exclude-support` | flag | off |

```cmd
uv run --locked python scripts/run_argus.py --smoke-test --challenge-retrieval bm25 --challenge-query claim
uv run --locked python scripts/run_argus.py --evaluate --challenge-retrieval bm25 --challenge-query claim
uv run --locked python scripts/run_argus.py --evaluate --challenge-query claim --challenge-exclude-support
uv run --locked python scripts/run_argus.py --evaluate --challenge-retrieval hybrid --challenge-query claim --challenge-exclude-support
uv run --locked python scripts/run_argus.py --evaluate --challenge-query counter_hypothesis
```

`counter_hypothesis` makes one extra Qwen call that sees only the claim and writes a search query
for conflicting findings (prompt `argus-challenge-query-v1`, recorded in run metadata). The query is
used only for searching and is never shown to the Verifier or Judge. If the call fails or its query
is invalid, the claim fails without a verdict; there is no retry or fallback to another query. With
`--challenge-exclude-support`, a claim can use the entire default budget of 12 model calls.

`hybrid` is the existing reciprocal rank fusion of BM25 and dense rankings with its default
settings. `--challenge-exclude-support` makes Challenge return only documents Support did not find, up to
`--k`, so a claim verifies up to twice as many abstracts and uses more model calls. With dense
retrieval and the claim as query, this returns Support's next-ranked documents, so treat that
configuration as a "retrieve more" control rather than a search for counter-evidence.

Without these options a run keeps its original `full` variant label. Otherwise `metadata.json`
records a variant such as `bm25_claim`, an `implementation_version` ending in
`-challenge-retrieval`, and the search under `retrieval.challenge`. All three are rejected with
`--no-challenge`. No claim that these options improve ARGUS is made until runs are complete.

## ARGUS v4 Judge reliability (opt-in)

The original ARGUS v3 Judge, verifier, retrieval behavior, and historical outputs are preserved.
Use `--judge-version v4` to select the separately recorded **argus-v4-judge** experiment.
The default `--judge-version v3` retains the original behavior.

Only the final Judge changes: its system prompt explicitly permits an empty evidence list for
INSUFFICIENT when no decisive evidence exists, requires real zero-based sentence indices,
and asks for at most three cited documents with one to three decisive sentence indices each.
The structured schema adds citation limits, and application-side validation still rejects
invalid references. The Evidence Verifier stays at argus-v3, and the same model, temperature,
retrieval, 512-token completion limit, and evidence budgets are used. This is a distinct
experimental configuration, **not a correction to any v3 prediction**.
`configs/argus-v4-judge.json` documents the intended experiment settings; the CLI is
authoritative (the JSON file is not automatically loaded).

To validate without evaluating the held-out official development split:

```bash
uv run --locked pytest -q tests/test_argus.py
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/run_argus.py --smoke-test --judge-version v4
uv run --locked python scripts/run_argus.py --smoke-test --judge-version v4 --no-challenge
```

Smoke tests use the training-derived development pool, disjoint from the 162 internal-validation
claims. When prompts are frozen, run full and no-Challenge v4 versions separately with
`--evaluate --judge-version v4` and optionally `--no-challenge`, preferably in the same
A40/Ollama environment as v3. Do not silently retry failed claims. Report failure-aware
accuracy/F1, invalid outputs, citation precision and recall, calls, tokens, latency, and
paired claim-level differences against the preserved v3 runs. The Judge fix does not
guarantee improved factual accuracy or semantic citation faithfulness.

## ARGUS Verifier v2 (opt-in)

`--verifier-version v2` selects a separately recorded **argus-verifier-v2** Evidence Verifier
prompt. The default `--verifier-version v1` keeps the original argus-v3 Verifier unchanged.

Only the Verifier changes. Its prompt keeps every v1 rule and adds four checks before labelling:
subject match (a different entity is INSUFFICIENT, not CONTRADICT), effect direction, negated
claims ("no effect", "ineffective"), and the paper's own findings rather than background it
overturns. The response schema and validation are unchanged. These checks come from the hand
audit in `reports/argus/challenge-error-audit.md`.
Runs record `agent.verifier_version`, the Verifier prompt SHA-256 under `prompts`, and an
`implementation_version` ending in `-verifier-v2`. It combines with any Judge version and
Challenge option:

```bash
uv run --locked python scripts/run_argus.py --smoke-test --judge-version v4.1 --verifier-version v2
uv run --locked python scripts/run_argus.py --evaluate --judge-version v4.1 --verifier-version v2
uv run --locked python scripts/run_argus.py --evaluate --judge-version v4.1 --verifier-version v2 \
  --no-challenge
```

Develop and compare it on the 162 internal-validation claims only. No claim that v2 improves
ARGUS is made until runs are complete.

## ARGUS no-Verifier ablation (opt-in)

`--no-verifier` removes the Evidence Verifier to test whether it helps. Support and Challenge
retrieval and the merge are unchanged; no Verifier call is made, and the Judge receives the merged
abstracts directly. The Judge prompt, schema, `<verified_evidence>` wrapper and citation validation
are unchanged so that only the Verifier is removed; each abstract simply arrives without its
`Verification:` line. Citations are still restricted to the retrieved abstracts.

This deliberately passes unverified retrieved text to the Judge, which the normal pipeline does
not do. It is a labelled ablation for comparison with the Verifier-enabled arms, not a recommended
configuration. `--verifier-version` cannot be combined with it.

Runs record `agent.verifier_enabled: false`, omit the Verifier from `prompts`, append
`_no_verifier` to the variant (`full_no_verifier`, `no_challenge_no_verifier`) and `-no-verifier`
to `implementation_version`. Verifier-enabled runs keep their earlier metadata unchanged.

```bash
uv run --locked python scripts/run_argus.py --smoke-test --judge-version v4.1 --no-verifier
uv run --locked python scripts/run_argus.py --evaluate --judge-version v4.1 --no-verifier
uv run --locked python scripts/run_argus.py --evaluate --judge-version v4.1 --no-verifier \
  --no-challenge
```

Compare it on the 162 internal-validation claims only. No claim that removing the Verifier helps
or hurts ARGUS is made until runs are complete.

Because the unchanged Judge prompt still calls its input "verified", `--neutral-judge-wording`
(requires `--no-verifier` and `--judge-version v4`/`v4.1`) checks whether that wording matters.
It swaps in a copy of the v4 Judge prompt that says "retrieved" abstracts and drops the mention of
verifier rationales, and wraps the evidence in `<retrieved_evidence>`; every other instruction,
the schema and citation validation are identical. Runs record
`agent.neutral_judge_wording: true`, Judge prompt version `argus-v4-judge-unverified`, the
`_neutral` variant suffix and `-neutral-judge` in `implementation_version`. Other runs are unaffected.

```bash
uv run --locked python scripts/run_argus.py --evaluate --judge-version v4.1 --no-verifier   --neutral-judge-wording
uv run --locked python scripts/run_argus.py --evaluate --judge-version v4.1 --no-verifier   --no-challenge --neutral-judge-wording
```

## Official dev split for the final evaluation (opt-in)

All claim-verification runs default to the 162-claim seed-42 internal validation subset, which was
used for development and prompt revision. `--split dev` evaluates the official labeled SciFact dev
split (`claims_dev.jsonl`: 300 claims, 188 with annotated evidence, 112 without) instead. It is
reserved for the final, frozen comparison: run each system once, with configurations fixed in
advance, and do not revise prompts or settings after seeing its results.

`--split dev` is accepted only with `--evaluate` (smoke tests stay on train-derived claims) and works
with `--limit` and `--resume`. Claims are evaluated in stable claim-ID order, and the split's counts
are checked before any LLM call. Runs record evaluation mode `official_dev` (or
`partial_official_dev` with `--limit`) and the split source; the Standard and Iterative RAG run
state reports `full_official_dev_claim_count`. Runs without `--split dev` keep their existing
metadata unchanged.

```bash
uv run --locked python scripts/run_standard_rag.py --evaluate --split dev --output-dir outputs/dev/rag
uv run --locked python scripts/run_iterative_rag.py --evaluate --split dev \
  --output-dir outputs/dev/iterative_rag
uv run --locked python scripts/run_argus.py --evaluate --split dev --judge-version v4.1 \
  --verifier-version v2 --no-challenge --output-dir outputs/dev/argus/A
uv run --locked python scripts/run_argus.py --evaluate --split dev --judge-version v4.1 \
  --verifier-version v2 --output-dir outputs/dev/argus/B
uv run --locked python scripts/run_argus.py --evaluate --split dev --judge-version v4.1 \
  --no-verifier --neutral-judge-wording --output-dir outputs/dev/argus/B-no-verifier
```

## OpenAI provider (opt-in)

`scripts/run_standard_rag.py`, `scripts/run_iterative_rag.py` and `scripts/run_argus.py` accept
`--provider openai` to run the same prompts, retrieval, budgets and scoring with an OpenAI model
through the Responses API (`src/argus/llm/openai_client.py`, standard library only). Ollama
remains the default, and Ollama runs record the same metadata as before.

Set the key in the shell; it is read from `OPENAI_API_KEY` and never written to outputs:

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run --locked python scripts/run_argus.py --smoke-test --provider openai --judge-version v4.1 --verifier-version v2 --output-dir outputs/openai/argus/B
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `gpt-5.6-luna` | API model; the response's resolved model name is saved per call |
| `--reasoning-effort` | `none` | Keeps hidden reasoning out of the `--num-predict` output cap; `omit` sends none |
| `--openai-schema` | `strict` | Strict structured outputs; `loose` sends the schema without strict mode |
| `--omit-temperature` | off | Sends no temperature, for models that reject `temperature=0` |
| `--max-retries` | 5 | Retries for rate limits, server errors and connection errors only |

Differences from the Ollama runs, recorded in `metadata.json`: no local model digest, provider-side
precision unknown, transient errors retried, and API outputs not guaranteed identical across runs
at temperature 0. Write OpenAI runs to separate directories such as `outputs/openai/`.

## Offline run analysis

`scripts/analyze_runs.py` reads finished runs' `predictions.jsonl` files and makes no model or
retrieval calls. For each run it reports agent steps (LLM plus retrieval calls), LLM-call success,
stop reasons, cost per correct claim, MRR@k/nDCG@k/Recall@k/Hit@k of the first retrieval, how
often gold abstracts reached the verdict stage, and a heuristic attribution of each wrong claim to
the earliest stage that explains it (invalid output, retrieval, Verifier or final verdict call). For
ARGUS runs it also scores the Verifier's label on each gold-annotated abstract and counts what the
Challenge agent's abstracts added. For every run it rescores accuracy under alternative rules for
invalid outputs.
Results go to a new `outputs/analysis/<UTC-run-id>/analysis.json`.

```bash
uv run --locked python scripts/analyze_runs.py --split dev \
  --run "Standard RAG=outputs/dev/rag/<run-id>" --run "ARGUS B=outputs/dev/argus/B/<run-id>"
```
