# ARGUS — Milestone 5 ARGUS Multi-agent RAG: Full Internal Validation Results

**Runs:** `20261026T151038Z` (full ARGUS, `outputs/argus/20261026T151038Z/`) and `20261026T151208Z`
(ARGUS without the Challenge agent, `outputs/argus/20261026T151208Z/`).

**Sources:** local `metadata.json`, `metrics.json` and `predictions.jsonl` from both runs; Slurm
accounting and `outputs/cluster/<job>/environment.txt` / `ollama.log` for jobs 339338 and 339339; gold
labels and rationales from the local official SciFact `claims_train.jsonl`. Standard RAG and Iterative
RAG comparison figures are taken from their **A40 reruns** `outputs/rag/20261022T064540Z/` and
`outputs/iterative_rag/20261022T072108Z/` (Slurm job 335327, same node), **not** from the Milestone 3
and Milestone 4 laptop runs.

**Status:** both runs completed; 162 of 162 intended claim IDs present, unique, in metadata order, with
0 remaining. Neither run was interrupted or resumed.

**Run duration.** Both variants ran concurrently as separate Slurm jobs on node `avenue` of the SMU
`crimson` cluster, each with its own NVIDIA A40. Times are UTC, with local time (UTC+8) in brackets.

| Event | Full ARGUS (job 339338) | No-Challenge (job 339339) |
| --- | --- | --- |
| Job started | 2026-10-26 15:10:23 (23:10:23) | 2026-10-26 15:10:24 (23:10:24) |
| Run directory created | 15:10:38 (23:10:38) | 15:12:08 (23:12:08), after a deliberate 90 s delay to avoid a run-directory timestamp collision |
| Run completed | 16:28:06 (27 Oct 00:28:06) | 16:24:07 (27 Oct 00:24:07) |
| Job wall-clock time | 1 h 17 min 46 s | 1 h 13 min 46 s |
| Sum of per-claim wall times | 1 h 17 min 23 s (4,643.3 s) | 1 h 11 min 56 s (4,315.8 s) |

## Executive summary

ARGUS (BGE-small/FAISS Support retrieval k=5 and Challenge retrieval k=5 → deduplicated merge by
SciFact ID → one independent Qwen Evidence Verifier call per abstract → Qwen Judge over verified
evidence → structured verdict/citation validation) attained **68.52% failure-aware claim accuracy
(111/162) and 0.69287 three-class macro-F1** on the seed-42 training-derived internal validation set.
The no-Challenge ablation attained **70.99% (115/162) and 0.70851**. 152 of 162 full-ARGUS attempts
(93.83%) and 158 of 162 ablation attempts (97.53%) yielded valid structured predictions.

On the same 162 claim IDs, same node type, Ollama runtime and model digest, Standard RAG scored
73.46% (119/162, macro-F1 0.73669) and Iterative RAG 72.84% (118/162, 0.73771). **Neither ARGUS variant
improved on either baseline, and no pairwise difference was statistically significant** (exact
McNemar, Holm-corrected, all adjusted p = 1.00). Full ARGUS cost 6.71× the Qwen calls of Standard RAG,
2.76× the recorded tokens and 2.21× the mean per-claim latency.

The Challenge agent contributed little new evidence: 695 of the 810 abstracts it returned (85.8%) had
already been found by the Support agent, and its 115 new abstracts contained 2 annotated gold
documents across all 162 claims (gold-document coverage 93.2% → 94.9%). Against the no-Challenge
ablation, adding the Challenge agent changed the correctness of 6 claims: it gained 1 and lost 5, and
all 5 losses were Judge output failures rather than wrong verdicts.

All 14 failures across both runs occurred at the final Judge call. When ARGUS produced a valid
verdict, its accuracy (111/152 = 73.0% full; 115/158 = 72.8% no-Challenge) was similar to the baselines
(Standard RAG 119/161 = 73.9%). This is a completed internal experiment, not a score on the official
labeled SciFact development split.

## Reproducibility configuration

| Item | Recorded value |
| --- | --- |
| Code | Branch `feature/argus-multi-agent-rag`, commit `201cf4c` ("fix: isolate invalid ARGUS verifier responses per document"), implementation version `argus-v3`. The commit is recorded in each job's `environment.txt`, not in `metadata.json`. |
| Dataset | Original AllenAI SciFact: 5,183 abstracts, 809 train claims |
| Evaluation set | 162 internal validation claims from train, seed 42; ordered IDs identical to the Standard RAG and Iterative RAG A40 runs |
| Gold labeling | SUPPORT/CONTRADICT from annotated evidence; no annotated evidence → INSUFFICIENT; mixed labels excluded (0 encountered) |
| Retriever | `BAAI/bge-small-en-v1.5`, SHA `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, query prefix `Represent this sentence for searching relevant passages: ` |
| Vector search | Normalized 384D float32 vectors; FAISS `IndexFlatIP`; top-5 per agent |
| Corpus fingerprint | `bb15cae5d1852943d020806a38ae4a4154380d3f3ef898d4e9aecd92f7ac08ff` (matches Standard RAG and Iterative RAG) |
| Generator | Qwen3-4B-Instruct-2507 in Ollama, tag `qwen3:4b-instruct-2507-q4_K_M` |
| Ollama model digest | `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0` (matches both baselines) |
| Ollama runtime | 0.40.5 (same as both A40 baseline runs); Q4_K_M quantization |
| Generation | temperature 0, generation top-k 40, `num_predict=512`, `num_ctx=8192`, non-streaming, 300 s client timeout per call |
| Hardware | SMU `crimson` Slurm cluster, partition `student`, node `avenue`; 1× NVIDIA A40 (44.4 GiB usable), 4 CPU cores and 24 GB RAM per job; full run on GPU PCI `0000:41:00.0`, ablation on `0000:81:00.0`; CUDA driver 13.1 (Ollama log). Baseline job 335327 used the same node and GPU model (PCI `0000:41:00.0`). |
| Orchestration | LangGraph; nodes decompose → support → challenge (full only) → merge → verify → judge. The decompose node passes the whole claim as one search objective (no decomposition). |
| Support query | The claim text |
| Challenge query | The claim text followed by the fixed suffix `exceptions limitations conflicting findings alternative outcomes` |
| Agent budgets | `retrieval_k=5` per agent, max 10 merged evidence documents, max 12 LLM calls per claim, deduplication by original SciFact document ID with provenance retained |
| Evidence Verifier | One call per merged abstract, shown the claim and the full abstract with numbered sentences, without retrieval provenance; prompt `argus-v3`, SHA-256 `811de57019988a77a1e7697b5171a6fb550e5791c9943f14ed40d5a5b17b54ee`; rationale `maxLength` 300 characters; SUPPORT/CONTRADICT require at least one cited sentence; an invalid response excludes that abstract and the claim continues |
| Judge | One call over all verified abstracts (verifier label, rationale and full numbered abstract); prompt `argus-v3`, SHA-256 `9a855df4b472fe1bab6301b75bef06345dbc02dd299097db4a3fd17e352e8d8b`; rationale `maxLength` 300 characters, at most 10 evidence items; output validated by the Standard RAG validator restricted to verified documents |
| Python/OS | Not recorded in the ARGUS `metadata.json`. The A40 baselines record Python 3.12.15 on Linux 5.14 (el9). |

**Prompt history.** `argus-v1` and `argus-v2` were replaced after live smoke tests on the A40 (see
*Development history*). Unlike Milestone 4, the five smoke-test claims (1000, 1011, 0, 10, 1001) come
from the train-derived development pool and were verified to be **disjoint from the 162 evaluation
claims**, so prompt revisions were not tuned on the evaluation set.

## Primary claim-verification results

| Metric | Full ARGUS | No-Challenge |
| --- | ---: | ---: |
| Total attempted | 162 | 162 |
| Correct verdicts | 111 | 115 |
| Failure-aware accuracy | 0.685185 (68.52%) | 0.709877 (70.99%) |
| Macro-F1 | 0.692869 | 0.708512 |
| Valid responses | 152 (93.83%) | 158 (97.53%) |
| Accuracy on valid responses only | 111/152 = 73.03% | 115/158 = 72.78% |
| Failures | 10: 6 `invalid_response`, 4 `invalid_json`, all at the Judge | 4: 3 `invalid_response`, 1 `invalid_json`, all at the Judge |
| Class-excluded/mixed-label cases | 0 | 0 |

### Per-class performance

| Gold class | Number | Full P | Full R | Full F1 | No-Ch P | No-Ch R | No-Ch F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SUPPORT | 66 | 0.7941 | 0.8182 | 0.8060 | 0.7910 | 0.8030 | 0.7970 |
| CONTRADICT | 35 | 0.5833 | 0.8000 | 0.6747 | 0.5833 | 0.8000 | 0.6747 |
| INSUFFICIENT | 61 | 0.8056 | 0.4754 | 0.5979 | 0.7907 | 0.5574 | 0.6538 |

### Confusion matrices

Rows are gold; columns are predicted. FAILURE means no valid prediction and remains in denominators.

**Full ARGUS**

| Gold \ Predicted | SUPPORT | CONTRADICT | INSUFFICIENT | FAILURE |
| --- | ---: | ---: | ---: | ---: |
| SUPPORT | 54 | 7 | 5 | 0 |
| CONTRADICT | 4 | 28 | 2 | 1 |
| INSUFFICIENT | 10 | 13 | 29 | 9 |

**No-Challenge**

| Gold \ Predicted | SUPPORT | CONTRADICT | INSUFFICIENT | FAILURE |
| --- | ---: | ---: | ---: | ---: |
| SUPPORT | 53 | 7 | 6 | 0 |
| CONTRADICT | 4 | 28 | 3 | 0 |
| INSUFFICIENT | 10 | 13 | 34 | 4 |

**Frequent errors** (full ARGUS, with A40 Standard RAG in brackets):

- 13 INSUFFICIENT claims were predicted CONTRADICT (14), essentially unchanged.
- 5 SUPPORT claims were predicted INSUFFICIENT (14). ARGUS recovers more SUPPORT claims.
- 7 SUPPORT claims were predicted CONTRADICT (1), and 4 CONTRADICT claims were predicted SUPPORT (0).
  ARGUS makes more opposite-polarity errors.
- 10 INSUFFICIENT claims were predicted SUPPORT (8).
- 9 of the 10 failures fall on INSUFFICIENT gold claims, which lowers INSUFFICIENT recall to 0.475
  (Standard RAG 0.623).

Whether an individual error is due to retrieval, verifier labelling or Judge reasoning requires
per-claim audits against gold evidence and the abstracts shown.

## Citation overlap with SciFact annotations

| Metric | Full: numerator / denominator | Full | No-Ch: numerator / denominator | No-Ch |
| --- | ---: | ---: | ---: | ---: |
| Valid structured responses | 152 / 162 | 93.83% | 158 / 162 | 97.53% |
| Cited-document precision | 108 / 339 | 31.86% | 107 / 361 | 29.64% |
| Cited-document recall | 108 / 117 | 92.31% | 107 / 118 | 90.68% |
| Cited-sentence precision | 110 / 1,086 | 10.13% | 107 / 1,390 | 7.70% |
| Cited-sentence recall | 110 / 202 | 54.46% | 107 / 203 | 52.71% |
| Gold-document coverage after Support retrieval | 110 / 118 | 93.22% | 110 / 118 | 93.22% |
| Gold-document coverage after all agents | 112 / 118 | 94.92% | 110 / 118 | 93.22% |

Values match `metrics.json`. Recall denominators count gold items only for claims with a valid
response. The Judge cites a mean of 2.23 documents (full) and 2.28 (no-Challenge) per verdict, but a mean
of 7.14 and 8.80 sentences, with maxima of 58 and 59. This is why sentence precision is about a third of
Standard RAG's (29.15%).

Structural validation guarantees cited IDs/offsets exist in the verified evidence shown to the Judge,
not that the cited content entails the verdict. Gold-evidence overlap is not a semantic-faithfulness or
exhaustive-scientific-truth measure.

## Efficiency and resource usage

| Measurement | Full ARGUS | No-Challenge |
| --- | ---: | ---: |
| Mean end-to-end latency | 28.66 s / claim | 26.64 s / claim |
| Median end-to-end latency | 26.85 s / claim | 25.60 s / claim |
| 95th-percentile end-to-end latency (nearest rank) | 39.91 s / claim | 37.16 s / claim |
| Range | 18.51 – 51.75 s / claim | 17.80 – 44.47 s / claim |
| Retrieval latency per claim (mean / median) | 1,707.94 ms / 2,095.59 ms | 1,612.16 ms / 1,988.39 ms |
| Retrieval calls | 324 (162 Support, 162 Challenge) | 162 |
| LLM calls | 1,087 (6.71 / claim): 925 verifier, 162 Judge | 972 (6.00 / claim): 810 verifier, 162 Judge |
| Prompt tokens reported | 1,014,269 (1,087 measured calls) | 881,891 (972 measured calls) |
| Completion tokens reported | 80,491 | 73,967 |
| Total recorded prompt + completion tokens | 1,094,760 (6,757.8 / claim) | 955,858 (5,900.4 / claim) |
| Largest single prompt | 5,356 tokens (a Judge call; limit 8,192) | 4,290 tokens (a Judge call) |
| LLM-call budget exhaustion / context-window failures | 0 / 0 | 0 / 0 |

Per-call cost by purpose:

| Call purpose | Calls | Mean latency | Median latency | Prompt tokens (mean / max) | Max completion tokens | Calls at the 512-token limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Verifier (full) | 925 | 2.98 s | 2.89 s | 623 / 2,026 | 120 | 0 |
| Judge (full) | 162 | 9.96 s | 8.01 s | 2,702 / 5,356 | 512 | 4 |
| Verifier (no-Challenge) | 810 | 3.00 s | 2.97 s | 617 / 2,026 | 98 | 0 |
| Judge (no-Challenge) | 162 | 10.03 s | 8.01 s | 2,358 / 4,290 | 512 | 1 |

Every LLM call has Ollama token counts (no timeouts occurred), so the token totals are complete.
Retrieval time is dominated by the Support call (272.9 s in total; the Challenge call totals 3.8 s); the
measurements do not establish why the first query per claim is slower. Retrieval is small next to
generation time.

## Agent behaviour

| Behaviour measure | Full ARGUS | No-Challenge |
| --- | --- | --- |
| Abstracts returned by retrieval calls | 1,620 (810 Support, 810 Challenge) | 810 |
| Challenge results already found by Support | 695 / 810 (85.8%) | — |
| Challenge-only abstracts | 115 across 88 claims (mean 0.71 / claim) | — |
| Challenge-only gold documents | 2 | — |
| Merged abstracts per claim | mean 5.71, max 8 (5: 74 claims, 6: 67, 7: 15, 8: 6); the 10-document cap never bound | 5 for every claim |
| Verifier attempts / valid | 925 / 924 (99.89%) | 810 / 810 (100%) |
| Abstracts excluded after an invalid verifier response | 1 (claim still answered) | 0 |
| Verifier labels | SUPPORT 112, CONTRADICT 92, INSUFFICIENT 720 | SUPPORT 108, CONTRADICT 88, INSUFFICIENT 614 |
| Claims where a Challenge-only abstract was verified SUPPORT or CONTRADICT | 7 | — |
| Verdicts citing a Challenge-only abstract | 19 / 152 | — |

The fixed Challenge query appends generic terms to the claim. Under dense retrieval this mostly
re-finds the same abstracts: on 74 claims the merged set was exactly the Support set. This matches the
Milestone 4 finding that paraphrased dense queries added no new gold evidence.

Full per-claim execution traces are stored in `predictions.jsonl`: each agent's query, retrieved IDs and
latency; merged evidence with provenance; each verifier and Judge call's prompt version and hash, raw
response, parsed result, validation status, tokens and latency.

## Comparison with Standard RAG and Iterative RAG (A40 reruns)

All four systems used the same 162 claim IDs, corpus, retriever, model digest, decoding settings,
Ollama 0.40.5 and node type. The Standard RAG and Iterative RAG figures below are from the 22 October
A40 reruns and differ from the Milestone 3 and Milestone 4 reports, which used laptop runs.

| Metric | Standard RAG (`20261022T064540Z`) | Iterative RAG (`20261022T072108Z`) | ARGUS no-Challenge | ARGUS full |
| --- | ---: | ---: | ---: | ---: |
| Correct verdicts | 119 / 162 | 118 / 162 | 115 / 162 | 111 / 162 |
| Failure-aware accuracy | 73.46% | 72.84% | 70.99% | 68.52% |
| Macro-F1 | 0.7367 | 0.7377 | 0.7085 | 0.6929 |
| SUPPORT F1 / recall | 0.8160 / 0.7727 | 0.7869 / 0.7273 | 0.7970 / 0.8030 | 0.8060 / 0.8182 |
| CONTRADICT F1 / recall | 0.7500 / 0.8571 | 0.7711 / 0.9143 | 0.6747 / 0.8000 | 0.6747 / 0.8000 |
| INSUFFICIENT F1 / recall | 0.6441 / 0.6230 | 0.6552 / 0.6230 | 0.6538 / 0.5574 | 0.5979 / 0.4754 |
| Valid responses (failures) | 161 (1) | 159 (3) | 158 (4) | 152 (10) |
| Cited-document precision / recall | 40.70% / 88.98% | 38.60% / 90.52% | 29.64% / 90.68% | 31.86% / 92.31% |
| Cited-sentence precision / recall | 29.15% / 49.26% | 28.78% / 49.25% | 7.70% / 52.71% | 10.13% / 54.46% |
| Retrieval calls | 162 | 450 | 162 | 324 |
| LLM calls (per claim) | 162 (1.00) | 828 (5.11) | 972 (6.00) | 1,087 (6.71) |
| Recorded prompt + completion tokens (per claim) | 396,703 (2,449) | 1,835,220 (11,329) | 955,858 (5,900) | 1,094,760 (6,758) |
| Mean / median end-to-end latency | 12.95 s / 12.02 s | 32.06 s / 32.24 s | 26.64 s / 25.60 s | 28.66 s / 26.85 s |

### Paired per-claim analysis

Exact two-sided McNemar tests on per-claim correctness (failures counted as incorrect), Holm-corrected
over all six pairs, and paired bootstrap 95% intervals on the accuracy difference (10,000 resamples of
the 162 claims, seed 42):

| A vs B | Only A correct | Only B correct | p | Holm p | A − B accuracy (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| Standard RAG vs Iterative RAG | 11 | 10 | 1.000 | 1.000 | — |
| Standard RAG vs ARGUS no-Challenge | 15 | 11 | 0.557 | 1.000 | ARGUS −2.47 pts [−8.64, +3.70] |
| Standard RAG vs ARGUS full | 19 | 11 | 0.201 | 1.000 | ARGUS −4.94 pts [−11.73, +1.85] |
| Iterative RAG vs ARGUS no-Challenge | 15 | 12 | 0.701 | 1.000 | ARGUS −1.85 pts [−8.04, +4.32] |
| Iterative RAG vs ARGUS full | 19 | 12 | 0.281 | 1.000 | ARGUS −4.32 pts [−11.11, +2.47] |
| ARGUS no-Challenge vs ARGUS full | 5 | 1 | 0.219 | 1.000 | Full −2.47 pts [−5.56, +0.00] |

Of the 19 claims Standard RAG answered correctly and full ARGUS did not, 8 were full-ARGUS failures. Full
ARGUS and the ablation gave different verdicts on 9 claims. The Challenge agent's single gain was claim
1311 (SUPPORT; the ablation said INSUFFICIENT). Its five losses (1342, 15, 255, 828, 96) were all
INSUFFICIENT claims that the ablation answered correctly and on which the full Judge failed.

**Caveats that limit what this comparison shows:**

- **No significant differences.** With 162 claims, the intervals span roughly ±6–7 points, so
  differences of the observed size cannot be distinguished from chance.
- **Run-to-run variation is of the same size as the system differences.** The identical Iterative RAG
  configuration (prompts `iterative-rag-v3`, Ollama 0.40.5, temperature 0) produced different verdicts
  on 25 of 162 claims, and different correctness on 21, between its laptop run (`20261017T165231Z`,
  70.99%) and its A40 rerun (`20261022T072108Z`, 72.84%). Standard RAG moved from 66.67% (Milestone 3,
  Ollama 0.23.2 on CPU) to 73.46% on the A40. Temperature 0 with the same digest does not make outputs
  identical across hardware/runtime, and same-hardware run-to-run variance has not been measured.
- **The four runs were not simultaneous.** The baselines ran on 22 October, ARGUS on 26 October, on the
  same node and GPU model but not necessarily the same physical GPU. The two ARGUS jobs ran
  concurrently on different A40s of the same node.
- **Latency is comparable across these four runs**, which all used A40s with Ollama 0.40.5, but not with
  the Milestone 3/4 laptop latencies.

## Failure investigation

All failures occurred at the final Judge call. No retrieval call failed, no verifier failure stopped a
claim, no call hit the context limit or the LLM-call budget, and no call timed out.

| Claim | Gold | Run(s) | Status | Merged docs | Judge prompt / completion tokens | Cause |
| --- | --- | --- | --- | ---: | --- | --- |
| 15 | INSUFFICIENT | full | invalid_json | 7 | 3,857 / 512 | Hit 512-token limit; JSON unterminated |
| 96 | INSUFFICIENT | full | invalid_json | 6 | 3,524 / 512 | Hit 512-token limit; JSON unterminated |
| 255 | INSUFFICIENT | full | invalid_json | 6 | 4,580 / 512 | Hit 512-token limit; JSON incomplete |
| 828 | INSUFFICIENT | full | invalid_json | 7 | 4,724 / 512 | Hit 512-token limit; JSON incomplete |
| 494 | INSUFFICIENT | no-Challenge | invalid_json | 5 | 3,722 / 512 | Hit 512-token limit; JSON incomplete |
| 244 | INSUFFICIENT | both | invalid_response | 5 | 2,187 / 239 | Evidence item with empty `sentence_indices` |
| 655 | INSUFFICIENT | both | invalid_response | 6 / 5 | 2,622 / 234 (full); 2,248 / 207 | Evidence item with empty `sentence_indices` |
| 245 | CONTRADICT | full | invalid_response | 6 | 2,603 / 390 | Evidence item with empty `sentence_indices` |
| 688 | INSUFFICIENT | full | invalid_response | 6 | 2,208 / 308 | Evidence item with empty `sentence_indices` |
| 1342 | INSUFFICIENT | full | invalid_response | 8 | 2,463 / 407 | Cited an out-of-range sentence in document 2722988 |
| 1364 | INSUFFICIENT | both | invalid_response | 5 | 2,928 / 417 | Cited an out-of-range sentence in document 25175997 |

- **Output-limit truncation (4 full, 1 ablation).** The v2 fix bounded the Judge's rationale but not the
  length of each `sentence_indices` list. These Judge prompts were the longest (3,524–4,724 tokens),
  and the Judge cited long sentence lists until the 512-token limit cut off the JSON.
- **Empty sentence lists (4 full, 2 ablation)** and **out-of-range sentence indices (2 full, 1
  ablation).** The shared validator correctly rejects these. They cluster on INSUFFICIENT claims, where
  the Judge lists abstracts even when none directly supports a verdict. An empty `evidence` list is
  accepted by the validator, but the prompt does not say so.
- **Overlap with the baselines.** Claim 1342 also failed in A40 Standard RAG; claim 494 also failed in
  A40 Iterative RAG. On the seven full-only failures, Standard RAG answered five correctly.

Do not erase, reclassify or silently retry these outcomes. A Judge fix must be predeclared and run as a
separately labelled version (`argus-v4`).

## Development history (smoke tests)

All smoke tests ran on an A40 on the same five development claims (1000, 1011, 0, 10, 1001), in both
variants.

| Version | Commit | Runs | Outcome | Change made |
| --- | --- | --- | --- | --- |
| `argus-v1` | `f893542` | `20261026T114630Z`, `20261026T114948Z` | 0/10 success: every claim failed at the first verifier call; the verifier rationale repeated itself until the 512-token limit, leaving JSON unterminated | `argus-v2`: rationale `maxLength` 300 characters and "one sentence, at most 40 words, do not repeat yourself" for verifier and Judge; Judge evidence capped at 10 items |
| `argus-v2` | `43dab9d` | `20261026T134627Z`, `20261026T134847Z` | 6/10 success; claims 1000 and 1001 failed in both variants because the verifier returned SUPPORT/CONTRADICT with no cited sentence for the first abstract, which failed the whole claim | `argus-v3`: prompt requires a cited sentence for direct labels; an invalid verifier response now excludes only that abstract |
| `argus-v3` | `201cf4c` | `20261026T150324Z`, `20261026T150557Z` | 10/10 success, 0 invalid verifier responses, maximum completion 350 tokens | Frozen for the full evaluation |

One further attempt was started by mistake on the GPU-less login node, was interrupted within minutes,
and was moved to `outputs/argus_invalid/20261026T132927Z/`. It is excluded from every result.

## Findings relevant to subsequent ARGUS research

- **The Challenge agent, as implemented, does not find counter-evidence.** Appending generic terms
  ("exceptions limitations conflicting findings alternative outcomes") to the claim under dense
  retrieval returns mostly the Support agent's abstracts (85.8% overlap) and added 2 gold documents in
  162 claims. Testing the research question properly needs a Challenge retrieval that is actually
  different: excluding already-retrieved abstracts, BM25/hybrid retrieval, or an LLM-generated
  negated/counter-hypothesis query, each as a separately labelled ablation.
- **The Judge is the reliability bottleneck.** All 14 failures were Judge outputs, and more evidence
  produced more failures (full 10 vs ablation 4). Bounding citation lists and stating that INSUFFICIENT
  may have no evidence are the obvious next steps.
- **Per-passage verification shifted errors rather than reducing them.** SUPPORT recall rose (0.773 →
  0.818 vs Standard RAG), but opposite-polarity errors rose (SUPPORT→CONTRADICT 1 → 7) and INSUFFICIENT
  recall fell. The verifier labelled 204 of 924 abstracts (22%) as direct evidence, which may be
  over-generous for a small quantized model.
- **Citation precision dropped sharply** (sentence precision 29% → 10%) because the Judge cites many
  sentences per document. Citation length needs an explicit bound or instruction.
- **Cost.** About 6.7 Qwen calls and 6,758 tokens per claim (2.8× Standard RAG's tokens, but 0.6×
  Iterative RAG's), with no accuracy gain on this split.
- **Measurement noise matters.** Cross-hardware variation of the same configuration (25/162 verdicts)
  is as large as the differences between systems. Same-hardware repeat runs are needed before small
  differences are interpreted.

## Methodological limitations to retain in final report

- This is a training-derived internal validation result, not official development/test performance.
  The official labeled dev split was not used for tuning or evaluation and remains reserved for
  predeclared final evaluations after configurations are frozen.
- `argus-v1` → `v3` were revised after smoke tests on five development claims disjoint from the 162
  evaluation claims. Any `argus-v4` change made after seeing these validation results is tuning on the
  internal validation set and must be disclosed as such.
- INSUFFICIENT is operationalized as no annotated gold evidence, which does not establish that a claim
  is unanswerable from the literature. Avoid direct comparison to published SciFact metrics without
  aligning protocols.
- The Evidence Verifier classifies whole abstracts with cited sentences, not gold rationales. Verifier
  labels are not evaluated against an independent sentence-level gold standard.
- The full original abstracts are sent to Qwen, but BGE's 512-token encoding limit affects how some
  documents are retrieved (same as both baselines).
- Q4_K_M quantization differs from the BF16 checkpoint. Outputs at temperature 0 are not guaranteed to
  be identical across hardware, GPU, runtime or concurrency.
- Results are single runs per system with no same-hardware repeat. The ARGUS `metadata.json` does not
  record the Python/OS environment or the Git commit; these come from the Slurm job files.
- Verifier and Judge decisions come from the same small quantized model, and their labels are not
  calibrated confidence.

## Checks performed while preparing this report

Independently re-read and recomputed from the 162 prediction records of each run and the local SciFact
files:

- All 162 claim IDs unique in each run; ordered IDs match `metadata.json`; status completed with 0
  remaining.
- Claim IDs equal the seed-42 internal validation split, and every recorded gold label equals the label
  re-derived from `claims_train.jsonl`.
- Full: accuracy 111/162 = 0.6851852, macro-F1 0.6928690. No-Challenge: 115/162 = 0.7098765, macro-F1
  0.7085125. All match `metrics.json`.
- Status counts: full 152 success, 6 `invalid_response`, 4 `invalid_json`; no-Challenge 158 success, 3
  `invalid_response`, 1 `invalid_json`.
- Latency statistics, per-purpose call counts (925/162 and 810/162) and token totals recomputed from
  per-call traces and matching `metrics.json`.
- Corpus fingerprint, retriever SHA, model digest, Ollama runtime and the 162 ordered claim IDs match
  both A40 baseline runs.
- The five smoke-test claim IDs are absent from the 162 evaluation IDs.
- Paired McNemar tests and bootstrap intervals computed from per-claim correctness of all four runs.
- The Challenge configuration and the GPU (NVIDIA A40, CUDA) were confirmed from job `environment.txt`
  and `ollama.log`.
- Not done: semantic review of whether cited sentences entail the verdicts, an audit of verifier labels
  against gold rationales, and a same-hardware repeat run.

## Recommended next research steps

1. Preserve both runs' `metadata.json`, `metrics.json` and `predictions.jsonl`, the smoke-test runs and
   the job files unchanged and backed up (currently on the cluster under `~/argus-results-backup/` and
   on a team laptop); commit only this summary if appropriate.
2. Measure noise: rerun Standard RAG (about 35 minutes on an A40) on the same node to quantify
   same-hardware run-to-run variation before interpreting differences of a few points.
3. As a predeclared, separately labelled `argus-v4`: bound the Judge's `sentence_indices` (for example
   1–3 per document) and total citations, state that INSUFFICIENT may have an empty evidence list, and
   instruct the Judge to cite only sentence numbers shown. Rerun both ARGUS variants on the same 162 IDs.
4. As separately labelled ablations, give the Challenge agent a genuinely different retrieval:
   exclude Support's abstracts, use BM25/hybrid retrieval, or generate a counter-hypothesis query.
   Report Challenge-only gold documents and verdict changes alongside accuracy.
5. Audit the 30 claims where Standard RAG and full ARGUS disagree in correctness, plus the 7
   SUPPORT→CONTRADICT errors, against gold evidence and the verifier labels shown to the Judge.
6. Record the Git commit and Python/OS environment in ARGUS `metadata.json`, as the baselines do.
7. Freeze prompts and architectures for all four systems, then evaluate each once on the official
   development claims. Do not extrapolate internal-validation results to held-out performance.
