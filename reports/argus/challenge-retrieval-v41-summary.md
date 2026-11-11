# ARGUS — Challenge Retrieval Ablation with the v4.1 Judge: Full Internal Validation Results

**Runs (v4.1 Judge):** arms A–F under `outputs/argus-challenge-v41/<arm>/<run-id>/`, listed below. All six
arms were rerun, including A and B, so every arm in this report uses the same code and Judge.

**Sources:** local `metadata.json`, `metrics.json` and `predictions.jsonl` from each run;
`outputs/cluster/<job>/environment.txt` for the jobs; gold labels and rationales from the local official
SciFact `claims_train.jsonl`. The v3 results are from `challenge-retrieval-summary.md` and its runs. Standard
RAG and Iterative RAG figures come from their A40 reruns `outputs/rag/20261022T064540Z/` and
`outputs/iterative_rag/20261022T072108Z/`.

**Status:** all six runs completed; 162 of 162 intended claim IDs present, unique and in metadata order.
None was interrupted or resumed.

| Arm | Variant | Challenge search | Challenge query | Skips Support's documents | Run | Job | Started (UTC+8) | Run duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | `no_challenge` | none | none | n/a | `20261105T144000Z` | 354640 | 5 Nov 22:40 | 1 h 02 min 26 s |
| B | `full` | dense | claim + fixed suffix | no | `20261105T171126Z` | 354641 | 6 Nov 01:11 | 1 h 09 min 03 s |
| C | `dense_claim_exclude` | dense | claim | yes | `20261105T113313Z` | 354410 | 5 Nov 19:33 | 1 h 45 min 25 s |
| D | `bm25_claim` | BM25 | claim | no | `20261105T113314Z` | 354411 | 5 Nov 19:33 | 1 h 28 min 07 s |
| E | `dense_counter_hypothesis` | dense | one LLM-written counter-hypothesis | no | `20261105T131144Z` | 354412 | 5 Nov 21:11 | 1 h 24 min 00 s |
| F | `hybrid_claim_exclude` | hybrid RRF | claim | yes | `20261105T132301Z` | 354413 | 5 Nov 21:23 | 1 h 45 min 33 s |

Run duration is from the run ID's start time to the last write recorded in `metadata.json`. Slurm
accounting was not read for these jobs.

## Executive summary

The v3 ablation found that Challenge retrieval could be made distinct, but Judge failures hid whether it
helped verdicts. This experiment reruns all six arms with the `argus-v4.1-citation-normalization` Judge,
with retrieval and the Verifier unchanged, to give a fair test of the retrieval strategies.

**The v4.1 Judge removed almost all failures.** Across 972 claim runs there was one failure, against 90
on v3. No Judge reply came near the 512-token limit (longest 210 tokens).

**Retrieval was unchanged from v3, so the Judge is the only difference.** Every arm retrieved the same
documents as on v3 for all 162 claims, and Verifier labels matched on every claim except one document on
claim 866 in C.

**No Challenge arm beat no-Challenge.** Failure-aware accuracy: A 74.69%, B 74.69%, D 73.46%,
E 73.46%, C 72.84%, F 72.84%. Macro-F1 ranges from 0.7156 (F) to 0.7355 (A). No difference between arms
is significant (all Holm-adjusted p = 1.000); the largest is 4 claims.

**Challenge documents are used, but did not raise accuracy.** Verdicts cited a document only Challenge
found in 4 (B), 19 (C), 20 (D), 11 (E) and 27 (F) claims. They changed outcomes in both directions: one
Challenge-only document flipped claim 494 from correct to wrong in all five Challenge arms, while
another corrected claim 951 in D and F.

**Every arm improved significantly over its v3 result,** by 6 (A) to 19 (F) claims, almost all by
turning failures into correct verdicts. This gain belongs to the Judge change, not retrieval.

**The cost rose with no accuracy gain:** 6.00 LLM calls per claim for A, 6.71 for B, 8.36 (E),
9.16 (D) and 11.00 (C, F). C and F use about twice A's tokens for 3 fewer correct verdicts.

**All systems are now statistically level.** A and B (121 correct) are not significantly different from
Standard RAG (119) or Iterative RAG (118). ARGUS has higher INSUFFICIENT F1 than both baselines but lower
CONTRADICT F1. This is a completed internal experiment, not a score on the official labeled SciFact
development split.

## Reproducibility configuration

| Item | Recorded value |
| --- | --- |
| Code | Branch `feat/challenge-retrieval`, commit `8d331ba` for all six arms, recorded in each job's `environment.txt`, not in `metadata.json`. This merges `feature/argus-multi-agent-rag` (v4.1 Judge) into the Challenge retrieval options. Implementation version `argus-v4.1-citation-normalization` for A and B, `argus-v4.1-citation-normalization-challenge-retrieval` for C–F. |
| Dataset | Original AllenAI SciFact: 5,183 abstracts, 809 train claims |
| Evaluation set | 162 internal validation claims from train, seed 42 (66 SUPPORT, 35 CONTRADICT, 61 no annotated evidence); ordered IDs identical across all arms, the v3 runs and both baselines |
| Support retrieval | Unchanged in every arm: BGE-small, FAISS `IndexFlatIP`, top-5, on the claim text |
| Retriever | `BAAI/bge-small-en-v1.5`, SHA `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; corpus fingerprint `bb15cae5d1852943d020806a38ae4a4154380d3f3ef898d4e9aecd92f7ac08ff` |
| Challenge retrieval | As in the v3 ablation: top-5 per the arm table. BM25: `BM25Okapi` over the shared document text. Hybrid: RRF with default `candidate_pool=50`, `rrf_constant=60`. Exclusion requests `k + len(Support)` results, removes Support's documents and keeps the first 5. |
| Suffix (B only) | `exceptions limitations conflicting findings alternative outcomes` |
| Counter-hypothesis (E only) | Prompt `argus-challenge-query-v1`, SHA-256 `17b66fae0b04f13a37d705a32147fa56fd773b5ca14894758d6c10fdbb7435fd` |
| Evidence Verifier | Unchanged: `argus-v3`, SHA-256 `811de57019988a77a1e7697b5171a6fb550e5791c9943f14ed40d5a5b17b54ee` |
| Judge | `--judge-version v4.1`: prompt `argus-v4-judge`, SHA-256 `9a1ba5249f38efd8de37906c1954e9cc953814f9cab4fd4caa1c652a0b3b8bdc`, identical in all arms. At most 3 citation entries with 1–3 sentence indices each; empty evidence allowed for INSUFFICIENT. Unlike v4, citations of the same document are merged rather than rejected; a repeated document/sentence pair is still rejected. |
| Generator | `qwen3:4b-instruct-2507-q4_K_M`, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`; Ollama 0.40.5 |
| Generation | temperature 0, top-k 40, `num_predict=512`, `num_ctx=8192`, non-streaming |
| Budgets | `retrieval_k=5` per agent, max 10 merged documents, max 12 LLM calls per claim |
| Hardware | SMU `crimson`, partition `student`, node `avenue`; 1× NVIDIA A40, 4 CPU cores, 24 GB RAM per job. GPU PCI: A, D `C1:00.0`; B, F `41:00.0`; C `81:00.0`; E `01:00.0`. |

## Primary claim-verification results

Failed claims remain in denominators as FAILURE.

| Metric | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct verdicts | 121 | 121 | 118 | 119 | 119 | 118 |
| Failure-aware accuracy | 0.746914 | 0.746914 | 0.728395 | 0.734568 | 0.734568 | 0.728395 |
| Macro-F1 | 0.735498 | 0.734330 | 0.717323 | 0.723076 | 0.723079 | 0.715556 |
| Valid responses | 162 | 162 | 162 | 161 | 162 | 162 |
| Failures | 0 | 0 | 0 | 1 | 0 | 0 |
| v3 correct verdicts, for reference | 115 | 111 | 100 | 104 | 108 | 99 |

### Per-class performance

| Gold class (number) | Metric | A | B | C | D | E | F |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SUPPORT (66) | Precision | 0.7971 | 0.8000 | 0.7714 | 0.7671 | 0.7857 | 0.7568 |
| | Recall | 0.8333 | 0.8485 | 0.8182 | 0.8485 | 0.8333 | 0.8485 |
| | F1 | 0.8148 | 0.8235 | 0.7941 | 0.8058 | 0.8088 | 0.8000 |
| CONTRADICT (35) | Precision | 0.5833 | 0.5833 | 0.5600 | 0.5625 | 0.5833 | 0.5714 |
| | Recall | 0.8000 | 0.8000 | 0.8000 | 0.7714 | 0.8000 | 0.8000 |
| | F1 | 0.6747 | 0.6747 | 0.6588 | 0.6506 | 0.6747 | 0.6667 |
| INSUFFICIENT (61) | Precision | 0.8444 | 0.8409 | 0.8571 | 0.9000 | 0.8182 | 0.8718 |
| | Recall | 0.6230 | 0.6066 | 0.5902 | 0.5902 | 0.5902 | 0.5574 |
| | F1 | 0.7170 | 0.7048 | 0.6990 | 0.7129 | 0.6857 | 0.6800 |

INSUFFICIENT F1 rose from 0.375–0.654 on v3 to 0.680–0.717, because those claims no longer fail. **CONTRADICT
recall did not rise in any arm** (0.7714–0.8000, as on v3), so this experiment still shows no gain in
detecting contradiction. CONTRADICT precision is low in every arm (0.56–0.58).

### Confusion matrices

Rows are gold; each cell is predicted SUPPORT / CONTRADICT / INSUFFICIENT / FAILURE.

| Gold | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- |
| SUPPORT | 55 / 7 / 4 / 0 | 56 / 6 / 4 / 0 | 54 / 8 / 4 / 0 | 56 / 8 / 2 / 0 | 55 / 7 / 4 / 0 | 56 / 7 / 3 / 0 |
| CONTRADICT | 4 / 28 / 3 / 0 | 4 / 28 / 3 / 0 | 5 / 28 / 2 / 0 | 5 / 27 / 2 / 1 | 3 / 28 / 4 / 0 | 5 / 28 / 2 / 0 |
| INSUFFICIENT | 10 / 13 / 38 / 0 | 10 / 14 / 37 / 0 | 11 / 14 / 36 / 0 | 12 / 13 / 36 / 0 | 12 / 13 / 36 / 0 | 13 / 14 / 34 / 0 |

**The largest error is now INSUFFICIENT claims predicted as CONTRADICT:** 13–14 in every arm, about a
fifth of INSUFFICIENT claims. Another 10–13 are predicted SUPPORT. Since these errors appear in A as well,
they do not come from Challenge retrieval.

## Citation overlap with SciFact annotations

| Metric | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cited-document precision | 56.18% | 56.22% | 53.89% | 54.21% | 52.11% | 53.06% |
| Cited-document recall | 84.75% | 88.14% | 88.14% | 88.03% | 83.90% | 88.14% |
| Cited-sentence precision | 44.19% | 47.42% | 43.38% | 44.19% | 42.79% | 45.21% |
| Cited-sentence recall | 46.80% | 49.75% | 46.80% | 47.50% | 46.80% | 48.77% |
| Cited sentence references | 215 | 213 | 219 | 215 | 222 | 219 |
| INSUFFICIENT verdicts with no citation | 42 | 39 | 39 | 38 | 41 | 38 |
| Mean cited documents per valid verdict | 1.10 | 1.14 | 1.19 | 1.18 | 1.17 | 1.21 |

Values match `metrics.json`. All arms now have valid responses on almost the same claims, so these figures
are comparable across arms, unlike on v3. Against v3, precision roughly doubled (sentence precision
7.7–24.3% on v3) and recall fell (cited-sentence recall 52.7–55.0% on v3), as expected from a Judge capped
at 3 citations. These are annotation-overlap measures, not semantic-faithfulness judgments.

## Efficiency and resource usage

| Measurement | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM calls | 972 | 1,087 | 1,782 | 1,484 | 1,354 | 1,782 |
| LLM calls per claim | 6.00 | 6.71 | 11.00 | 9.16 | 8.36 | 11.00 |
| Verifier calls | 810 | 925 | 1,620 | 1,322 | 1,030 | 1,620 |
| Counter-hypothesis calls | 0 | 0 | 0 | 0 | 162 | 0 |
| Total recorded tokens | 960,376 | 1,099,183 | 1,871,192 | 1,514,975 | 1,255,615 | 1,870,026 |
| Tokens per claim | 5,928 | 6,785 | 11,551 | 9,352 | 7,751 | 11,543 |
| Mean Judge prompt tokens | 2,454 | 2,798 | 4,678 | 3,796 | 3,095 | 4,673 |
| Largest prompt (limit 8,192) | 4,386 | 5,452 | 7,569 | 6,727 | 6,658 | 7,890 |
| Mean / longest Judge reply (tokens) | 108.8 / 190 | 111.0 / 190 | 115.1 / 210 | 113.2 / 207 | 112.9 / 192 | 115.2 / 197 |
| Mean end-to-end latency (s) | 23.10 | 25.55 | 39.00 | 32.60 | 31.06 | 39.06 |
| Median end-to-end latency (s) | 22.69 | 25.45 | 38.55 | 32.62 | 30.67 | 38.75 |

There were no LLM-budget exhaustions or context-window failures. Call counts are identical to v3, since the
Judge change affects only the final call. The v4 Judge prompt is longer, so Judge prompts grew by about 96
tokens; the largest prompt (F) came within 302 tokens of the context limit.

**Judge replies are now short.** The mean reply is 109–115 tokens, against a v3 median of 131.5–143, and the
longest is 210. The 512-token limit that caused 66 v3 failures was never approached.

Latency is lower than on v3 for every arm (for example A 23.10 s against 26.64 s), but the jobs ran on
different GPUs and dates, so cost should still be compared by calls and tokens, not latency.

## Agent behaviour

Retrieval is identical to the v3 runs, so the retrieval figures match the v3 report. Verdict use of
Challenge documents is new.

| Behaviour measure | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: |
| Challenge results already found by Support | 695 / 810 (85.8%) | 0 / 810 (0%) | 298 / 810 (36.8%) | 590 / 810 (72.8%) | 0 / 810 (0%) |
| Challenge-only documents | 115 | 810 | 512 | 220 | 810 |
| Claims with a Challenge-only document | 88 | 162 | 158 | 123 | 162 |
| Mean merged documents per claim (A: 5.00) | 5.71 | 10.00 | 8.16 | 6.36 | 10.00 |
| Challenge-only gold documents | 2 | 5 | 3 | 1 | 4 |
| Gold coverage after all agents (of 118; Support alone 110) | 112 | 115 | 113 | 111 | 114 |
| Challenge-only documents the Verifier labelled SUPPORT / CONTRADICT | 4 / 4 | 18 / 29 | 21 / 20 | 6 / 7 | 28 / 35 |
| Challenge-only documents the Verifier labelled INSUFFICIENT | 106 (93.0%) | 760 (94.2%) | 469 (92.0%) | 207 (94.1%) | 744 (92.2%) |
| Claims with a Challenge-only document verified as direct evidence | 7 | 27 | 29 | 11 | 36 |
| **Verdicts citing a Challenge-only document (v4.1)** | **4** | **19** | **20** | **11** | **27** |
| Verdicts citing a Challenge-only document (v3, for reference) | 19 | 32 | 36 | 27 | 42 |
| Abstracts excluded after an invalid Verifier response | 1 | 3 | 2 | 0 | 3 |

The v4.1 Judge cites fewer documents, so fewer verdicts cite Challenge-only documents than on v3.
Percentages use documents with a valid Verifier response. SciFact annotations are not exhaustive, so a
non-gold document may still be relevant, and these labels were not audited against the abstracts.

### How Challenge changed verdicts against A

| Arm | Claims gained against A | Claims lost against A |
| --- | --- | --- |
| B | 9 | 494 |
| C | none | 197, 494, 866 |
| D | 951 | 494, 856 (failure), 866 |
| E | 571 | 494, 680, 866 |
| F | 951 | 494, 555, 866, 1074 |

- **Claim 494 (INSUFFICIENT) was lost by all five Challenge arms through one document.** Every arm's
  Challenge search found document 34873974, which Support did not find. The Verifier labelled it
  CONTRADICT, citing sentence 14, and every Judge cited it to answer CONTRADICT. SciFact lists this
  document as the claim's source (`cited_doc_ids`) but annotates no rationale, so the gold label is
  INSUFFICIENT. Support's five documents were all off-topic (goitre, ADHD, headache, lung cancer).
  Sentence 14 says the test "may not be appropriate for patients who present less than three hours after
  symptom onset", which appears to agree with the claim rather than contradict it. On this reading,
  Challenge found the relevant source document and the Verifier assigned the wrong polarity; this was
  checked by reading the abstract, not by a domain expert.
- **Claim 951 (SUPPORT) was gained by D and F through one document.** Both found document 4422723,
  which Support did not; the Verifier labelled it SUPPORT and the verdict became correct. This document
  is also not gold, so a non-annotated Challenge document produced a correct verdict here.
- **Claim 866 (INSUFFICIENT) was lost by C–F without citing a Challenge-only document.** They all cited
  documents Support had found and answered SUPPORT, whereas A and B said INSUFFICIENT. The extra evidence
  changed the Judge's decision even though it was not cited.

### Claims with a Challenge-only gold document

| Claim | Gold | Added by | Verifier label | Outcome |
| --- | --- | --- | --- | --- |
| 1128 | SUPPORT | C, D, F | INSUFFICIENT | Correct in every arm, including A |
| 1224 | SUPPORT | C | SUPPORT | Correct in every arm, including A |
| 69 | CONTRADICT | B, C, D, F | CONTRADICT | Correct in every arm, including A |
| 1311 | SUPPORT | B, C, E, F | SUPPORT | Correct in every arm, including A (A said INSUFFICIENT on v3) |
| 963 | CONTRADICT | C, D, F | SUPPORT | Wrong in every arm. C, D and F cited the gold document and answered SUPPORT; A, B and E said INSUFFICIENT. |

**Finding gold documents gained no claim.** Four of the five claims were already correct without
Challenge. On claim 963, finding the gold CONTRADICT document turned an INSUFFICIENT error into an
opposite-polarity error in C, D and F, because the Verifier labelled it SUPPORT. On v3 this happened only
in D, because C and F failed on this claim.

## Comparison with Standard RAG and Iterative RAG (A40 reruns)

| Metric | Standard RAG | Iterative RAG | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct verdicts (of 162) | 119 | 118 | 121 | 121 | 118 | 119 | 119 | 118 |
| Failure-aware accuracy | 73.46% | 72.84% | 74.69% | 74.69% | 72.84% | 73.46% | 73.46% | 72.84% |
| Macro-F1 | 0.7367 | 0.7377 | 0.7355 | 0.7343 | 0.7173 | 0.7231 | 0.7231 | 0.7156 |
| SUPPORT F1 | 0.8160 | 0.7869 | 0.8148 | 0.8235 | 0.7941 | 0.8058 | 0.8088 | 0.8000 |
| CONTRADICT F1 | 0.7500 | 0.7711 | 0.6747 | 0.6747 | 0.6588 | 0.6506 | 0.6747 | 0.6667 |
| INSUFFICIENT F1 | 0.6441 | 0.6552 | 0.7170 | 0.7048 | 0.6990 | 0.7129 | 0.6857 | 0.6800 |
| Failures | 1 | 3 | 0 | 0 | 0 | 1 | 0 | 0 |
| LLM calls per claim | 1.00 | 5.11 | 6.00 | 6.71 | 11.00 | 9.16 | 8.36 | 11.00 |

**ARGUS trades CONTRADICT for INSUFFICIENT.** Every ARGUS arm is better than both baselines on
INSUFFICIENT and worse on CONTRADICT, so macro-F1 ends up about level. The baselines do not use the ARGUS
Judge and were run on 22 October, not alongside these arms.

### Paired per-claim analysis

Exact two-sided McNemar tests on per-claim correctness, with failures counted as incorrect. Holm
correction is over the comparisons within each table. Intervals are paired bootstrap 95% intervals for
the accuracy difference (10,000 resamples, seed 42), given as arm minus base.

**Between arms, all 162 claims (primary; Holm over nine comparisons):**

| Base vs arm | Only base correct | Only arm correct | p | Holm p | Difference, points (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| A vs B | 1 | 1 | 1.000 | 1.000 | +0.00 [−1.85, +1.85] |
| A vs C | 3 | 0 | 0.250 | 1.000 | −1.85 [−4.32, +0.00] |
| A vs D | 3 | 1 | 0.625 | 1.000 | −1.23 [−3.70, +1.23] |
| A vs E | 3 | 1 | 0.625 | 1.000 | −1.23 [−3.70, +1.23] |
| A vs F | 4 | 1 | 0.375 | 1.000 | −1.85 [−4.94, +0.62] |
| B vs C | 3 | 0 | 0.250 | 1.000 | −1.85 [−4.32, +0.00] |
| B vs D | 3 | 1 | 0.625 | 1.000 | −1.23 [−3.70, +1.23] |
| B vs E | 3 | 1 | 0.625 | 1.000 | −1.23 [−3.70, +1.23] |
| B vs F | 4 | 1 | 0.375 | 1.000 | −1.85 [−4.94, +0.62] |

Only 1 to 5 claims differ between any two arms. Restricting to the 161 claims valid in every arm (which
drops D's one failure) gives the same conclusion: correct A 120, B 120, C 117, D 119, E 118, F 117, with
all Holm p = 1.000.

**Same arm, v3 against v4.1 (Holm over six comparisons):**

| Arm | Only v3 correct | Only v4.1 correct | p | Holm p | Difference, points (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 0 | 6 | 0.031 | **0.031** | +3.70 [+1.23, +6.79] |
| B | 0 | 10 | 0.002 | **0.006** | +6.17 [+3.09, +9.88] |
| C | 2 | 20 | <0.001 | **<0.001** | +11.11 [+6.17, +16.67] |
| D | 3 | 18 | 0.002 | **0.006** | +9.26 [+4.32, +14.81] |
| E | 1 | 12 | 0.003 | **0.007** | +6.79 [+3.09, +11.11] |
| F | 1 | 20 | <0.001 | **<0.001** | +11.73 [+6.79, +17.28] |

Most of the gain comes from former failures: of the v3 failures, 4 of 4 (A), 8 of 10 (B), 20 of 22 (C),
18 of 20 (D), 10 of 12 (E) and 20 of 22 (F) became correct. On claims valid in both versions, only 2–4
verdicts per arm changed label.

**Against the baselines (Holm over twelve comparisons):**

| Base vs arm | Only base correct | Only arm correct | p | Holm p | Difference, points (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| Standard vs A | 10 | 12 | 0.832 | 1.000 | +1.23 [−4.32, +6.79] |
| Standard vs B | 11 | 13 | 0.839 | 1.000 | +1.23 [−4.94, +7.41] |
| Standard vs C | 12 | 11 | 1.000 | 1.000 | −0.62 [−6.79, +4.94] |
| Standard vs D | 12 | 12 | 1.000 | 1.000 | +0.00 [−6.17, +5.56] |
| Standard vs E | 11 | 11 | 1.000 | 1.000 | +0.00 [−5.56, +5.56] |
| Standard vs F | 13 | 12 | 1.000 | 1.000 | −0.62 [−6.79, +5.56] |
| Iterative vs A | 11 | 14 | 0.690 | 1.000 | +1.85 [−4.32, +8.02] |
| Iterative vs B | 11 | 14 | 0.690 | 1.000 | +1.85 [−4.32, +8.02] |
| Iterative vs C | 12 | 12 | 1.000 | 1.000 | +0.00 [−6.17, +6.17] |
| Iterative vs D | 13 | 14 | 1.000 | 1.000 | +0.62 [−5.56, +6.79] |
| Iterative vs E | 12 | 13 | 1.000 | 1.000 | +0.62 [−5.56, +6.79] |
| Iterative vs F | 14 | 14 | 1.000 | 1.000 | +0.00 [−6.17, +6.17] |

The systems disagree on 22–28 claims each, but the disagreements split almost evenly, so none differs
significantly.

**Caveats:**

- Nine comparisons on 162 claims have limited power. A true difference of a few claims could not be
  detected here; the intervals show that differences of more than about 5 points between arms are
  unlikely on this set.
- The v3 against v4.1 comparison changes only the Judge, but the v4 and v4.1 changes were motivated by
  Judge failures seen on internal validation runs, so the size of that gain may be optimistic.

## Run-to-run repeatability

Retrieval was identical to the v3 runs on all 972 claim runs. Verifier raw responses were identical on
954 of the 972 claim runs (A 157, B 156, C 158, D 160, E 162, F 161 of 162), on different A40s and dates.
Where the text differed, the Verifier label changed in one case only: on claim 866 in C, one document
Support had found went from CONTRADICT to INSUFFICIENT. **So the inputs to the Judge were essentially the
same as on v3, and the differences in this report come from the Judge.** The small text differences show
that responses are not fully byte-identical across GPUs, unlike the B repeat in the v3 report.

## Failure investigation

| Failure type | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `invalid_json`: Judge reached the 512-token limit | 0 | 0 | 0 | 0 | 0 | 0 |
| `invalid_response`: out-of-range sentence | 0 | 0 | 0 | 1 | 0 | 0 |
| **Total** | 0 | 0 | 0 | 1 | 0 | 0 |
| v3 total, for reference | 4 | 10 | 22 | 20 | 12 | 22 |

- **The one failure** is D on claim 856 (gold CONTRADICT, 9 documents): the Judge cited a sentence that
  does not exist in document 7975937, using 128 completion tokens. The same claim was correct in A on v4.1
  and in D on v3. It was not retried, reclassified or removed.
- **Citation normalization was used in 5–9 claims per arm** (A 8, B 9, C 8, D 6, E 5, F 8), where the Judge
  cited the same document in two entries. The v4 Judge would reject those replies. Of these claims, 8, 8,
  5, 5, 5 and 5 were correct. Plain v4 was not run, so whether it would have produced the same replies is
  inferred from the identical prompt and schema, not measured.
- No retrieval call, counter-hypothesis call or Verifier failure stopped a claim. Invalid Verifier
  responses (B 1, C 3, D 2, F 3) excluded that document and the claim continued.

## Development history

**Merge.** `feature/argus-multi-agent-rag`, which adds the v4.1 Judge, was merged into
`feat/challenge-retrieval` as commit `8d331ba`. Conflicts in `scripts/run_argus.py` and
`src/argus/agents/argus.py` were resolved by keeping both changes; v4.1 runs with a Challenge option are
recorded under their own implementation version. Two CLI tests were added for v4.1 with Challenge options.
All 134 tests passed locally with `ruff check` and `ruff format --check` passing, and the 73 tests in
`tests/test_argus.py` passed on the cluster.

**Smoke test.** Job 354364 ran arm D with v4.1 on the five development claims (1000, 1011, 0, 10, 1001),
which are disjoint from the 162 evaluation claims. All 5 succeeded, including claim 0, which failed on v3
at the 512-token limit. The output is on the cluster only, at
`outputs/argus-challenge-v41-smoke/D/20261105T105532Z`. No prompt or setting was changed after the smoke
test.

**Order.** C–F were run first; A and B were run afterwards once it was confirmed that no v4.1 runs of
them existed in the repository.

## Findings relevant to subsequent ARGUS research

- **With a reliable Judge, Challenge retrieval does not improve verdicts on this set.** No arm beats A or B,
  and the more distinct arms (C, F) are slightly lower, though not significantly.
- **Distinct evidence reaches the verdict but cuts both ways.** Challenge-only documents corrected some
  claims (951) and broke others (494). The net effect is zero or slightly negative.
- **The Verifier is the main limit on Challenge's value.** About 92–94% of Challenge-only documents are
  labelled INSUFFICIENT, and when Challenge does find a gold CONTRADICT document (claim 963), the Verifier
  labelled it SUPPORT in every arm that found it.
- **The ceiling is low.** Support already finds 110 of the 118 gold documents, so Challenge can add at most
  8, and the 5 it found gained no claim.
- **The main remaining error is over-calling CONTRADICT.** 13–14 INSUFFICIENT claims per arm are predicted
  CONTRADICT, including in A. This is a Verifier or Judge problem, not a retrieval one.
- **The v4.1 Judge fixed the v3 failure problem.** It should be the Judge for any later ARGUS comparison.
- **Counter-evidence is not yet worth its cost.** On this set, Challenge costs 12–83% more LLM calls with
  no accuracy gain, and ARGUS as a whole is level with Standard and Iterative RAG.

## Methodological limitations to retain in final report

- This is a training-derived internal validation result. The official labeled dev split was not used for
  tuning or evaluation.
- These same 162 claims were used for the v3 ablation, for screening which arms to run, and by the Judge
  changes. Any configuration chosen from these results must be confirmed on the official dev split.
- Results are single runs per arm on one Q4_K_M model, and the same model serves as Verifier and Judge.
- Plain v4 was not run here, so the separate effects of v4 and v4.1 are not measured.
- Standard and Iterative RAG were not rerun and do not use the ARGUS Judge; they were run on different dates.
- INSUFFICIENT means no annotated gold evidence, and SciFact annotations are not exhaustive. Documents such
  as 34873974 (claim 494, the claim's cited source) and 4422723 (claim 951) are not gold but changed
  verdicts. Only 34873974 was read against its abstract.
- The analysis used for this report, and the screening script, are not yet committed to the repository.

## Checks performed while preparing this report

Independently re-read and recomputed from the 162 prediction records of each run and local SciFact files:

- All 162 claim IDs are unique in each run, and the order is identical across A–F, the v3 runs and both
  baselines.
- Every recorded gold label in A–F equals the label re-derived from `claims_train.jsonl`.
- Accuracy, macro-F1, failure counts, call counts and citation figures agree with `metrics.json`.
- Variant, implementation version, Judge version, prompt hashes, model digest, generation settings and
  corpus fingerprint were confirmed from `metadata.json`; commit `8d331ba`, node, GPU and flags from each
  job's `environment.txt`.
- Agent settings and generator settings equal the v3 runs except for the Judge version (A's v3 metadata
  predates the Challenge fields, which here take their default values).
- Retrieved and merged document IDs equal the v3 runs on all 162 claims in every arm; Verifier labels and
  raw responses were compared as described above.
- The five smoke-test claims are absent from the 162 evaluation IDs.
- No Judge reply reached 512 completion tokens in any arm.
- McNemar tests, Holm correction and bootstrap intervals were computed from per-claim correctness.
- Document 34873974 (claim 494) was read against its abstract and the claim record.
- Not done: a semantic review of cited sentences, and a systematic audit of Challenge-only documents or
  Verifier labels against gold rationales or abstracts.

## Recommended next research steps

1. Keep each run's files and job records unchanged and backed up. Commit this summary, not the outputs.
2. Finish auditing the Challenge-only documents that changed verdicts (4422723 for claim 951, 4162857 for
   claim 963; 34873974 for claim 494 is done above), plus a sample of Challenge-only documents labelled
   SUPPORT or CONTRADICT. This decides whether the problem is the Verifier, the annotations, or the
   retrieved documents.
3. Look at the INSUFFICIENT claims predicted CONTRADICT, which are the largest error in every arm,
   including A. Improving the Verifier is more likely to help than further retrieval changes.
4. Commit the screening script and analysis under `scripts/`, with tests where they become reusable.
5. As a team, freeze one configuration across all systems with the v4.1 Judge, disclosing that it was
   chosen on these claims. Evaluate it once on the official development claims.
