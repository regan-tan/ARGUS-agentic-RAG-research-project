# ARGUS — Challenge Retrieval Ablation: Full Internal Validation Results

**Runs (v3 Judge):** arms B–F under `outputs/argus-challenge/<arm>/<run-id>/`, listed below. Arm A
reuses the preserved no-Challenge run `outputs/argus/20261026T151208Z/`.

**Sources:** local `metadata.json`, `metrics.json` and `predictions.jsonl` from each run; Slurm
accounting and `outputs/cluster/<job>/environment.txt` for the new jobs; gold labels and rationales
from the local official SciFact `claims_train.jsonl`. Standard RAG and Iterative RAG figures come from
their A40 reruns `outputs/rag/20261022T064540Z/` and `outputs/iterative_rag/20261022T072108Z/`.

**Status:** all five new runs completed; 162 of 162 intended claim IDs present, unique and in metadata
order. None was interrupted or resumed. **The combined run with the v4 Judge has not been done, so this
report says nothing about v4.**

**Follow-up:** all six arms were rerun with the v4.1 Judge; see `challenge-retrieval-v41-summary.md`,
which supersedes the next steps below.

| Arm | Variant | Challenge search | Challenge query | Skips Support's documents | Run | Job | Started (UTC+8) | Job wall-clock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | `no_challenge` | none | none | n/a | `20261026T151208Z` | 339339 | 26 Oct 23:10 | 1 h 13 min 46 s |
| B | `full` | dense | claim + fixed suffix | no | `20261104T115334Z` | 353149 | 4 Nov 19:53 | 1 h 38 min 57 s |
| C | `dense_claim_exclude` | dense | claim | yes | `20261104T115333Z` | 353150 | 4 Nov 19:53 | 1 h 58 min 07 s |
| D | `bm25_claim` | BM25 | claim | no | `20261104T133233Z` | 353151 | 4 Nov 21:32 | 2 h 06 min 36 s |
| E | `dense_counter_hypothesis` | dense | one LLM-written counter-hypothesis | no | `20261104T153914Z` | 353152 | 4 Nov 23:38 | 1 h 31 min 32 s |
| F | `hybrid_claim_exclude` | hybrid RRF | claim | yes | `20261104T142238Z` | 353287 | 4 Nov 22:22 | 1 h 57 min 29 s |

## Executive summary

This experiment asks whether a genuinely different Challenge retrieval contributes distinct,
scientifically relevant evidence, with the Judge held fixed at `argus-v3` so that retrieval is isolated
from the Judge fixes.

**Challenge can be made to retrieve distinct documents, but this did not improve verdicts.** Overlap with
Support fell from 85.8% (B) to 72.8% (E), 36.8% (D) and 0% (C, F). Challenge-only gold documents rose
from 2 (B) to 5 (C), 4 (F) and 3 (D), but fell to 1 (E). Support had already found 110 of the 118 gold
documents, so at most 8 could be added on this set.

**Every new arm had lower failure-aware accuracy than A and B:** A 70.99%, B 68.52%, E 66.67%, D 64.20%,
C 61.73%, F 61.11%. After Holm correction over nine comparisons, only C and F differ significantly from
A (adjusted p = 0.022 for both). No new arm differs significantly from B.

**Most of the loss comes from Judge failures, not wrong verdicts.** Judge replies hitting the 512-token
limit rose from 4 (B) to 19 (C, F), 17 (D) and 6 (E). Of the claims A answered correctly and C lost, 18
of 19 were failures; the same figures are 16 of 17 for D, 19 of 21 for F and 8 of 10 for E. These
retrieval arms are therefore penalized through a v3 Judge limit they trigger more often.

**On the 132 claims valid in all six arms, no difference is significant.** D scored highest (96/132),
against 92/132 for A and 93/132 for B. This subset is exploratory and omits 28 of the 61 INSUFFICIENT
claims. It does not show that D is better.

**The new gold documents rarely changed outcomes.** Across the five claims involved, three were already
correct in A. Claim 1311 was gained by every Challenge arm, including B. Claim 963 stayed wrong in every
arm, and in D the gold CONTRADICT document was verified as SUPPORT.

**The cost rose with no accuracy gain:** 6.71 LLM calls per claim in B against 8.36 (E), 9.16 (D) and
11.00 (C, F). This is a completed internal experiment, not a score on the official labeled SciFact
development split.

## Reproducibility configuration

| Item | Recorded value |
| --- | --- |
| Code | Branch `feat/challenge-retrieval`, commit `a4177fe` for arms B–F, recorded in each job's `environment.txt`, not in `metadata.json`. Arm A used commit `201cf4c`. Implementation version `argus-v3` for B, `argus-v3-challenge-retrieval` for C–F. |
| Dataset | Original AllenAI SciFact: 5,183 abstracts, 809 train claims |
| Evaluation set | 162 internal validation claims from train, seed 42 (66 SUPPORT, 35 CONTRADICT, 61 no annotated evidence); ordered IDs identical across all arms and both baselines |
| Support retrieval | Unchanged in every arm: BGE-small, FAISS `IndexFlatIP`, top-5, on the claim text |
| Retriever | `BAAI/bge-small-en-v1.5`, SHA `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; corpus fingerprint `bb15cae5d1852943d020806a38ae4a4154380d3f3ef898d4e9aecd92f7ac08ff` |
| Challenge retrieval | Top-5 per the arm table. BM25: `BM25Okapi` over the shared document text. Hybrid: RRF with default `candidate_pool=50`, `rrf_constant=60`. Exclusion requests `k + len(Support)` results, removes Support's documents and keeps the first 5. |
| Suffix (B only) | `exceptions limitations conflicting findings alternative outcomes` |
| Counter-hypothesis (E only) | One extra LLM call per claim that sees the claim only; prompt `argus-challenge-query-v1`, SHA-256 `17b66fae0b04f13a37d705a32147fa56fd773b5ca14894758d6c10fdbb7435fd`. A failed call fails the claim, with no fallback. |
| Evidence Verifier | Unchanged: `argus-v3`, SHA-256 `811de57019988a77a1e7697b5171a6fb550e5791c9943f14ed40d5a5b17b54ee` |
| Judge | Unchanged: `argus-v3`, SHA-256 `9a855df4b472fe1bab6301b75bef06345dbc02dd299097db4a3fd17e352e8d8b`; prompt hashes match in all arms |
| Generator | `qwen3:4b-instruct-2507-q4_K_M`, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`; Ollama 0.40.5 |
| Generation | temperature 0, top-k 40, `num_predict=512`, `num_ctx=8192`, non-streaming |
| Budgets | `retrieval_k=5` per agent, max 10 merged documents, max 12 LLM calls per claim |
| Hardware | SMU `crimson`, partition `student`, node `avenue`; 1× NVIDIA A40, 4 CPU cores, 24 GB RAM per job. GPU PCI: B, D, E `81:00.0`; C `C1:00.0`; F `01:00.0`. |

## Primary claim-verification results

Failed claims remain in denominators as FAILURE.

| Metric | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct verdicts | 115 | 111 | 100 | 104 | 108 | 99 |
| Failure-aware accuracy | 0.709877 | 0.685185 | 0.617284 | 0.641975 | 0.666667 | 0.611111 |
| Macro-F1 | 0.708512 | 0.692869 | 0.635111 | 0.655371 | 0.675653 | 0.617754 |
| Valid responses | 158 | 152 | 140 | 142 | 150 | 140 |
| Accuracy on each arm's own valid responses | 72.78% | 73.03% | 71.43% | 73.24% | 72.00% | 70.71% |
| Failures | 4 | 10 | 22 | 20 | 12 | 22 |

The "own valid responses" row compares different claim sets, so it is descriptive only.

### Per-class performance

| Gold class (number) | Metric | A | B | C | D | E | F |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SUPPORT (66) | Precision | 0.7910 | 0.7941 | 0.8060 | 0.7887 | 0.7826 | 0.7778 |
| | Recall | 0.8030 | 0.8182 | 0.8182 | 0.8485 | 0.8182 | 0.8485 |
| | F1 | 0.7970 | 0.8060 | 0.8120 | 0.8175 | 0.8000 | 0.8116 |
| CONTRADICT (35) | Precision | 0.5833 | 0.5833 | 0.5833 | 0.5714 | 0.5745 | 0.5714 |
| | Recall | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.7714 | 0.8000 |
| | F1 | 0.6747 | 0.6747 | 0.6747 | 0.6667 | 0.6585 | 0.6667 |
| INSUFFICIENT (61) | Precision | 0.7907 | 0.8056 | 0.7200 | 0.9091 | 0.7941 | 0.7895 |
| | Recall | 0.5574 | 0.4754 | 0.2951 | 0.3279 | 0.4426 | 0.2459 |
| | F1 | 0.6538 | 0.5979 | 0.4186 | 0.4819 | 0.5684 | 0.3750 |

The arms differ mainly on INSUFFICIENT, where most failures fall. SUPPORT and CONTRADICT F1 barely move.
CONTRADICT recall did not rise in any arm, so this experiment shows no gain in detecting contradiction.

### Confusion matrices

Rows are gold; each cell is predicted SUPPORT / CONTRADICT / INSUFFICIENT / FAILURE.

| Gold | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- |
| SUPPORT | 53 / 7 / 6 / 0 | 54 / 7 / 5 / 0 | 54 / 6 / 5 / 1 | 56 / 8 / 1 / 1 | 54 / 7 / 4 / 1 | 56 / 7 / 2 / 1 |
| CONTRADICT | 4 / 28 / 3 / 0 | 4 / 28 / 2 / 1 | 4 / 28 / 2 / 1 | 5 / 28 / 1 / 1 | 4 / 27 / 3 / 1 | 4 / 28 / 2 / 1 |
| INSUFFICIENT | 10 / 13 / 34 / 4 | 10 / 13 / 29 / 9 | 9 / 14 / 18 / 20 | 10 / 13 / 20 / 18 | 11 / 13 / 27 / 10 | 12 / 14 / 15 / 20 |

## Citation overlap with SciFact annotations

| Metric | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cited-document precision | 29.64% | 31.86% | 38.01% | 37.15% | 31.30% | 36.00% |
| Cited-document recall | 90.68% | 92.31% | 96.52% | 93.04% | 93.91% | 93.91% |
| Cited-sentence precision | 7.70% | 10.13% | 24.28% | 20.80% | 10.08% | 24.21% |
| Cited-sentence recall | 52.71% | 54.46% | 54.50% | 54.50% | 55.00% | 53.50% |
| Cited sentence references | 1,390 | 1,086 | 449 | 524 | 1,091 | 442 |

Values match `metrics.json`. Recall denominators count only claims with a valid response, and the valid
sets differ. **The higher sentence precision in C, D and F should not be read as better citation
behaviour.** Their long Judge outputs are the ones cut off at 512 tokens and removed from these figures,
so this is probably survivorship; it was not separately tested. These are annotation-overlap measures,
not semantic-faithfulness judgments.

## Efficiency and resource usage

| Measurement | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM calls | 972 | 1,087 | 1,782 | 1,484 | 1,354 | 1,782 |
| LLM calls per claim | 6.00 | 6.71 | 11.00 | 9.16 | 8.36 | 11.00 |
| Verifier calls | 810 | 925 | 1,620 | 1,322 | 1,030 | 1,620 |
| Counter-hypothesis calls | 0 | 0 | 0 | 0 | 162 | 0 |
| Total recorded tokens | 955,858 | 1,094,760 | 1,868,862 | 1,512,548 | 1,251,966 | 1,867,978 |
| Tokens per claim | 5,900 | 6,758 | 11,536 | 9,337 | 7,728 | 11,531 |
| Mean Judge prompt tokens | 2,358 | 2,702 | 4,582 | 3,700 | 2,999 | 4,577 |
| Largest prompt (limit 8,192) | 4,290 | 5,356 | 7,473 | 6,631 | 6,562 | 7,794 |
| Mean end-to-end latency (s) | 26.64 | 36.50 | 43.63 | 46.74 | 33.75 | 43.40 |
| Median end-to-end latency (s) | 25.60 | 34.41 | 42.17 | 44.20 | 32.30 | 41.46 |

There were no LLM-budget exhaustions or context-window failures. The largest prompts came within 398
tokens of the context limit, so more evidence would leave little margin.

**Compare cost by calls and tokens, not latency.** Latency was affected by the run environment: the
rerun of B took 5,913 s of claim time against 4,643 s for the identical original, with byte-identical
responses. Mean Verifier latency also ranged from about 2.95 s to 3.76 s between arms with similar
prompts.

## Agent behaviour

| Behaviour measure | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: |
| Challenge results already found by Support | 695 / 810 (85.8%) | 0 / 810 (0%) | 298 / 810 (36.8%) | 590 / 810 (72.8%) | 0 / 810 (0%) |
| Challenge-only documents | 115 | 810 | 512 | 220 | 810 |
| Claims with a Challenge-only document | 88 | 162 | 158 | 123 | 162 |
| Mean merged documents per claim (A: 5.00) | 5.71 | 10.00 | 8.16 | 6.36 | 10.00 |
| Challenge-only gold documents | 2 | 5 | 3 | 1 | 4 |
| of which SUPPORT / CONTRADICT | 1 / 1 | 3 / 2 | 1 / 2 | 1 / 0 | 2 / 2 |
| Gold coverage after all agents (of 118; Support alone 110) | 112 (94.92%) | 115 (97.46%) | 113 (95.76%) | 111 (94.07%) | 114 (96.61%) |
| Challenge-only documents the Verifier labelled SUPPORT | 4 | 18 | 21 | 6 | 28 |
| Challenge-only documents the Verifier labelled CONTRADICT | 4 | 29 | 20 | 7 | 35 |
| Challenge-only documents the Verifier labelled INSUFFICIENT | 106 (93.0%) | 760 (94.2%) | 469 (92.0%) | 207 (94.1%) | 744 (92.2%) |
| Claims with a Challenge-only document verified as direct evidence | 7 | 27 | 29 | 11 | 36 |
| Verdicts citing a Challenge-only document | 19 | 32 | 36 | 27 | 42 |
| Abstracts excluded after an invalid Verifier response | 1 | 3 | 2 | 0 | 3 |

Percentages use documents with a valid Verifier response. SciFact annotations are not exhaustive, so a
non-gold document may still be relevant, and these labels were not audited against the abstracts.

**The ceiling is low.** Support missed only 8 gold documents here, on claims 1109 (2), 1128, 1224, 1311,
69, 856 and 963, and only 3 of the 8 are CONTRADICT. No arm found a document missed on claim 1109 or 856.

**Gold-document yield is small.** C needed 810 new documents to add 5 gold ones, and D needed 512 to add 3.

**E's query did not make it a strong search for counter-evidence.** All 162 queries were generated with
no failures, and they keep the claim's specific terms while reversing or negating it. For example, claim
963 ("...are less stable than...") became "...are more stable than...". Even so, 72.8% of E's results
were Support's documents and it found the fewest gold documents, including no CONTRADICT gold document.
Direct verdict effects of E were not separately audited.

### Claims with a Challenge-only gold document

| Claim | Gold | Added by | Outcome |
| --- | --- | --- | --- |
| 1128 | SUPPORT | C, D, F | Already correct in A; remained correct |
| 1224 | SUPPORT | C | Already correct in A; remained correct |
| 69 | CONTRADICT | B, C, D, F | Already correct in A; remained correct |
| 1311 | SUPPORT | B, C, E, F | A said INSUFFICIENT; correct in all Challenge arms, including D, which did not retrieve this document |
| 963 | CONTRADICT | C, D, F | A, B, C, E and F said INSUFFICIENT. D said SUPPORT: the Verifier labelled the gold CONTRADICT document as SUPPORT, and the Judge followed it. |

## Comparison with Standard RAG and Iterative RAG (A40 reruns)

| Metric | Standard RAG | Iterative RAG | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct verdicts (of 162) | 119 | 118 | 115 | 111 | 100 | 104 | 108 | 99 |
| Failure-aware accuracy | 73.46% | 72.84% | 70.99% | 68.52% | 61.73% | 64.20% | 66.67% | 61.11% |
| Macro-F1 | 0.7367 | 0.7377 | 0.7085 | 0.6929 | 0.6351 | 0.6554 | 0.6757 | 0.6178 |
| LLM calls per claim | 1.00 | 5.11 | 6.00 | 6.71 | 11.00 | 9.16 | 8.36 | 11.00 |

Neither baseline was beaten by any arm. No significance test against the baselines is reported here,
because the v3 Judge failures would dominate it. They should be compared once a Judge-reliable
configuration is frozen. These runs were not simultaneous.

### Paired per-claim analysis

Exact two-sided McNemar tests on per-claim correctness, with failures counted as incorrect. Holm
correction is over the nine comparisons within each table. Intervals are paired bootstrap 95% intervals
for the accuracy difference (10,000 resamples, seed 42), and differences are given as arm minus base.

**All 162 claims (primary):**

| Base vs arm | Only base correct | Only arm correct | p | Holm p | Difference, points (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| A vs B | 5 | 1 | 0.219 | 0.568 | −2.47 [−5.56, +0.00] |
| A vs C | 19 | 4 | 0.003 | **0.022** | −9.26 [−14.81, −3.70] |
| A vs D | 17 | 6 | 0.035 | 0.173 | −6.79 [−12.35, −1.23] |
| A vs E | 10 | 3 | 0.092 | 0.369 | −4.32 [−8.64, +0.00] |
| A vs F | 21 | 5 | 0.002 | **0.022** | −9.88 [−16.05, −4.32] |
| B vs C | 15 | 4 | 0.019 | 0.118 | −6.79 [−11.73, −1.85] |
| B vs D | 14 | 7 | 0.189 | 0.568 | −4.32 [−9.88, +1.23] |
| B vs E | 8 | 5 | 0.581 | 0.581 | −1.85 [−6.17, +2.47] |
| B vs F | 17 | 5 | 0.017 | 0.118 | −7.41 [−12.96, −1.85] |

**The 132 claims valid in all six arms (exploratory, chosen after seeing the results):** 65 SUPPORT, 34
CONTRADICT and 33 INSUFFICIENT. Correct: A 92, B 93, C 94, D 96, E 92, F 93.

| Base vs arm | Only base correct | Only arm correct | p | Holm p | Difference, points (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| A vs B | 0 | 1 | 1.000 | 1.000 | +0.76 [+0.00, +2.27] |
| A vs C | 1 | 3 | 0.625 | 1.000 | +1.52 [−1.52, +4.55] |
| A vs D | 1 | 5 | 0.219 | 1.000 | +3.03 [+0.00, +6.82] |
| A vs E | 2 | 2 | 1.000 | 1.000 | +0.00 [−3.03, +3.03] |
| A vs F | 2 | 3 | 1.000 | 1.000 | +0.76 [−2.27, +3.79] |
| B vs C | 1 | 2 | 1.000 | 1.000 | +0.76 [−1.52, +3.79] |
| B vs D | 1 | 4 | 0.375 | 1.000 | +2.27 [−0.76, +5.30] |
| B vs E | 2 | 1 | 1.000 | 1.000 | −0.76 [−3.79, +1.52] |
| B vs F | 2 | 2 | 1.000 | 1.000 | +0.00 [−3.03, +3.03] |

**Caveats:**

- The 132-claim subset removes claims where the Judge failed in any arm, which is mostly INSUFFICIENT
  claims. It therefore cannot estimate what these arms would score with a reliable Judge.
- Excluding documents guarantees more evidence, so C and F's significant losses against A reflect the
  retrieval arm and this Judge together. They do not establish that the retrieved documents are harmful.
- Nine comparisons on 162 claims have limited power, and the arm-level differences were not predeclared
  as a single primary hypothesis.

## Run-to-run repeatability

Arm B repeats the original full run on the same node type, with different code but B's behaviour
unchanged. **Every raw response on all 162 claims matched** the original
`outputs/argus/20261026T151038Z/`, on a different physical A40 and nine days later. This shows that the
new options are inactive when not selected. It also means differences between arms in this experiment
are unlikely to come from model nondeterminism, for this one configuration on this hardware. It does not
measure variation across other hardware or runtimes.

## Failure investigation

All failures occurred at the final Judge call. No retrieval call or counter-hypothesis call failed, and
no Verifier failure stopped a claim. No call hit the context limit or the LLM-call budget.

| Failure type | A | B | C | D | E | F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `invalid_json`: Judge reached the 512-token limit | 1 | 4 | 19 | 17 | 6 | 19 |
| `invalid_response`: empty `sentence_indices` | 2 | 4 | 3 | 2 | 5 | 2 |
| `invalid_response`: out-of-range sentence | 1 | 2 | 0 | 1 | 1 | 1 |
| **Total** | 4 | 10 | 22 | 20 | 12 | 22 |
| Failures on INSUFFICIENT gold claims | 4 | 9 | 20 | 18 | 10 | 20 |
| Of the failed claims, correct in A | — | — | 18 | 16 | 8 | 19 |
| Of the failed claims, correct in B | — | — | 14 | 13 | 6 | 15 |

- **Truncation grows with evidence.** In every run, `invalid_json` occurs on exactly the claims where the
  Judge reached 512 completion tokens. The Judge's median output grew little (131.5 tokens in B, 137.5 in
  C, 139.5 in D, 143 in F), so the extra failures come from the tail of long replies, not from all replies
  growing.
- **Failures concentrate on INSUFFICIENT claims,** as in Milestone 5. This explains why INSUFFICIENT
  recall is what falls.
- **Different arms fail on overlapping claims.** 30 claims failed in at least one arm; 4 failed in all of
  B–F, and 21 failed in both C and F.
- **Wrong-verdict losses are few.** Against A, C and D each had 1 loss with a valid but wrong verdict
  (claim 866), E had 2 (680, 866) and F had 2 (866, 1074). On claim 866 all four arms predicted SUPPORT on
  an INSUFFICIENT gold claim.

Failed claims were not retried, reclassified or removed from the primary results.

## Development history

**No-LLM screen.** Before any LLM run, retrieval options were screened on the 647 train-derived
development claims, not the 162 validation claims. Support found 417 of 446 gold documents (93.5%).

| Option | Overlap with Support | New documents per claim | New gold | New CONTRADICT gold |
| --- | ---: | ---: | ---: | ---: |
| Dense + suffix (current) | 84.9% | 0.75 | 2 | 0 |
| BM25, claim | 38.9% | 3.06 | 12 | 2 |
| BM25 + suffix | 37.4% | 3.13 | 12 | 2 |
| Hybrid, claim | 56.4% | 2.18 | 9 | 1 |
| Hybrid + suffix | 55.1% | 2.24 | 8 | 1 |
| Dense, claim, exclude | 0% | 5.00 | 17 | 3 |
| Dense + suffix, exclude | 0% | 5.00 | 16 | 3 |
| BM25, claim, exclude | 0% | 5.00 | 16 | 2 |
| Hybrid, claim, exclude | 0% | 5.00 | 20 | 3 |

The suffix added almost nothing, so C, D and F search with the claim alone. Arm E cannot be screened
without the LLM.

**Smoke tests.** Jobs 353134–353137 ran C, D, E and F on the five development claims (1000, 1011, 0, 10,
1001), which are disjoint from the 162 evaluation claims. C, D and F each succeeded on 4 of 5 and E on 5
of 5. All three failures were claim 0, where the Judge hit 512 tokens with 10 documents; the original B
run answered it with 6 documents and 350 tokens. This risk was seen before the full runs, which then
proceeded on v3 so that retrieval stayed isolated from the Judge fix. No prompt or setting was changed
after the smoke tests.

## Findings relevant to subsequent ARGUS research

- **Distinct retrieval is achievable.** Exclusion, BM25 and hybrid retrieval all reduce overlap with
  Support and add more gold documents than the original suffix.
- **Distinct retrieval has not been shown to improve verdicts.** The gold ceiling is low, about 92% to 94%
  of Challenge-only documents were labelled INSUFFICIENT, and contradiction recall did not rise.
- **A counter-hypothesis query alone is not enough under dense retrieval.** E produced plausible queries
  but mostly retrieved Support's documents, and it found the fewest gold documents.
- **More evidence exposes the Judge's limit.** With 512 completion tokens, v3 fails far more often at 8 to
  10 documents. This confounds the retrieval comparison and is the main reason to test these arms with a
  Judge that bounds its citations.
- **Verification can undo a retrieval gain.** On claim 963, finding the gold document in D led to an
  opposite-polarity error because the Verifier misread it. This is one case, not an estimate.

## Methodological limitations to retain in final report

- This is a training-derived internal validation result. The official labeled dev split was not used for
  tuning or evaluation.
- The primary comparison is confounded by Judge failures, and the 132-claim comparison is post hoc and
  under-represents INSUFFICIENT claims. Neither one alone gives the effect of retrieval under a reliable
  Judge.
- **Choosing a "best" arm from these 162 claims and then reporting it on the same claims is selection on
  the validation set.** Any v4 run of a chosen arm must disclose this. Only a frozen configuration on the
  official dev split gives a clean estimate.
- Arm A was not rerun. It used earlier code but the same prompt hashes, model digest, corpus fingerprint
  and claim IDs, and the B rerun supports treating these runs as comparable. A was run on a different
  date from the new arms.
- INSUFFICIENT means no annotated gold evidence, and SciFact annotations are not exhaustive. Challenge-only
  gold counts therefore understate possibly relevant documents, and these Verifier labels are unaudited.
- The screen used different claims from the evaluation and measures only retrieval. It cannot be read
  against verdict results.
- Results are single runs per arm on one Q4_K_M model, and the same model serves as Verifier and Judge.
- The screening script and the analysis used for this report are not yet committed to the repository.

## Checks performed while preparing this report

Independently re-read and recomputed from the 162 prediction records of each run and local SciFact files:

- All 162 claim IDs are unique in each run, and the order is identical across A–F and both baselines.
- Every recorded gold label in A–F equals the label re-derived from `claims_train.jsonl`.
- Accuracy, macro-F1, failure counts and call counts agree with the values shown from `metrics.json`.
- Prompt hashes, model digest, corpus fingerprint and Ollama runtime were confirmed from `metadata.json`
  and `environment.txt`; commit `a4177fe` was confirmed from the job files inspected.
- The five smoke-test claims are absent from the 162 evaluation IDs.
- `invalid_json` was confirmed to coincide exactly with Judge replies at 512 tokens in all arms.
- B's raw responses and evidence were compared with the original run on all 162 claims.
- McNemar tests, Holm correction and bootstrap intervals were computed from per-claim correctness.
- Not done: a semantic review of cited sentences, an audit of Challenge-only documents or Verifier labels
  against gold rationales or abstracts, and any v4 run.

## Recommended next research steps

1. Keep each run's files and job records unchanged and backed up. The smoke-test runs are on the cluster
   only; commit this summary, not the outputs.
2. Run a selected arm with `--judge-version v4`, separately labelled, and compare with the v4 full and
   no-Challenge runs. Check that those used the same settings. D is the natural candidate because it adds
   fewer documents than C and F and scored highest on the shared claims, but this is a weak basis. C is
   worth including, since it is the simplest control and found the most gold documents.
3. Audit a sample of Challenge-only documents the Verifier labelled SUPPORT or CONTRADICT against their
   abstracts, plus claims 963 and 866, to see whether they are relevant evidence.
4. Commit the screening script and analysis under `scripts/`, with tests where they become reusable.
5. Freeze one configuration across all systems before evaluating once on the official development claims.
