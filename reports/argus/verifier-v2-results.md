# ARGUS — Verifier v2: Full Internal Validation Results

**Runs (Verifier v2, Judge v4.1):** arms A, B, D and F under `outputs/argus-verifier-v2/<arm>/<run-id>/`, listed
below. Each is compared with the same arm's Verifier v1 run under `outputs/argus-challenge-v41/<arm>/<run-id>/`,
described in [challenge-retrieval-v41-summary.md](challenge-retrieval-v41-summary.md).

**Sources:** local `metadata.json`, `metrics.json` and `predictions.jsonl` from each run;
`outputs/cluster/<job>/environment.txt` and the job's `.out` file; gold labels from the local official SciFact
`claims_train.jsonl`. Standard RAG and Iterative RAG figures come from their A40 reruns
`outputs/rag/20261022T064540Z/` and `outputs/iterative_rag/20261022T072108Z/`.

**Status:** all four runs completed; 162 of 162 intended claim IDs present, in metadata order. None was
interrupted or resumed. Arms C and E were not rerun with Verifier v2.

| Arm | Variant | Challenge search | v2 run | Job | Started (UTC+8) | Run duration |
| --- | --- | --- | --- | --- | --- | --- |
| A | `no_challenge` | none | `20261106T143101Z` | 355683 | 6 Nov 22:31 | 1 h 15 min |
| B | `full` | dense, claim + suffix | `20261106T143057Z` | 355684 | 6 Nov 22:30 | 1 h 23 min |
| D | `bm25_claim` | BM25, claim | `20261106T155429Z` | 355746 | 6 Nov 23:54 | 1 h 47 min |
| F | `hybrid_claim_exclude` | hybrid RRF, claim, skips Support's documents | `20261106T154638Z` | 355745 | 6 Nov 23:46 | 2 h 08 min |

Run duration is from the run ID's start time to `run.updated_utc` in `metadata.json`. D and F ran at the same
time on the same node, so their durations are not comparable with A and B.

## Executive summary

The error audit ([challenge-error-audit.md](challenge-error-audit.md)) found the Evidence Verifier to be the
primary failure in 7 of 11 inspected cases: effect direction flipped, a different subject treated as
contradiction, and negated claims mishandled. Verifier v2 adds four checks to the v1 prompt and changes
nothing else.

**Verifier v2 improved every arm tested.** Correct verdicts: A 121 → 126, B 121 → 127, D 119 → 124,
F 118 → 128. Macro-F1 rose in every arm, by 0.039 to 0.071.

**No improvement is significant after correction.** F alone passes p < 0.05 (exact McNemar p = 0.041; 95%
bootstrap CI +0.6 to +11.7 points), but its Holm-adjusted p over the four arms is 0.166. A, B and D have
Holm-adjusted p = 0.922.

**The four arms are not independent evidence.** They share the Support agent and most papers, and the
changes fall largely on the same claims: 11 claims were fixed by v2 in all four arms and 5 were broken in all
four.

**v2 works by saying SUPPORT less often, not by labelling gold papers better.** On papers verified by both
versions, the commonest change is SUPPORT → INSUFFICIENT (25 to 40 pairs per arm). INSUFFICIENT claims
predicted SUPPORT fell from 10–13 to 5–7 per arm, but SUPPORT claims predicted INSUFFICIENT rose from 2–4 to
6–9. On gold papers v2 agreed with the gold label slightly *less* often than v1 in every arm, and the number
with the direction flipped did not fall (11–12 → 12).

**With v2, ARGUS has the highest accuracy on this set:** 124–128 correct, against 119 for Standard RAG and
118 for Iterative RAG. None of these differences is significant (smallest uncorrected p = 0.087, F against
Iterative RAG). The v2 arms are within 4 claims of each other, so the Challenge agent still does not
measurably change accuracy.

**Cost:** about 19% more tokens per claim in every arm, from the longer prompt; LLM calls per claim are
unchanged. This is a completed internal experiment on claims that were also used to design v2, not a score on
the official labeled SciFact development split.

## What changed

Verifier v2 is opt-in (`--verifier-version v2`; default `v1`). Its system prompt is the v1 prompt with the
version label replaced and this text inserted before the final schema instruction:

> Before choosing a label, check the following. Within the one-sentence rationale, state whether the subject
> matches and the direction of effect in the claim and paper.
> 1. Subject: does the paper study the same entity, intervention, population, and outcome as the claim? If it
> studies a different one, use INSUFFICIENT, not CONTRADICT.
> 2. Direction: compare the direction of the effect in the claim and in the paper (increase or decrease, more
> or less, higher or lower, protective or harmful). The same direction is SUPPORT; the opposite direction is
> CONTRADICT. Shared terms or topic alone are not support.
> 3. Negation: a claim of no effect, no association, or ineffectiveness is contradicted by a reported effect,
> and supported by a reported absence of effect.
> 4. Findings: use the paper's own results and conclusions, not background statements or prior work that the
> paper reports and then overturns.

The output schema, its validation, the Judge, retrieval and generation settings are unchanged. A first
version also asked for the rationale before the label; Ollama's structured output did not keep schema
property order (smoke job 355674), so that request was removed in commit `122f206` before these runs.

## Reproducibility configuration

| Item | Recorded value |
| --- | --- |
| Code | Branch `feat/verifier-v2`, commit `122f206` (`122f20699545645c0e03496c46bd291eb735f151`) for all four runs, recorded in each job's `environment.txt`. Implementation version `argus-v4.1-citation-normalization-verifier-v2` for A and B, `argus-v4.1-citation-normalization-challenge-retrieval-verifier-v2` for D and F. |
| v1 comparison runs | Commit `8d331ba`; see `challenge-retrieval-v41-summary.md` |
| Dataset | Original AllenAI SciFact: 5,183 abstracts, 809 train claims |
| Evaluation set | 162 internal validation claims from train, seed 42 (66 SUPPORT, 35 CONTRADICT, 61 no annotated evidence); ordered IDs identical to the v1 runs and both baselines |
| Evidence Verifier | v1: `argus-v3`, SHA-256 `811de57019988a77a1e7697b5171a6fb550e5791c9943f14ed40d5a5b17b54ee`. v2: `argus-verifier-v2`, SHA-256 `0a5985498c5405b14a24d67985400acb8246b8715000a78e67ede6bf0dca925e` |
| Judge | Unchanged: `argus-v4-judge` (v4.1), SHA-256 `9a1ba5249f38efd8de37906c1954e9cc953814f9cab4fd4caa1c652a0b3b8bdc` |
| Retrieval | Unchanged from the v1 runs, per arm; retriever `BAAI/bge-small-en-v1.5`, SHA `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Generator | `qwen3:4b-instruct-2507-q4_K_M`, same Ollama digest as the v1 runs; Ollama 0.40.5 |
| Generation | temperature 0, top-k 40, `num_predict=512`, `num_ctx=8192`, non-streaming |
| Budgets | `retrieval_k=5` per agent, max 10 merged documents, max 12 LLM calls per claim |
| Hardware | SMU `crimson`, partition `student`, node `avenue`; 1× NVIDIA A40, 4 CPU cores, 24 GB RAM per job |

For each arm a script checked that the v1 and v2 metadata agree on the claim IDs, agent settings apart from
the Verifier version, retrieval settings, generator settings, model digest and Judge prompt. All checks
passed.

## Primary results

Failed claims remain in denominators as FAILURE. Statistics compare v2 with v1 within each arm: exact
McNemar test, Holm correction over the four arms, and a paired bootstrap of the accuracy difference (10,000
resamples, seed 42).

| Metric | A v1 | A v2 | B v1 | B v2 | D v1 | D v2 | F v1 | F v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct verdicts | 121 | **126** | 121 | **127** | 119 | **124** | 118 | **128** |
| Failure-aware accuracy | 74.69% | 77.78% | 74.69% | 78.40% | 73.46% | 76.54% | 72.84% | 79.01% |
| Macro-F1 | 0.7355 | 0.7757 | 0.7343 | 0.7810 | 0.7231 | 0.7624 | 0.7156 | 0.7862 |
| Failures | 0 | 1 | 0 | 0 | 1 | 2 | 0 | 1 |

| Paired comparison, v2 against v1 | A | B | D | F |
| --- | ---: | ---: | ---: | ---: |
| Claims only v2 got right | 13 | 15 | 13 | 15 |
| Claims only v1 got right | 8 | 9 | 8 | 5 |
| Exact McNemar p | 0.383 | 0.308 | 0.383 | 0.041 |
| Holm-adjusted p | 0.922 | 0.922 | 0.922 | 0.166 |
| Accuracy difference (points) | +3.09 | +3.70 | +3.09 | +6.17 |
| 95% bootstrap CI (points) | −2.47 to +8.64 | −1.85 to +9.88 | −2.47 to +8.64 | +0.62 to +11.73 |

**Claims changed in all four arms.** Fixed by v2 in every arm: 37, 46, 62, 153, 245, 638, 764, 801, 841,
1111, 1291. Broken by v2 in every arm: 68, 76, 79, 1128, 1224.

**Failures.** Every v2 failure is at the Judge, not the Verifier: claim 825 (A, D, F) cites an out-of-range
sentence in document 2319305; claim 1260 (D) exceeds the v4.1 citation limits. In B, claim 825 was answered
correctly.

### Per-class F1

| Gold class (number) | A v1 | A v2 | B v1 | B v2 | D v1 | D v2 | F v1 | F v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SUPPORT (66) | 0.815 | 0.813 | 0.824 | 0.816 | 0.806 | 0.819 | 0.800 | 0.835 |
| CONTRADICT (35) | 0.675 | 0.738 | 0.675 | 0.762 | 0.651 | 0.707 | 0.667 | 0.738 |
| INSUFFICIENT (61) | 0.717 | 0.776 | 0.705 | 0.765 | 0.713 | 0.761 | 0.680 | 0.786 |

CONTRADICT and INSUFFICIENT F1 rose in every arm; SUPPORT F1 moved by less than 0.04 either way.

### Confusion matrices (v2)

Rows are gold; each cell is predicted SUPPORT / CONTRADICT / INSUFFICIENT / FAILURE. The v1 matrices are in
`challenge-retrieval-v41-summary.md`.

| Gold | A | B | D | F |
| --- | --- | --- | --- | --- |
| SUPPORT | 50 / 7 / 9 / 0 | 51 / 6 / 9 / 0 | 52 / 7 / 7 / 0 | 53 / 7 / 6 / 0 |
| CONTRADICT | 2 / 31 / 1 / 1 | 2 / 32 / 1 / 0 | 2 / 29 / 2 / 2 | 2 / 31 / 1 / 1 |
| INSUFFICIENT | 5 / 11 / 45 / 0 | 6 / 11 / 44 / 0 | 7 / 11 / 43 / 0 | 6 / 11 / 44 / 0 |

## How the gain arises

### Error types

| Error (gold → predicted) | A v1 | A v2 | B v1 | B v2 | D v1 | D v2 | F v1 | F v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INSUFFICIENT → SUPPORT | 10 | 5 | 10 | 6 | 12 | 7 | 13 | 6 |
| INSUFFICIENT → CONTRADICT | 13 | 11 | 14 | 11 | 13 | 11 | 14 | 11 |
| CONTRADICT → SUPPORT | 4 | 2 | 4 | 2 | 5 | 2 | 5 | 2 |
| SUPPORT → CONTRADICT | 7 | 7 | 6 | 6 | 8 | 7 | 7 | 7 |
| SUPPORT → INSUFFICIENT | 4 | 9 | 4 | 9 | 2 | 7 | 3 | 6 |
| CONTRADICT → INSUFFICIENT | 3 | 1 | 3 | 1 | 2 | 2 | 2 | 1 |

v2 removes false SUPPORT verdicts on INSUFFICIENT and CONTRADICT claims, and a few false CONTRADICT verdicts on
INSUFFICIENT claims, at the price of missing more true SUPPORT claims. SUPPORT → CONTRADICT errors, the
verdict-level form of a direction flip, are unchanged.

### Verifier labels

Gold papers are counted when the Verifier saw them and their gold annotations carry one label.

| Measure | A v1 | A v2 | B v1 | B v2 | D v1 | D v2 | F v1 | F v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gold papers labelled as gold | 90 | 86 | 92 | 89 | 91 | 88 | 92 | 89 |
| Gold papers with direction flipped | 11 | 12 | 11 | 12 | 12 | 12 | 12 | 12 |
| Gold papers labelled INSUFFICIENT | 9 | 12 | 9 | 11 | 10 | 13 | 10 | 13 |

| Label changes on claim–paper pairs verified by both | A | B | D | F |
| --- | ---: | ---: | ---: | ---: |
| Pairs changed / shared | 58 / 809 | 57 / 923 | 83 / 1,319 | 94 / 1,615 |
| SUPPORT → INSUFFICIENT | 25 | 25 | 36 | 40 |
| CONTRADICT → INSUFFICIENT | 14 | 13 | 26 | 33 |
| SUPPORT → CONTRADICT | 9 | 9 | 10 | 10 |
| INSUFFICIENT → CONTRADICT | 7 | 7 | 8 | 8 |
| CONTRADICT → SUPPORT | 2 | 2 | 2 | 2 |
| INSUFFICIENT → SUPPORT | 1 | 1 | 1 | 1 |

Most changes move a label to INSUFFICIENT (39 of 58 in A, 73 of 94 in F); one pair per arm moves from
INSUFFICIENT to SUPPORT. The direction check did not repair the direction errors the audit found. A likely
reading is that the 4B model follows the prompt's caution about subject and topic more reliably than its
direction comparison; this was not tested directly.

### Audit claims

Verdicts v1 → v2. Gold label in brackets.

| Claim | A | B | D | F |
| --- | --- | --- | --- | --- |
| 494 [INS] | INS → INS | CON → **INS** | CON → **INS** | CON → CON |
| 963 [CON] | INS → INS | INS → INS | SUP → INS | SUP → INS |
| 951 [SUP] | INS → INS | INS → INS | SUP → SUP | SUP → SUP |
| 866 [INS] | INS → *SUP* | INS → *SUP* | SUP → SUP | SUP → SUP |
| 856 [CON] | CON → CON | CON → CON | FAIL → **CON** | CON → CON |
| 153 [INS] | CON → **INS** | CON → **INS** | CON → **INS** | CON → **INS** |
| 582 [INS] | SUP → CON | SUP → CON | SUP → CON | CON → CON |
| 1026, 602, 688, 876 [INS] | CON → CON | CON → CON | CON → CON | CON → CON |

Bold = fixed by v2; italic = broken by v2. 153 (different subject treated as contradiction) is fixed in all
four arms. 866 is newly wrong in A and B, and the background-as-finding errors in 1026, 602, 688 and 876 are
untouched. In 688, and arguably 602 and 582, the audit judged the CONTRADICT verdict scientifically defensible.

## Comparison with the baselines

Exact McNemar, uncorrected.

| System | Correct | Accuracy | Macro-F1 | vs Standard RAG (ARGUS-only / baseline-only, p) | vs Iterative RAG |
| --- | ---: | ---: | ---: | --- | --- |
| Iterative RAG | 118 | 72.84% | 0.738 | n/a | n/a |
| Standard RAG | 119 | 73.46% | 0.737 | n/a | n/a |
| ARGUS A, v2 | 126 | 77.78% | 0.776 | 17 / 10, p = 0.248 | 18 / 10, p = 0.185 |
| ARGUS B, v2 | 127 | 78.40% | 0.781 | 17 / 9, p = 0.169 | 17 / 8, p = 0.108 |
| ARGUS D, v2 | 124 | 76.54% | 0.762 | 16 / 11, p = 0.442 | 17 / 11, p = 0.345 |
| ARGUS F, v2 | 128 | 79.01% | 0.786 | 19 / 10, p = 0.136 | 19 / 9, p = 0.087 |

With Verifier v1, ARGUS A and B were 2–3 claims ahead of the baselines (p ≥ 0.69). With v2 the lead is 5–10
claims, still not significant. The baselines have not been rerun since 22 Oct and received no comparable prompt
revision.

## Cost

Per claim, averaged over 162 claims.

| Arm | LLM calls | Tokens v1 → v2 | End-to-end latency v1 → v2 |
| --- | ---: | --- | --- |
| A | 6.00 | 5,928 → 7,047 (+18.9%) | 23.1 s → 27.8 s |
| B | 6.71 | 6,785 → 8,064 (+18.9%) | 25.5 s → 30.8 s |
| D | 9.16 | 9,352 → 11,169 (+19.4%) | 32.6 s → 39.5 s |
| F | 11.00 | 11,543 → 13,770 (+19.3%) | 39.1 s → 47.3 s |

Standard RAG makes one LLM call per claim. Latency in D and F also reflects the two jobs sharing a node.

## Conclusions

1. Verifier v2 raised accuracy and macro-F1 in all four arms tested, by 5 to 10 claims. The direction of the
   effect is consistent, but no arm's gain survives Holm correction, and the arms share most of their errors.
2. The gain comes from more cautious labelling: fewer SUPPORT labels and fewer false SUPPORT and CONTRADICT
   verdicts. Direction flips on gold papers did not decrease, so the main failure mode found by the audit
   remains.
3. With v2, every ARGUS arm tested scores above Standard RAG and Iterative RAG on this set, with no
   significant difference, at roughly 6 to 11 times the LLM calls of Standard RAG.
4. The Challenge agent still does not measurably change accuracy: the v2 arms span 4 claims.
5. These 162 claims were used both for the audit that motivated v2 and for this evaluation, so v2's score is
   optimistic. Its effect has to be confirmed once, on the official development split, after the Verifier
   version is frozen.

## Decisions for the team (not acted on)

- Whether to freeze Verifier v1 or v2 for the final development-split run.
- Which systems go in that run. A and B were the planned ARGUS arms; choosing F because it scored highest here
  would be selection on these claims and should be labelled as such.
- Whether to report a larger or LoRA-tuned Verifier as future work, given that the 4B model did not apply the
  direction check.
