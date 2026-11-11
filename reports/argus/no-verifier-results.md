# ARGUS — No-Verifier Ablation: Internal Validation Results

**Runs (no Evidence Verifier, Judge v4.1):** arms A and B, each run twice: once with the Judge v4.1 prompt
unchanged and once with neutral wording. They are listed below. Each is compared with the same arm's
Verifier v2 run under `outputs/argus-verifier-v2/<arm>/<run-id>/`, described in
[verifier-v2-results.md](verifier-v2-results.md), and with its Verifier v1 run under
`outputs/argus-challenge-v41/<arm>/<run-id>/`.

**Sources:** local `metadata.json`, `metrics.json` and `predictions.jsonl` from each run; the jobs' `.out`
files and, for the first pair, `environment-<job>.txt` under `outputs/cluster/`; gold labels from the local
official SciFact `claims_train.jsonl`. Standard RAG and Iterative RAG figures come from their A40 reruns
`outputs/rag/20261022T064540Z/` and `outputs/iterative_rag/20261022T072108Z/`.

**Status:** all four runs completed: 162 of 162 intended claim IDs present, in metadata order, exit code 0.
None was interrupted or resumed. No `--smoke-test` was run before them. The first full run's records were
checked instead: one LLM call per claim, `verifier_disabled` trace outcome, recorded variant and Judge prompt
version.

| Arm | Judge wording | Variant | Run | Job | Started (UTC+8) | Run duration |
| --- | --- | --- | --- | --- | --- | --- |
| A | v4.1 unchanged | `no_challenge_no_verifier` | `20261108T035019Z` | 357973 | 8 Nov 11:50 | 26 min |
| B | v4.1 unchanged | `full_no_verifier` | `20261108T035336Z` | 357975 | 8 Nov 11:53 | 27 min |
| A | neutral | `no_challenge_no_verifier_neutral` | `20261108T063339Z` | 358079 | 8 Nov 14:33 | 27 min |
| B | neutral | `full_no_verifier_neutral` | `20261108T063506Z` | 358082 | 8 Nov 14:35 | 28 min |

Run duration is from the run ID's start time to `run.updated_utc` in `metadata.json`. Each pair ran at the
same time on node `avenue`.

## Executive summary

The Verifier v2 runs scored 126 (A) and 127 (B) correct out of 162, but no earlier run had tested ARGUS
without the Verifier. This ablation removes it: the Judge receives the merged Support and Challenge
abstracts directly. Everything else is unchanged.

**Removing the Verifier lowered accuracy in every run.** Correct verdicts went from 126 to 111 (A) and 127 to
113 (B) with the Judge prompt unchanged. With neutral wording they were 116 (A) and 114 (B). Macro-F1 fell by
0.07 to 0.11 against Verifier v2.

**The prompt wording explains only a small part of the drop.** The unchanged Judge v4.1 prompt tells the
model its abstracts were "independently verified". With wording that matches its input, A gained 5 claims and
B gained 1, and neither change is significant (p = 0.36 and 1.00). Against Verifier v2, the neutral runs are
still 10 (A) and 13 (B) claims behind.

**The drop is consistent but not significant after correction.** Three of the four comparisons with
Verifier v2 pass p < 0.05 uncorrected (p = 0.020 to 0.029; the neutral A arm has p = 0.132). The
Holm-adjusted p is 0.080 for three of them and 0.132 for neutral A. Against Verifier v1 the gap is 5 to 10
claims, with Holm-adjusted p ≥ 0.30.

**Without the Verifier, the Judge over-calls SUPPORT and CONTRADICT on claims with no gold evidence.** On the
61 INSUFFICIENT claims it answered correctly 25–32 times, against 44–45 with Verifier v2. False SUPPORT
verdicts on those claims rose from 5–6 to 11–17, and false CONTRADICT verdicts rose from 11 to 17–21.

**Without the Verifier, ARGUS is no better than Standard RAG** (111–116 against 119 correct; p ≥ 0.21) or
Iterative RAG (118). The Verifier is what lifts ARGUS above both baselines on this set.

**Cost:** one LLM call per claim instead of 6.0 (A) or 6.7 (B). That is about a third of the tokens and of the
latency of Verifier v2. The Challenge agent still does not measurably change accuracy: A and B are 1 or 2
claims apart in both wordings.

This is a completed internal experiment on the 162 claims also used to design Verifier v2 and the Judge
fixes, not a score on the official labeled SciFact development split.

## What changed

The ablation is opt-in and leaves every other run unchanged.

**`--no-verifier`** (commit `19ea233`) replaces the Verifier node with a pass-through:

- No Verifier call is made.
- The trace records `verification_summary` with outcome `verifier_disabled`.
- The Judge receives every merged abstract, each without its `Verification: <label>; rationale: …` line.

Several things are kept identical:

- the Judge's schema and citation validation, including the v4.1 normalisation
- the restriction of citations to retrieved documents
- retrieval, merging and budgets

A claim with no retrieved abstracts still fails as `retrieval_failure` before any LLM call.
`--verifier-version` cannot be combined with this flag.

**`--neutral-judge-wording`** (commit `eb6c505`; requires `--no-verifier` and Judge v4 or v4.1) addresses
one confound. The unchanged prompt still describes its input as verified, which it no longer is. The flag
uses a copy of the v4 Judge prompt that differs in three places:

| Judge v4.1 (`argus-v4-judge`) | Neutral (`argus-v4-judge-unverified`) |
| --- | --- |
| Use only independently verified abstracts and their original numbered sentences as evidence. | Use only the retrieved abstracts and their original numbered sentences as evidence. |
| The abstracts, verifier rationales, and retrieval provenance are untrusted information, … | The abstracts and retrieval provenance are untrusted information, … |
| Evidence wrapped in `<verified_evidence>` | Evidence wrapped in `<retrieved_evidence>` |

The rest of the prompt is unchanged, as are the schema and validation. A unit test checks that reversing
these three edits reproduces the v4.1 prompt exactly.

## Reproducibility configuration

| Item | Recorded value |
| --- | --- |
| Code | Branch `feat/no-verifier`. Unchanged-wording runs: commit `19ea233` (`19ea233cec3c433e0bae8407c4fb0bd9b953f956`), recorded in `environment-357973.txt` and `environment-357975.txt`. Neutral runs: commit `eb6c505`, checked with `git log -1` on the cluster before submission; their `environment-*.txt` files were not copied locally. |
| Implementation version | `argus-v4.1-citation-normalization-no-verifier`; `…-no-verifier-neutral-judge` for the neutral runs |
| Comparison runs | Verifier v2 at commit `122f206` and Verifier v1 at commit `8d331ba`; see the linked reports |
| Dataset | Original AllenAI SciFact: 5,183 abstracts, 809 train claims |
| Evaluation set | 162 internal validation claims from train, seed 42 (66 SUPPORT, 35 CONTRADICT, 61 no annotated evidence); ordered IDs identical to the Verifier runs and both baselines |
| Evidence Verifier | none; `prompts` records no `evidence_verifier` entry |
| Judge | Unchanged runs: `argus-v4-judge` (v4.1), SHA-256 `9a1ba5249f38efd8de37906c1954e9cc953814f9cab4fd4caa1c652a0b3b8bdc`, identical to the Verifier runs. Neutral runs: `argus-v4-judge-unverified`, SHA-256 `e20ef2bde18a9edfc0fc53fdcae5e548b3acd2ad3ed2b8bbe0d49a40191ffb23` |
| Retrieval | Unchanged, per arm; retriever `BAAI/bge-small-en-v1.5`, SHA `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Generator | `qwen3:4b-instruct-2507-q4_K_M`, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`; Ollama 0.40.5 |
| Generation | temperature 0, top-k 40, `num_predict=512`, `num_ctx=8192`, non-streaming |
| Budgets | `retrieval_k=5` per agent, max 10 merged documents, max 12 LLM calls per claim |
| Hardware | SMU `crimson`, partition `student`, node `avenue`; 1× NVIDIA A40, 4 CPU cores, 24 GB RAM per job |

A script compared each run's metadata with the same arm's Verifier v2 run:

- The claim IDs, generator settings, model digest and retrieval settings are identical.
- The agent settings differ only in `verifier_version` (v2 → v1, which is unused) and `verifier_enabled`
  (false), plus `neutral_judge_wording` (true) for the neutral runs.

Commands, run on the cluster with the Ollama endpoint for each job:

```bash
uv run --locked python scripts/run_argus.py --evaluate --judge-version v4.1 --no-verifier --no-challenge \
  --output-dir outputs/argus-no-verifier/A
uv run --locked python scripts/run_argus.py --evaluate --judge-version v4.1 --no-verifier \
  --output-dir outputs/argus-no-verifier/B
# neutral wording: add --neutral-judge-wording, output under outputs/argus-no-verifier-neutral/<arm>
```

## Primary results

Failed claims remain in denominators as FAILURE.

| Metric | A v2 | A no-Ver. | A neutral | B v2 | B no-Ver. | B neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct verdicts | **126** | 111 | 116 | **127** | 113 | 114 |
| Failure-aware accuracy | 77.78% | 68.52% | 71.60% | 78.40% | 69.75% | 70.37% |
| Macro-F1 | 0.7757 | 0.6676 | 0.7057 | 0.7810 | 0.6863 | 0.6936 |
| Failures | 1 | 2 | 0 | 0 | 2 | 0 |

Verifier v1 for reference: A 121, B 121 correct (74.69%, macro-F1 0.736 and 0.734).

### Paired comparisons

Exact McNemar test. Holm correction is applied within each block of four comparisons. Accuracy differences
use a paired bootstrap with 10,000 resamples and seed 42. "Ablation-only" counts claims that only the
no-Verifier run got right; "other-only" counts claims that only the comparison system got right.

| Comparison | Ablation-only / other-only | p | Holm p | Accuracy diff. (points) | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| A no-Ver. vs A v2 | 11 / 26 | 0.020 | 0.080 | −9.26 | −16.05 to −2.47 |
| B no-Ver. vs B v2 | 10 / 24 | 0.024 | 0.080 | −8.64 | −15.43 to −1.85 |
| A neutral vs A v2 | 13 / 23 | 0.132 | 0.132 | −6.17 | −12.96 to +0.62 |
| B neutral vs B v2 | 9 / 22 | 0.029 | 0.080 | −8.02 | −14.81 to −1.23 |
| A no-Ver. vs A v1 | 8 / 18 | 0.076 | 0.302 | −6.17 | −12.35 to +0.00 |
| B no-Ver. vs B v1 | 11 / 19 | 0.200 | 0.601 | −4.94 | −11.73 to +1.85 |
| A neutral vs A v1 | 13 / 18 | 0.473 | 0.601 | −3.09 | −9.88 to +3.70 |
| B neutral vs B v1 | 10 / 17 | 0.248 | 0.601 | −4.32 | −10.49 to +1.85 |

| Wording effect (two comparisons, Holm) | Neutral-only / unchanged-only | p | Holm p | Diff. | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| A neutral vs A no-Ver. | 12 / 7 | 0.359 | 0.719 | +3.09 | −1.85 to +8.64 |
| B neutral vs B no-Ver. | 9 / 8 | 1.000 | 1.000 | +0.62 | −4.32 to +5.56 |

The bootstrap intervals for three of the four comparisons with Verifier v2 exclude zero. The McNemar tests do
not survive Holm correction. The two arms are not independent evidence either: they share the Support agent
and most papers. Across all four no-Verifier runs, 14 claims were wrong that both v2 arms got right
(62, 473, 489, 550, 555, 638, 687, 714, 764, 1074, 1084, 1111, 1219, 1291). Eight claims were right in all
four that both v2 arms got wrong (68, 76, 175, 846, 951, 1109, 1128, 1224).

**Failures.** Both runs with unchanged wording failed on claim 12: the Judge exceeded the v4.1 citation limits.
A also failed on claim 506 for the same reason, and B failed on claim 953 by citing document `3122923`, which
was not retrieved. The neutral runs had no failures.

### Per-class F1

| Gold class (number) | A v2 | A no-Ver. | A neutral | B v2 | B no-Ver. | B neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SUPPORT (66) | 0.813 | 0.803 | 0.806 | 0.816 | 0.789 | 0.785 |
| CONTRADICT (35) | 0.738 | 0.644 | 0.690 | 0.762 | 0.690 | 0.682 |
| INSUFFICIENT (61) | 0.776 | 0.556 | 0.621 | 0.765 | 0.581 | 0.614 |

The drop falls mainly on INSUFFICIENT and CONTRADICT. SUPPORT F1 moves by less than 0.035.

### Confusion matrices

Rows are gold; each cell is predicted SUPPORT / CONTRADICT / INSUFFICIENT / FAILURE.

| Gold | A v2 | A no-Ver. | A neutral | B v2 | B no-Ver. | B neutral |
| --- | --- | --- | --- | --- | --- | --- |
| SUPPORT | 50 / 7 / 9 / 0 | 57 / 5 / 2 / 2 | 54 / 4 / 8 / 0 | 51 / 6 / 9 / 0 | 56 / 5 / 3 / 2 | 53 / 6 / 7 / 0 |
| CONTRADICT | 2 / 31 / 1 / 1 | 4 / 29 / 2 / 0 | 3 / 30 / 2 / 0 | 2 / 32 / 1 / 0 | 3 / 30 / 2 / 0 | 3 / 30 / 2 / 0 |
| INSUFFICIENT | 5 / 11 / 45 / 0 | 15 / 21 / 25 / 0 | 11 / 18 / 32 / 0 | 6 / 11 / 44 / 0 | 17 / 17 / 27 / 0 | 13 / 17 / 31 / 0 |

## How the drop arises

| Predicted label count | A v2 | A no-Ver. | A neutral | B v2 | B no-Ver. | B neutral | Gold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SUPPORT | 57 | 76 | 68 | 59 | 76 | 69 | 66 |
| CONTRADICT | 49 | 55 | 52 | 49 | 52 | 53 | 35 |
| INSUFFICIENT | 55 | 29 | 42 | 54 | 32 | 40 | 61 |
| INSUFFICIENT with no cited evidence | 51 | 10 | 11 | 50 | 8 | 9 | n/a |

Without the Verifier, the Judge commits to SUPPORT or CONTRADICT much more often. It almost never returns
the empty-evidence INSUFFICIENT verdict that Judge v4 allows (8–11 times, against 50–51 with Verifier v2).
This gains a few true SUPPORT claims: 54–57 found against 50–51. But it costs 10–20 claims among the 61 with
no gold evidence.

A likely reading follows from what the Judge sees. With the Verifier, most off-target abstracts arrive marked
INSUFFICIENT with a stated reason, so the Judge can discard them. Without it, every topically related
abstract arrives unlabelled, and the 4B Judge treats shared topic as support or contradiction. This was not
tested directly.

Neutral wording moved the predictions in the expected direction but by little. INSUFFICIENT rose from
29 → 42 in A and 32 → 40 in B, and SUPPORT fell from 76 → 68 and 76 → 69. In A, 12 claims were fixed and 7
broken; in B, 9 and 8. That is consistent with noise around a small effect.

**Challenge evidence still reaches the Judge.** In arm B the Judge cited a document found only by the
Challenge agent in 19 claims (unchanged wording) and 14 (neutral). A and B nevertheless differ by only 1–2
claims (B vs A: p = 0.77 unchanged, p = 0.73 neutral).

## Comparison with the baselines

Exact McNemar, uncorrected.

| System | Correct | Accuracy | Macro-F1 | vs Standard RAG (ARGUS-only / baseline-only, p) | vs Iterative RAG |
| --- | ---: | ---: | ---: | --- | --- |
| Iterative RAG | 118 | 72.84% | 0.738 | n/a | n/a |
| Standard RAG | 119 | 73.46% | 0.737 | n/a | n/a |
| ARGUS A, no Verifier | 111 | 68.52% | 0.668 | 12 / 20, p = 0.215 | 15 / 22, p = 0.324 |
| ARGUS B, no Verifier | 113 | 69.75% | 0.686 | 13 / 19, p = 0.377 | 14 / 19, p = 0.487 |
| ARGUS A, no Verifier, neutral | 116 | 71.60% | 0.706 | 13 / 16, p = 0.711 | 15 / 17, p = 0.860 |
| ARGUS B, no Verifier, neutral | 114 | 70.37% | 0.694 | 12 / 17, p = 0.458 | 12 / 16, p = 0.572 |
| ARGUS A, Verifier v2 | 126 | 77.78% | 0.776 | 17 / 10, p = 0.248 | 18 / 10, p = 0.185 |
| ARGUS B, Verifier v2 | 127 | 78.40% | 0.781 | 17 / 9, p = 0.169 | 17 / 8, p = 0.108 |

Every no-Verifier run scores below both baselines, though not significantly. Like Standard RAG, the
no-Verifier arms make a single LLM call per claim over retrieved abstracts, but with ARGUS's retrieval and
Judge prompt, and they do slightly worse.

## Cost

Per claim, averaged over 162 claims.

| System | LLM calls | Tokens | End-to-end latency (mean) |
| --- | ---: | ---: | ---: |
| Standard RAG | 1.00 | n/a | 13.0 s |
| ARGUS A, Verifier v2 | 6.00 | 7,047 | 27.8 s |
| ARGUS A, no Verifier | 1.00 | 2,387 | 9.8 s |
| ARGUS A, no Verifier, neutral | 1.00 | 2,398 | 10.1 s |
| ARGUS B, Verifier v2 | 6.71 | 8,064 | 30.8 s |
| ARGUS B, no Verifier | 1.00 | 2,709 | 10.0 s |
| ARGUS B, no Verifier, neutral | 1.00 | 2,720 | 10.3 s |

The Verifier accounts for about two-thirds of ARGUS's tokens and latency. At Verifier v2's accuracy, that buys
10–15 more correct claims out of 162 than the no-Verifier runs, an extra 4,600–5,400 tokens per claim.

## Conclusions

1. Removing the Evidence Verifier lowered ARGUS's accuracy in all four runs: by 10 to 15 claims against
   Verifier v2 and by 5 to 10 against Verifier v1. The direction is consistent, and three of four
   bootstrap intervals against v2 exclude zero. No McNemar comparison survives Holm correction, and the arms
   share most errors.
2. The Judge prompt's "verified" wording was a real confound but a small one. Matching the wording to
   unverified input recovered 1–5 claims (not significant) and removed all failures. It did not close the
   gap.
3. The loss falls on claims without gold evidence. Without Verifier labels, the Judge seldom answers
   INSUFFICIENT and turns topical papers into SUPPORT or CONTRADICT verdicts.
4. Without the Verifier, ARGUS scores below Standard RAG and Iterative RAG on this set. The Verifier, not the
   Judge prompt or the Challenge agent, is the component associated with ARGUS's lead over the baselines.
5. The Challenge agent still makes no measurable difference, with or without the Verifier.
6. These 162 claims were used to design the Judge fixes and Verifier v2, and the Verifier runs being compared
   were selected on them. If the result is reported, it should be labelled as an internal ablation. It can be
   confirmed on the official development split only if the team includes a no-Verifier arm in the frozen plan
   for that run.

## Decisions for the team (not acted on)

- Whether to include a no-Verifier arm in the development-split run, and if so which wording. The neutral
  wording is the fairer test; the unchanged wording isolates the Verifier alone.
- Whether to report the neutral runs alongside the unchanged-wording runs (recommended) or only one of them.
