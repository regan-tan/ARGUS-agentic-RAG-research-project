# ARGUS — Final Evaluation on the Official SciFact Dev Split

**Runs:** five systems on all 300 claims of the official labeled SciFact `claims_dev.jsonl`, run with
`--evaluate --split dev`. The split had not been used for any design, tuning or prompt decision before these
runs. Each system's configuration is the one frozen after the internal-validation experiments.

**Sources:** local `metadata.json`, `metrics.json` and `predictions.jsonl` under `outputs/dev/`; the jobs'
`.out` files under `outputs/cluster/`; gold labels from the local official SciFact `claims_dev.jsonl`.
Internal-validation figures come from [no-verifier-results.md](no-verifier-results.md) and
[verifier-v2-results.md](verifier-v2-results.md).

**Status:** all five runs completed: 300 of 300 claims, evaluation mode `official_dev`, run status
`completed`, SLURM state `COMPLETED (exit code 0)`. None was interrupted or resumed, and none hit the 4-hour
time limit. All five used the same generator, `qwen3:4b-instruct-2507-q4_K_M` (Ollama digest
`0edcdef3…`, runtime 0.40.5), and the same pinned BGE retriever. The code was the `feat/dev-split` change
merged to `main`. `metadata.json` does not record a commit hash.

| System | Configuration | Run | Job | Started (UTC+8) | Run duration |
| --- | --- | --- | --- | --- | --- |
| Standard RAG | `standard-rag-v1` | `rag/20261108T101334Z` | 358108 | 8 Nov 18:13 | 1 h 03 min |
| Iterative RAG | `iterative-rag-v3` | `iterative_rag/20261108T073505Z` | 358106 | 8 Nov 15:35 | 2 h 38 min |
| ARGUS A | no Challenge, Verifier v2, Judge v4.1 | `argus/A/20261108T095704Z` | 358107 | 8 Nov 17:57 | 2 h 09 min |
| ARGUS B | full: original Challenge, Verifier v2, Judge v4.1 | `argus/B/20261108T073436Z` | 358105 | 8 Nov 15:34 | 2 h 22 min |
| ARGUS B, no Verifier | full, no Verifier, neutral Judge wording | `argus/B-no-verifier/20261108T111653Z` | 358366 | 8 Nov 19:16 | 51 min |

Run directories are under `outputs/dev/`. Run duration is from the run ID's start time to `run.updated_utc`
in `metadata.json`; it matches the SLURM wall-clock time to within a minute.

The dev split has 300 claims: 124 SUPPORT, 64 CONTRADICT and 112 INSUFFICIENT (no annotated evidence). None
has mixed labels, so all 300 are scored.

## Executive summary

**ARGUS does not beat Standard RAG on held-out claims.** Full ARGUS (arm B) got 223 of 300 correct; Standard
RAG got 220. The difference is +1.0 percentage point (95% CI −4.0 to +6.0), with exact McNemar p = 0.79.
Macro-F1 is the same: 0.732 against 0.733.

**None of the four comparisons fixed before the run is significant.** Against Standard RAG, Iterative RAG,
ARGUS A and ARGUS without the Verifier, Holm-adjusted p is 0.88 or higher.

**The Challenge agent has no effect.** ARGUS A (no Challenge) and B both got 223 correct. They disagree on
only 6 claims, 3 in each direction.

**The Verifier points the right way but is not confirmed.** Removing it costs 10 claims (223 to 213; McNemar
p = 0.22, Holm 0.88). That is the same direction as on the internal set, where it cost 13 claims (127 to 114,
neutral wording). The Verifier also makes citations much more precise: 57% of cited documents are gold,
against 36% without it.

**ARGUS changes which mistakes are made, not how many.** Compared with Standard RAG it is better at
INSUFFICIENT claims (87 against 77 correct) but more often calls a supported claim CONTRADICT (29 against 11).
The two roughly cancel.

**ARGUS costs much more.** Full ARGUS makes 6.8 LLM calls per claim against 1, uses 3.3× the tokens and takes
2.3× the time per claim.

**The internal advantage did not replicate.** On the 162 internal claims B led Standard RAG by 8 claims
(78.4% against 73.5%). On dev the lead is 1 percentage point. Standard RAG's accuracy barely moved (73.5% to
73.3%), while ARGUS fell from 78.4% to 74.3%. Verifier v2 and Judge v4.1 were designed on those 162 claims, so
part of the internal lead was probably fitted to that set.

**Answer to the research question:** on SciFact with Qwen3-4B, explicit counter-evidence retrieval does not
improve claim verification. The multi-agent pipeline is not more accurate than one-shot Standard RAG and costs
several times more. This is a negative result from a test fixed in advance.

## Main results

| System | Correct /300 | Accuracy | Macro-F1 | F1 SUPPORT | F1 CONTRADICT | F1 INSUFFICIENT | Failed outputs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard RAG | 220 | 0.733 | 0.733 | 0.780 | 0.703 | 0.716 | 4 |
| Iterative RAG | 215 | 0.717 | 0.722 | 0.764 | 0.707 | 0.696 | 6 |
| ARGUS A | 223 | 0.743 | 0.733 | 0.776 | 0.624 | 0.798 | 1 |
| ARGUS B | **223** | **0.743** | 0.732 | 0.778 | 0.617 | **0.802** | 0 |
| ARGUS B, no Verifier | 213 | 0.710 | 0.703 | 0.765 | 0.650 | 0.694 | 1 |

Failed outputs are invalid or unparseable responses. They stay in the denominator and count as wrong. Accuracy
and macro-F1 were recomputed from `predictions.jsonl` and match `metrics.json` exactly.

## Comparisons fixed before the run

These four comparisons were chosen before the dev runs started. Each pairs ARGUS B with one other system on the
same 300 claims. p is from the exact two-sided McNemar test on per-claim correctness, and Holm corrects across
the four. The confidence intervals are 95% paired bootstrap intervals (10,000 resamples of claims, seed 42) for
B minus the other system.

| ARGUS B vs | Only B right | Only other right | McNemar p | Holm p | Accuracy difference (95% CI) | Macro-F1 difference (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| Standard RAG | 29 | 26 | 0.788 | 1.000 | +1.0 pp (−4.0, +6.0) | −0.001 (−0.048, +0.047) |
| Iterative RAG | 37 | 29 | 0.389 | 1.000 | +2.7 pp (−2.7, +8.0) | +0.010 (−0.040, +0.062) |
| ARGUS A (no Challenge) | 3 | 3 | 1.000 | 1.000 | 0.0 pp (−1.7, +1.7) | 0.000 (−0.016, +0.015) |
| ARGUS B, no Verifier | 32 | 22 | 0.220 | 0.881 | +3.3 pp (−1.3, +8.0) | +0.029 (−0.017, +0.077) |

Every interval includes zero. Against Standard RAG the interval rules out an ARGUS advantage larger than about
6 percentage points.

**Exploratory comparisons** (not in the fixed set, uncorrected):

| Comparison | Only first right | Only second right | McNemar p |
| --- | --- | --- | --- |
| ARGUS A vs Standard RAG | 30 | 27 | 0.791 |
| ARGUS B, no Verifier, vs Standard RAG | 28 | 35 | 0.450 |
| Iterative RAG vs Standard RAG | 16 | 21 | 0.511 |

## Where the systems differ

Confusion matrices (rows: gold label; columns: prediction):

| System | Gold | → SUPPORT | → CONTRADICT | → INSUFFICIENT | → failed |
| --- | --- | --- | --- | --- | --- |
| Standard RAG | SUPPORT (124) | **92** | 11 | 20 | 1 |
| | CONTRADICT (64) | 6 | **51** | 6 | 1 |
| | INSUFFICIENT (112) | 14 | 19 | **77** | 2 |
| ARGUS B | SUPPORT (124) | **86** | 29 | 9 | 0 |
| | CONTRADICT (64) | 5 | **50** | 9 | 0 |
| | INSUFFICIENT (112) | 6 | 19 | **87** | 0 |
| ARGUS B, no Verifier | SUPPORT (124) | **93** | 22 | 8 | 1 |
| | CONTRADICT (64) | 4 | **52** | 8 | 0 |
| | INSUFFICIENT (112) | 22 | 22 | **68** | 0 |
| ARGUS A | SUPPORT (124) | **85** | 26 | 13 | 0 |
| | CONTRADICT (64) | 5 | **49** | 9 | 1 |
| | INSUFFICIENT (112) | 5 | 18 | **89** | 0 |
| Iterative RAG | SUPPORT (124) | **84** | 10 | 28 | 2 |
| | CONTRADICT (64) | 4 | **53** | 6 | 1 |
| | INSUFFICIENT (112) | 8 | 23 | **78** | 3 |

**ARGUS is better at saying there is not enough evidence.** On the 112 INSUFFICIENT claims, B was correct 87
times against Standard RAG's 77. False SUPPORT verdicts on those claims fell from 14 to 6.

**ARGUS turns more supported claims into CONTRADICT.** On the 124 SUPPORT claims, B called 29 CONTRADICT
against Standard RAG's 11. Arm A shows the same pattern (26), so the Challenge agent is not the cause. These
errors were not audited claim by claim in this run. The internal audit
([challenge-error-audit.md](challenge-error-audit.md)) found false CONTRADICT verdicts caused by direction flips
(an agreeing paper labelled CONTRADICT) and by papers on a different subject treated as contradiction. The same
errors would produce this pattern, but that has not been checked on dev.

**The Verifier is what makes ARGUS cautious.** Without it, B's correct INSUFFICIENT verdicts drop from 87 to
68, and false SUPPORT verdicts on those claims rise from 6 to 22. This is the same failure seen internally.
In return it recovers some supported claims (93 against 86 correct).

## Citations

Citation scores compare cited document and sentence IDs with SciFact's gold annotations over valid responses.
They measure overlap with annotated evidence, not semantic faithfulness, and SciFact's annotations are not
exhaustive.

| System | Document precision | Document recall | Sentence precision | Sentence recall |
| --- | --- | --- | --- | --- |
| Standard RAG | 0.400 | 0.854 | 0.299 | 0.486 |
| Iterative RAG | 0.373 | 0.874 | 0.302 | 0.493 |
| ARGUS A | 0.587 | 0.764 | 0.439 | 0.497 |
| ARGUS B | 0.570 | 0.780 | 0.434 | 0.497 |
| ARGUS B, no Verifier | 0.360 | 0.846 | 0.296 | 0.507 |

With the Verifier, ARGUS cites far fewer non-gold documents. Document precision is 0.57–0.59 against
0.36–0.40 for every system without it, and sentence recall is the same. Document recall is lower (0.76–0.78
against 0.85–0.87). This is the clearest measured benefit of the Verifier on dev. It was not one of the fixed
comparisons, and no significance test was run on it.

## Cost

| System | LLM calls per claim | Tokens per claim | Seconds per claim | Total run time |
| --- | --- | --- | --- | --- |
| Standard RAG | 1.0 | about 2,500 | 12.5 | 1 h 03 min |
| Iterative RAG | 5.1 | about 11,700 | 31.6 | 2 h 38 min |
| ARGUS A | 6.0 | about 7,200 | 25.8 | 2 h 09 min |
| ARGUS B | 6.8 | about 8,300 | 28.3 | 2 h 22 min |
| ARGUS B, no Verifier | 1.0 | about 2,800 | 10.1 | 51 min |

Tokens are recorded prompt plus completion tokens. Seconds per claim is the mean recorded end-to-end latency on
one A40 GPU.

The Evidence Verifier makes one call for every merged abstract (5.0 per claim for A, 5.8 for B), so it makes
83–85% of ARGUS's calls and most of its latency: 5,246 s of Verifier time against 1,972 s for the Judge in arm A.
The Challenge agent adds 0.8 Verifier calls per claim and changes no outcome. Against Standard RAG, ARGUS B
spends 6.8× the calls, 3.3× the tokens and 2.3× the time for 3 more correct claims, which is not significant.

## Internal validation against dev

| System | Internal correct /162 | Internal accuracy | Dev correct /300 | Dev accuracy | Change |
| --- | --- | --- | --- | --- | --- |
| Standard RAG | 119 | 0.735 | 220 | 0.733 | −0.2 pp |
| Iterative RAG | 118 | 0.728 | 215 | 0.717 | −1.1 pp |
| ARGUS A | 126 | 0.778 | 223 | 0.743 | −3.5 pp |
| ARGUS B | 127 | 0.784 | 223 | 0.743 | −4.1 pp |
| ARGUS B, no Verifier (neutral) | 114 | 0.704 | 213 | 0.710 | +0.6 pp |

The baselines and the no-Verifier arm were essentially unchanged from internal to dev. The two arms with
Verifier v2 and Judge v4.1 lost 3.5–4.1 points. Both components were revised after error audits on the 162
internal claims, so their internal scores were probably optimistic. This is why the frozen dev evaluation was
needed. The order of the systems is the same on both sets, but on dev the gaps are within noise.

## Limitations

- **One run per system.** Generation used the frozen configuration once per system. Run-to-run variation
  from the LLM was not measured on dev.
- **Power.** With 300 claims, a difference of a few claims cannot be told apart from noise. The intervals rule
  out large effects but not small ones (for example, up to +6 points for B against Standard RAG).
- **One model and one dataset.** All results are for Qwen3-4B (Q4_K_M) on SciFact. A larger Judge or a
  different domain could behave differently.
- **Gold labels.** INSUFFICIENT means SciFact annotators found no evidence in the corpus. It is not an
  exhaustive relevance judgment, so some "false" SUPPORT or CONTRADICT verdicts may be defensible.
- **The commit hash is not recorded** in run metadata. The model digest, prompt versions and configuration
  are recorded, and they match the frozen configuration.

## Conclusion

On the held-out SciFact dev split, full ARGUS is as accurate as one-shot Standard RAG, not more, and costs
about 7× the LLM calls. Retrieving counter-evidence with the Challenge agent changed no measurable outcome. The
per-document Evidence Verifier is the only component with a consistent effect: it makes the Judge more willing
to answer INSUFFICIENT and its citations more precise. Its accuracy gain (+10 claims) is not significant. The
lead seen on the internal validation set did not hold on held-out claims.
