# ARGUS — Trajectory, Cost, Retrieval and Failure Analysis on Dev

Offline analysis of the five official-dev runs in [dev-results.md](dev-results.md). It reads the saved
`predictions.jsonl` files only. No model or retrieval calls were made, so no result changed. Output:
`outputs/analysis/20261109T025223Z/analysis.json`, from `scripts/analyze_runs.py`.

## Summary

- **The Judge follows the Verifier every time.** Every wrong ARGUS claim traces back to retrieval or
  the Verifier. In 0 cases did the Judge have correctly labelled gold evidence and still get it wrong.
- **Wrong-direction Verifier labels cause most ARGUS errors:** 57 of B's 77 wrong claims. These include
  28 supported claims called CONTRADICT.
- **Retrieval is not the bottleneck.** A gold abstract reaches the verdict stage for 91–96% of
  evidence-bearing claims. Retrieval misses explain only 1–8 wrong claims per system.
- **ARGUS costs 2.2× Standard RAG's time and 3.3× its tokens per correct claim.**
- **The Verifier over-calls CONTRADICT.** It labels 23% of gold SUPPORT abstracts CONTRADICT. Only 27%
  of its CONTRADICT labels fall on gold CONTRADICT abstracts.
- **The Challenge agent mostly finds unrelated papers.** 2% of the abstracts only it found are gold, and
  it found gold that Support missed for just 3 claims.
- **The conclusion holds under every scoring rule.** ARGUS B's lead over Standard RAG is 0.0–1.0
  points in every case.

## 1. Trajectory

An agent step is one LLM call or one retrieval call.

| System | Steps per claim | LLM calls per claim | Retrievals per claim | LLM call success | Budget hit |
| --- | --- | --- | --- | --- | --- |
| Standard RAG | 2.00 | 1.00 | 1.00 | 98.7% | — |
| Iterative RAG | 7.82 | 5.09 | 2.73 | 99.6% | 147 / 300 |
| ARGUS A | 7.00 | 6.00 | 1.00 | 99.9% | 0 |
| ARGUS B | 8.79 | 6.79 | 2.00 | 100% | 0 |
| ARGUS B, no Verifier | 3.00 | 1.00 | 2.00 | 99.7% | 0 |

A failed LLM call is one whose output was unparseable or failed validation. No retrieval call failed.
All failed calls were final verdict calls. Iterative RAG stopped on its round limit for 147 claims,
found no new evidence for 121, repeated a query for 10, and judged the evidence sufficient for 22.
ARGUS is a fixed, bounded graph and recorded 0 budget exhaustions.

## 2. Cost per correct claim

| System | Correct | Seconds per correct | Tokens per correct | LLM calls per correct |
| --- | --- | --- | --- | --- |
| Standard RAG | 220 | 17.1 | 3,439 | 1.36 |
| Iterative RAG | 215 | 44.1 | 16,288 | 7.10 |
| ARGUS A | 223 | 34.7 | 9,670 | 8.07 |
| ARGUS B | 223 | 38.1 | 11,185 | 9.13 |
| ARGUS B, no Verifier | 213 | 14.3 | 3,988 | 1.41 |

Totals include failed outputs. Time is recorded end-to-end latency on one A40.

## 3. Retrieval quality

There are 188 evidence-bearing claims; 112 with no gold evidence are excluded. The first retrieval
uses the claim as the query and is identical in all five systems: MRR@5 0.832, nDCG@5 0.842, Recall@5
0.896, Hit@5 0.910. nDCG uses binary relevance.

| System | Gold recall in evidence pool | Claims with at least one gold abstract in pool |
| --- | --- | --- |
| Standard RAG, ARGUS A | 0.896 | 0.910 |
| ARGUS B (both variants) | 0.910 | 0.926 |
| Iterative RAG | 0.945 | 0.963 |

The evidence pool is what reaches the Verifier (ARGUS) or the final verdict call. The Challenge agent
adds about 1.5 points of gold coverage. Iterative RAG's extra searches add about 5 points, but it
still has fewer correct claims.

## 4. Failure attribution

Each wrong claim goes to the earliest stage that explains it. This is a heuristic, not causal proof:

- **Invalid output:** the response failed parsing or validation.
- **Retrieval miss:** no gold abstract reached the pool.
- **Verifier wrong evidence:** the Verifier gave some abstract the wrong final label and labelled no
  gold abstract correctly.
- **Conflicting evidence:** both correct and wrong-direction evidence reached the Judge.
- **Verifier miss:** the Verifier labelled no gold abstract correctly.
- **Verdict:** the final verdict call (the Judge in ARGUS, or the single generator) made the error
  itself.

| System | Wrong | Invalid output | Retrieval miss | Verifier wrong evidence | Conflicting evidence | Verifier miss | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard RAG | 80 | 4 | 6 | — | — | — | 70 |
| Iterative RAG | 85 | 6 | 1 | — | — | — | 78 |
| ARGUS A | 77 | 1 | 8 | 53 | 0 | 15 | 0 |
| ARGUS B | 77 | 0 | 6 | 57 | 0 | 14 | 0 |
| ARGUS B, no Verifier | 87 | 1 | 5 | — | — | — | 81 |

The 57 wrong-evidence cases in ARGUS B break down as follows:

| Gold label | Predicted | Claims |
| --- | --- | --- |
| SUPPORT | CONTRADICT | 28 |
| INSUFFICIENT | CONTRADICT | 19 |
| INSUFFICIENT | SUPPORT | 6 |
| CONTRADICT | SUPPORT | 4 |

In none of these cases did the Verifier label a gold abstract correctly.

**What it means:** ARGUS moves the decision from the final call into the Verifier, but does not make it
better. The Judge adds no independent check. Improving ARGUS would mean improving per-abstract direction
labelling, which is the same direction-flip problem found in
[challenge-error-audit.md](challenge-error-audit.md). These are dev-split observations; any fix must be
developed on the internal split.

## 5. Verifier accuracy per abstract

Each Verifier label is scored against the gold label of the abstract it judged. Only gold-annotated
abstracts have a reference label.

| Gold label (ARGUS B) | → SUPPORT | → CONTRADICT | → INSUFFICIENT | Recall |
| --- | --- | --- | --- | --- |
| SUPPORT (126) | **84** | 29 | 13 | 0.667 |
| CONTRADICT (60) | 5 | **47** | 8 | 0.783 |

| System | Gold abstracts | Accuracy | Direction flips | CONTRADICT labels on gold CONTRADICT | SUPPORT labels on gold SUPPORT |
| --- | --- | --- | --- | --- | --- |
| ARGUS A | 181 | 0.718 | 31 (17%) | 29% | 64% |
| ARGUS B | 186 | 0.704 | 34 (18%) | 27% | 63% |

A direction flip is SUPPORT labelled CONTRADICT, or the reverse; 29 of B's 34 flips are SUPPORT
labelled CONTRADICT. On non-gold abstracts the Verifier gave 97 CONTRADICT and 45 SUPPORT labels. Some
of these may be defensible because SciFact's annotations are not exhaustive, but the imbalance matches
the false CONTRADICT verdicts in section 4.

## 6. Sensitivity to the scoring rule

The reported rule counts an invalid output as wrong. The other rules change only the 0–6 invalid
outputs per system.

| System | Strict (reported) | Invalid as INSUFFICIENT | Invalid excluded | Raw verdict kept |
| --- | --- | --- | --- | --- |
| Standard RAG | 0.733 | 0.740 | 0.743 | 0.740 |
| Iterative RAG | 0.717 | 0.727 | 0.731 | 0.730 |
| ARGUS A | 0.743 | 0.743 | 0.746 | 0.747 |
| ARGUS B | 0.743 | 0.743 | 0.743 | 0.743 |
| ARGUS B, no Verifier | 0.710 | 0.710 | 0.712 | 0.710 |

"Raw verdict kept" takes the verdict string from an invalid response and ignores citation validity.
The strict rule slightly favours ARGUS, which has the fewest invalid outputs. Under the lenient rules
ARGUS B's lead over Standard RAG shrinks from 1.0 to 0.0–0.3 points. The order of the systems hardly changes,
and the conclusion stays the same.

## 7. What the Challenge agent adds

| | ARGUS B |
| --- | --- |
| Claims where Challenge found an abstract Support did not | 184 of 300 |
| Abstracts found only by Challenge | 236 |
| Of those, gold-annotated | 5 (2%) |
| Claims where only Challenge found a gold abstract | 3 |
| Verifier labels on Challenge-only abstracts | 219 INSUFFICIENT, 12 CONTRADICT, 5 SUPPORT |

Challenge adds an abstract for most claims, but 93% of them are judged irrelevant, and gold evidence
is new for only 3 claims. That is why arms A and B score the same, while B pays 0.8 extra Verifier
calls per claim.

## Evaluation methods that do not apply

- **BLEU, ROUGE, BERTScore:** the output is a label, so exact-match accuracy is the right check.
  Citations are scored against gold IDs, not by text overlap.
- **Validating an LLM-as-judge:** no metric here uses an LLM. All scoring is deterministic code against
  gold labels. The ARGUS Judge is part of the system under test, not an evaluator.
- **Tool-name, tool-description, memory and scratchpad sensitivity:** ARGUS runs a fixed graph. There is
  no free choice of tools and no long-running conversation memory.

## Evaluations not run

These evaluations were considered but not run. Most would need new LLM runs. Rerunning on dev would
break the rule that the frozen dev comparison runs once, so they belong on the 162-claim internal
split.

| Evaluation | What it would test | Why not run | How to run it |
| --- | --- | --- | --- |
| Repeated runs | Run-to-run variation; whether ARGUS's internal 8-claim lead is noise | Every system used temperature 0 and was run once; repeats cost 1–2.5 GPU-hours each | Standard RAG and ARGUS B on the internal split with 2 or more seeds or temperature above 0; report mean and spread |
| Robustness | Stability under reworded claims, a different retrieval k, smaller budgets and prompt-injection text in abstracts | Each change needs new code and a full set of runs per system | One variable changed at a time on the internal split, with the same systems and metrics |
| Judge bias probes | Position bias (shuffled evidence order) and verbosity bias (padded evidence that adds nothing) | Needs a probe harness. Section 4 shows the Judge follows the Verifier, so the Verifier is the bigger risk | Re-judge the saved evidence in a swapped order and with padding; measure how often the verdict stays the same |
| Human check | Whether cited sentences really support the verdict, beyond overlap with gold annotations | Needs manual labelling | Label about 30 sampled claims by hand; report agreement with the automatic scores |

Because none of these were run, the dev results cannot say how stable or how robust any system is.

## Limitations

- The attribution uses SciFact's non-exhaustive gold annotations. A "wrong-direction" abstract may
  sometimes be a defensible reading.
- No claims were audited by hand for this analysis.
- The dev runs were single runs, so no run-to-run variance is available.
