# ARGUS Evaluation Report

**Project:** ARGUS (Adversarial Retrieval and Grounded Understanding System), SMU IS469

**Research question:** Does explicit counter-evidence retrieval in multi-agent RAG improve scientific
claim verification enough to justify its extra inference cost?

**Answer:** No. On the held-out SciFact dev split, full ARGUS is as accurate as one-shot Standard RAG
(74.3% against 73.3%, not significant), and it uses about 7 times as many LLM calls.

This report merges [dev-results.md](dev-results.md) and
[dev-trajectory-analysis.md](dev-trajectory-analysis.md). Those two files hold the full tables.

## 1. Setup

**Task.** Each system gets a scientific claim and returns SUPPORT, CONTRADICT or INSUFFICIENT, with
cited abstract sentences.

**Systems compared:**

| System | Pipeline | LLM calls per claim |
| --- | --- | --- |
| Standard RAG | One retrieval, then one verdict call | 1.0 |
| Iterative RAG | One agent that can rewrite its query and search again, up to a budget | 5.1 |
| ARGUS A | Support agent, Evidence Verifier on each abstract, Judge | 6.0 |
| ARGUS B (full) | Support and Challenge agents, Verifier, Judge | 6.8 |
| ARGUS B, no Verifier | Support and Challenge agents, Judge only | 1.0 |

**Fixed across all systems:**
- Generator `qwen3:4b-instruct-2507-q4_K_M` at temperature 0.
- Pinned BGE dense retriever over all 5,183 SciFact abstracts.
- Same evidence budget for every system.
- One A40 GPU.

**Protocol:**
- All design and prompt changes were made on a 162-claim internal validation split drawn from the
  training claims.
- Each system's configuration was then frozen and run once on the official labeled dev split: 300
  claims, of which 124 are SUPPORT, 64 CONTRADICT and 112 INSUFFICIENT.
- Four comparisons were fixed before these runs. Each was tested with an exact McNemar test, with Holm
  correction across the four, and a 95% paired bootstrap interval (10,000 resamples, seed 42).
- All scoring is deterministic code against SciFact gold labels. No LLM is used as an evaluator.

## 2. Final-answer correctness

| System | Correct /300 | Accuracy | Macro-F1 | Invalid outputs |
| --- | --- | --- | --- | --- |
| Standard RAG | 220 | 0.733 | 0.733 | 4 |
| Iterative RAG | 215 | 0.717 | 0.722 | 6 |
| ARGUS A | 223 | 0.743 | 0.733 | 1 |
| **ARGUS B** | **223** | **0.743** | 0.732 | 0 |
| ARGUS B, no Verifier | 213 | 0.710 | 0.703 | 1 |

Invalid outputs stay in the denominator and count as wrong.

**Comparisons fixed in advance (ARGUS B against each):**

| Compared with | McNemar p | Holm p | Accuracy difference (95% CI) |
| --- | --- | --- | --- |
| Standard RAG | 0.788 | 1.000 | +1.0 pp (−4.0, +6.0) |
| Iterative RAG | 0.389 | 1.000 | +2.7 pp (−2.7, +8.0) |
| ARGUS A (no Challenge) | 1.000 | 1.000 | 0.0 pp (−1.7, +1.7) |
| ARGUS B, no Verifier | 0.220 | 0.881 | +3.3 pp (−1.3, +8.0) |

**None of the differences is significant, and every interval includes zero.**

**Error types.** Overall accuracy hides a trade-off. Compared with Standard RAG, ARGUS B gets more
INSUFFICIENT claims right (87 against 77). It also calls more supported claims CONTRADICT (29 against
11). The two effects cancel out.

**Generalisation.** On the 162 internal claims, ARGUS B led Standard RAG by 4.9 points (78.4% against
73.5%). On dev the lead is 1.0 point. Standard RAG's accuracy held steady, while ARGUS dropped 4.1
points. Its Verifier and Judge prompts were revised on the internal set, so part of that lead was
probably fitted to it.

## 3. Verifiable checks and scoring-rule sensitivity

**Schema and citation checks.** Every response must parse as JSON and give a valid verdict. Each
citation must point to a retrieved document and a sentence that exists in it. Outputs that fail are
recorded as invalid, never silently mapped to INSUFFICIENT. Valid output rates were 98–100%.

**Sensitivity to the scoring rule.** Accuracy was recomputed under three alternative rules for
invalid outputs:
- count them as INSUFFICIENT;
- leave them out of the denominator;
- keep the verdict in the raw response and ignore citation validity.

| System | Strict (reported) | Invalid as INSUFFICIENT | Invalid excluded | Raw verdict kept |
| --- | --- | --- | --- | --- |
| Standard RAG | 0.733 | 0.740 | 0.743 | 0.740 |
| ARGUS B | 0.743 | 0.743 | 0.743 | 0.743 |

Only these two systems are shown; the other three move by at most 1.4 points. ARGUS B's lead over
Standard RAG is between 0.0 and 1.0 points under every rule, so the conclusion does not depend on the
rule.

## 4. Retrieval and evidence quality

**Retrieval.** The first retrieval is identical in all systems. Over the 188 claims with gold evidence,
it scores MRR@5 0.832, nDCG@5 0.842, Recall@5 0.896 and Hit@5 0.910. A gold abstract reaches the final
stage for 91–96% of these claims, so retrieval is not the bottleneck.

**Citations.** Citations are compared with SciFact's gold annotations.

| System | Document precision | Document recall | Sentence precision | Sentence recall |
| --- | --- | --- | --- | --- |
| Standard RAG | 0.400 | 0.854 | 0.299 | 0.486 |
| ARGUS B | 0.570 | 0.780 | 0.434 | 0.497 |
| ARGUS B, no Verifier | 0.360 | 0.846 | 0.296 | 0.507 |

The Verifier makes citations much more precise. This is its clearest measured benefit, but it was not
one of the comparisons fixed in advance.

**Verifier accuracy per abstract.** The Verifier labels each abstract separately. On the 186
abstracts that SciFact annotates, its label is correct 70% of the time. It labels 23% of supported
abstracts CONTRADICT. Only 27% of its CONTRADICT labels fall on abstracts annotated as contradicting.
**The Verifier is biased toward CONTRADICT.**

## 5. Trajectory, reliability and efficiency

An agent step is one LLM call or one retrieval call.

| System | Steps per claim | LLM call success | Seconds per correct claim | Tokens per correct claim |
| --- | --- | --- | --- | --- |
| Standard RAG | 2.0 | 98.7% | 17.1 | 3,439 |
| Iterative RAG | 7.8 | 99.6% | 44.1 | 16,288 |
| ARGUS A | 7.0 | 99.9% | 34.7 | 9,670 |
| ARGUS B | 8.8 | 100% | 38.1 | 11,185 |
| ARGUS B, no Verifier | 3.0 | 99.7% | 14.3 | 3,988 |

- **No retrieval call failed.** Every failed LLM call was a final verdict call.
- **ARGUS never ran out of budget.** It is a fixed, bounded graph.
- **Iterative RAG often did.** It hit its round limit on 147 of 300 claims.
- **Cost.** Per correct claim, ARGUS B costs 2.2 times Standard RAG's time and 3.3 times its tokens.
  The Verifier accounts for 83–85% of ARGUS's LLM calls.

## 6. Failure attribution and component ablations

**Failure attribution.** Each wrong claim was assigned to the earliest stage that explains it:

| System | Wrong | Invalid output | Retrieval miss | Verifier error | Final verdict call |
| --- | --- | --- | --- | --- | --- |
| Standard RAG | 80 | 4 | 6 | — | 70 |
| ARGUS B | 77 | 0 | 6 | 71 | 0 |

"Verifier error" covers two cases:
- 57 claims where the Verifier gave wrong-direction evidence;
- 14 claims where it labelled no gold abstract correctly.

**In ARGUS B the Judge never overrode correct evidence. It followed the Verifier every time.** Most
errors are direction flips: 28 supported claims were called CONTRADICT. Making ARGUS better therefore
means fixing how the Verifier labels each abstract, not adding agents.

**Ablations.**
- **Challenge agent.** It has no measurable effect: arms A and B both score 223, and they disagree on
  only 6 claims. It found 236 abstracts that Support did not. Only 5 of them (2%) are gold, and it
  added new gold evidence for just 3 claims.
- **Verifier.** Removing it costs 10 claims (p = 0.22). It also lowers correct INSUFFICIENT verdicts
  from 87 to 68 and lowers document citation precision from 0.57 to 0.36.

## 7. Coverage of evaluation methods

**Applied:**
- Final-answer accuracy and macro-F1, per-label F1 and confusion matrices.
- Schema and citation validation, and sensitivity to the scoring rule.
- Retrieval Recall@k, Hit@k, MRR and nDCG, plus citation precision and recall.
- Steps, call success, stop reasons and budget hits.
- Latency, tokens and cost per correct claim.
- Trace-based failure attribution and component ablations.
- Paired significance tests with uncertainty intervals.

**Not applicable:**
- **BLEU, ROUGE and BERTScore:** verdicts are labels, so exact match is the right check.
- **Validating an LLM-as-judge:** no LLM is used for scoring. The ARGUS Judge is part of the system,
  not an evaluator.
- **Tool-name, tool-description and memory sensitivity:** ARGUS is a fixed graph, with no free choice
  of tools and no long conversation memory.

**Not run.** These would need new LLM runs. They belong on the internal split, so that the frozen dev
comparison is used only once.

| Evaluation | What it would add |
| --- | --- |
| Repeated runs at temperature above 0 | Run-to-run variation; all runs used temperature 0 and ran once |
| Robustness (reworded claims, different k, smaller budgets, prompt injection) | Stability under controlled changes |
| Judge bias probes (evidence order, padded evidence) | Position and verbosity bias; lower priority because the Judge follows the Verifier |
| Human check of about 30 claims | Whether the Verifier's direction flips are real errors or defensible readings |

## 8. Limitations

- **One run per system.** Run-to-run variation was not measured.
- **Power.** With 300 claims, small effects cannot be ruled out. For example, a lead of up to 6 points
  for ARGUS B over Standard RAG is still possible.
- **One model and one dataset.** All results are for Qwen3-4B (Q4_K_M) on SciFact.
- **Gold labels are not exhaustive.** Some "wrong" verdicts or Verifier labels may be defensible.
- **Attribution is a heuristic.** No claims were audited by hand.

## 9. Conclusion

On held-out SciFact claims, explicit counter-evidence retrieval does not improve claim verification.
Full ARGUS matches one-shot Standard RAG, at about 7 times the LLM calls and 2.2 times the time per
correct claim.

The Challenge agent rarely finds new relevant evidence. The per-abstract Verifier is the only
component with a consistent effect. It makes citations more precise and makes the system more willing
to say INSUFFICIENT. It also mislabels supported abstracts as contradicting, and because the Judge
always follows it, that error passes straight through to the final verdict.

This is a negative result from a test fixed in advance.
