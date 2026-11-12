# Add-on: Does a stronger model change the result? (GPT-5.6 Luna)

**Runs (dev, 300 claims, GPT-5.6 Luna):** all under `outputs/openai/dev/`

| System | Run |
| --- | --- |
| Standard RAG | `rag/20261111T184811Z` |
| ARGUS A (no Challenge) | `argus-A/20261111T190118Z` |
| ARGUS B (Full ARGUS) | `argus-B/20261111T190104Z` |
| ARGUS B, no Verifier (neutral Judge wording) | `argus-B-no-verifier/20261111T190612Z`; repeat `20261111T190812Z` |
| Iterative RAG (exploratory, added later) | `iterative_rag/20261112T024655Z` |

All five reported runs completed 300 of 300 claims. Qwen3-4B figures come from
[dev-results.md](dev-results.md). The frozen settings, the pre-specified comparisons and the
deviations are recorded in `outputs/openai/FROZEN.md`.

## Summary

**A stronger model did not make ARGUS beat Standard RAG.** With GPT-5.6 Luna, Standard RAG got 237
of 300 dev claims right, ARGUS A 234 and Full ARGUS (B) 228. None of the three pre-specified
comparisons is significant (Holm p ≥ 0.56). Full ARGUS was 3.0 points below Standard RAG
(95% CI −7.0 to +1.0).

**The stronger model fixed the Verifier's main error but ARGUS still gained nothing.** The Verifier
labelled 3% of supporting gold abstracts CONTRADICT, against 23% with Qwen3-4B. Yet the same
model also improved one-shot Standard RAG, by 17 claims (220 to 237), more than it improved
ARGUS B (223 to 228).

**Neither ARGUS component helped with GPT-5.6 Luna.** Removing the Challenge agent (ARGUS A) gave
6 more correct claims, and removing the Verifier gave 1 fewer; neither difference is significant.
With Qwen3-4B the Verifier had added 10 claims.

**Single-agent iteration did not help either (exploratory).** Iterative RAG, run after the other
results had been seen, got 234 of 300, 3 fewer than Standard RAG (McNemar p = 0.55) and 6 more than
Full ARGUS (p = 0.41), at 3.5 LLM calls per claim.

**Answer to the add-on question:** the lack of improvement in the main experiment is not explained
only by the small model. On SciFact, neither the multi-agent pipeline nor single-agent iteration
outperformed one-shot RAG with either model, while costing 3.5 to 7 times the LLM calls.

## Setup

| Item | Main experiment | Add-on |
| --- | --- | --- |
| LLM | Qwen3-4B-Instruct Q4_K_M, Ollama 0.40.5, one A40 | `gpt-5.6-luna`, OpenAI Responses API |
| Claims | 300 official SciFact dev claims | same |
| Retrieval | BGE-small + FAISS, top 5 | same |
| Prompts | standard-rag-v1, Verifier v2, Judge v4.1 | same, not re-tuned |
| Decoding | temperature 0, 512 output tokens, 8,192 context | temperature 0, 512 output tokens, reasoning effort `none`, strict JSON schema |
| Budgets, validation, scoring | 10 papers, 12 LLM calls per claim; invalid output counts as wrong | same |

The OpenAI client is `src/argus/llm/openai_client.py` (branch `feat/openai-provider`, standard
library only). It retries only rate-limit, server and connection errors and records the model name
returned with every call.

**Deviations from the main protocol:**

- The 162-claim internal stage was skipped. Smoke tests (5 claims per system) were clean, so no
  setting was changed before the dev runs.
- Standard RAG was run on dev before the comparisons were written down. Comparisons 1 and 2 were
  recorded before the ARGUS runs started; comparison 3 was added while they were running, before
  any of their accuracy figures were looked at.
- The no-Verifier run was started twice by mistake. The first started is reported; the second is
  used only as a repeat.
- Iterative RAG was not in the original add-on, because `run_iterative_rag.py` had no OpenAI
  option. The option was added on 2026-11-12, after all other GPT-5.6 Luna results had been seen,
  with the same frozen settings and the main experiment's iterative budgets (3 retrieval rounds,
  6 LLM calls, 5 abstracts per round, prompt version iterative-rag-v3). A 5-claim smoke test was
  clean, and the dev split was then run once. Its comparisons are exploratory and are not in the
  Holm-corrected set.

## Results on the dev split

| System | Qwen3-4B correct | GPT-5.6 Luna correct | GPT accuracy | GPT macro-F1 | F1 S | F1 C | F1 I | GPT invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard RAG | 220 | **237** | 79.0% | 0.783 | 0.833 | 0.743 | 0.773 | 0 |
| ARGUS A (no Challenge) | 223 | 234 | 78.0% | 0.786 | 0.804 | 0.772 | 0.782 | 6 |
| ARGUS B (Full) | 223 | 228 | 76.0% | 0.763 | 0.800 | 0.730 | 0.760 | 7 |
| ARGUS B, no Verifier | 213 | 227 | 75.7% | 0.761 | 0.795 | 0.731 | 0.758 | 8 |
| Iterative RAG (exploratory) | 215 | 234 | 78.0% | 0.773 | 0.820 | 0.732 | 0.767 | 0 |

S, C, I = SUPPORT, CONTRADICT, INSUFFICIENT. Invalid outputs count as wrong. GPT accuracy and
macro-F1 are from each run's `metrics.json`. The no-Verifier repeat scored 225.

### Pre-specified comparisons (GPT-5.6 Luna)

Exact two-sided McNemar test on per-claim correctness, Holm correction across the three. Intervals
are 95% paired bootstrap intervals (10,000 resamples, seed 42), ARGUS B minus the other system.

| ARGUS B vs | Only B right | Only other right | p | Holm p | Difference, points (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Standard RAG | 14 | 23 | 0.188 | 0.563 | −3.0 (−7.0, +1.0) |
| ARGUS A (no Challenge) | 8 | 14 | 0.286 | 0.573 | −2.0 (−5.0, +1.0) |
| ARGUS B, no Verifier | 16 | 15 | 1.000 | 1.000 | +0.3 (−3.3, +4.0) |

No comparison is significant, and every interval includes zero. The interval against Standard RAG
rules out an ARGUS advantage larger than about 1 point.

### Exploratory: Iterative RAG (GPT-5.6 Luna)

Added after the comparisons above had been seen; uncorrected. Same tests and bootstrap settings.

| Comparison | Only first right | Only second right | p | Difference, points (95% CI) |
| --- | --- | --- | --- | --- |
| ARGUS B vs Iterative RAG | 15 | 21 | 0.405 | −2.0 (−6.0, +2.0) |
| Iterative RAG vs Standard RAG | 4 | 7 | 0.549 | −1.0 (−3.3, +1.0) |

Iterative RAG and Standard RAG give the same correctness on 289 of 300 claims. Rewriting the query
found slightly more gold evidence (gold abstract in the evidence pool for 93.6% of evidence-bearing
claims, against 91.0% for Standard RAG) but did not turn it into more correct verdicts. The agent
stopped because evidence was judged sufficient on 191 claims, because a new search returned no new
abstracts on 27, and because it reached the 3-round limit on 82. It produced no invalid outputs.

Confusion matrix (rows gold; SUPPORT / CONTRADICT / INSUFFICIENT / invalid): SUPPORT (124)
**98** / 11 / 15 / 0; CONTRADICT (64) 4 / **52** / 8 / 0; INSUFFICIENT (112) 13 / 15 / **84** / 0.
This is close to Standard RAG's pattern, not ARGUS B's.

### Where the mistakes moved (confusion matrices, GPT-5.6 Luna)

Rows gold, columns prediction (SUPPORT / CONTRADICT / INSUFFICIENT / invalid):

| Gold | Standard RAG | ARGUS B |
| --- | --- | --- |
| SUPPORT (124) | **100** / 10 / 14 / 0 | **90** / 7 / 23 / 4 |
| CONTRADICT (64) | 3 / **52** / 9 / 0 | 1 / **46** / 15 / 2 |
| INSUFFICIENT (112) | 13 / 14 / **85** / 0 | 10 / 9 / **92** / 1 |

With Qwen3-4B, ARGUS's typical error was calling a supported claim CONTRADICT (29 claims). With
GPT-5.6 Luna that error almost disappears (7), but ARGUS now answers INSUFFICIENT too often: 23
supported and 15 contradicted claims, against 14 and 9 for Standard RAG.

### Verifier accuracy per abstract (gold-annotated abstracts only)

| Gold label of abstract | Model | → SUPPORT | → CONTRADICT | → INSUFFICIENT | Right |
| --- | --- | --- | --- | --- | --- |
| SUPPORT (126) | Qwen3-4B | 84 | 29 | 13 | 66.7% |
| SUPPORT (126) | GPT-5.6 Luna | **107** | **4** | 15 | **84.9%** |
| CONTRADICT (60) | Qwen3-4B | 5 | 47 | 8 | 78.3% |
| CONTRADICT (60) | GPT-5.6 Luna | 4 | 46 | 10 | 76.7% |

ARGUS B arms only. The direction flip that drove ARGUS's errors with Qwen3-4B (supporting abstract
labelled CONTRADICT) fell from 23% to 3%. Accuracy on contradicting abstracts did not change.

## Sensitivity to the citation rule

Every invalid ARGUS output with GPT-5.6 Luna but one broke the Judge v4.1 citation limit in the
same way: the Judge cited one paper in two entries whose sentences, once merged, exceeded the
3-sentence limit. That limit exists only in the ARGUS Judge, so it can penalise ARGUS and not
Standard RAG. The conclusion does not depend on it:

| System | As reported | Invalid excluded (accuracy on valid outputs) | Invalid verdicts accepted as written |
| --- | --- | --- | --- |
| Standard RAG | 237 | 79.0% (0 invalid) | 237 |
| ARGUS A | 234 | 79.6% | 238 |
| ARGUS B | 228 | 77.8% | 234 |
| ARGUS B, no Verifier | 227 | 77.7% | 235 |

Even with every rejected verdict accepted, Full ARGUS (234) does not exceed Standard RAG (237).

## Run-to-run variation

The two no-Verifier runs have identical settings. They gave the same answer on 289 of 300 claims
and scored 227 and 225. Single runs of the same system can therefore differ by a few claims, which
is the same size as several of the differences above.

## Cost (GPT-5.6 Luna, per claim)

| System | LLM calls | Input tokens | Output tokens | Seconds | Cost for 300 claims |
| --- | --- | --- | --- | --- | --- |
| Standard RAG | 1.00 | 2,278 | 83 | 1.7 | about $0.83 |
| ARGUS A | 6.00 | 6,871 | 383 | 9.9 | about $2.75 |
| ARGUS B | 6.79 | 7,950 | 432 | 11.2 | about $3.16 |
| ARGUS B, no Verifier | 1.00 | 2,635 | 76 | 1.7 | about $0.93 |
| Iterative RAG | 3.46 | 7,844 | 231 | 6.3 | about $2.77 |

Cost uses OpenAI's listed price of $1 per million input and $6 per million output tokens; the
billing dashboard is authoritative. Latency depends on the API, not local hardware, so it is not
comparable with the A40 runs. Full ARGUS used 6.8 times the calls and about 3.8 times the cost of
Standard RAG for 9 fewer correct claims.

## Exploratory, across models (descriptive)

| Comparison | Only GPT right | Only Qwen right | Exact McNemar p |
| --- | --- | --- | --- |
| Standard RAG | 35 | 18 | 0.027 |
| ARGUS A | 36 | 25 | 0.200 |
| ARGUS B | 36 | 31 | 0.625 |
| Iterative RAG | 41 | 22 | 0.023 |

Not pre-specified and not corrected. The models differ in more than size (precision, training
data, provider), so this compares a different, stronger model, not model size alone.

## Interpretation and caveats

- The result supports the main conclusion: ARGUS did not outperform one-shot RAG with a much
  stronger reader either. The add-on does not identify why ARGUS gained less than Standard RAG;
  one possibility, not tested here, is that verifying each abstract separately removes context that
  a single call over all five abstracts can use, and makes the final answer more cautious.
- Prompts were developed on Qwen3-4B's errors and not re-tuned for GPT-5.6 Luna.
- SciFact has been public since 2020 and may be in GPT-5.6 Luna's training data. This would raise
  all GPT scores; comparisons within GPT-5.6 Luna are less affected than the absolute numbers.
- One run per system (plus one repeat), 300 claims, one dataset. Small differences cannot be
  separated from run-to-run variation.
