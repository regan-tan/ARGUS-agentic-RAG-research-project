# Milestone 3 Standard RAG summary

This report documents the completed Standard RAG run
`outputs/rag/20261017T115246Z/`. It is a retrospective verification of preserved
artifacts, not a new benchmark run. Aggregate values below were independently recomputed from
`predictions.jsonl` and the local SciFact training annotations. The copies of `metadata.json` and
`metrics.json` under `reports/rag/` are byte-identical to the run artifacts.

## Research question and protocol

Milestone 3 establishes the non-agentic baseline for the ARGUS research question: how accurately
and efficiently can one retrieval followed by one local LLM call verify scientific claims, before
iterative or adversarial agent behavior is introduced?

The experiment searched all 5,183 SciFact corpus abstracts and evaluated the deterministic,
seed-42 internal validation split derived from the 809 official training claims. Independent split
reconstruction reproduced all 162 selected claim IDs: 66 SUPPORT, 35 CONTRADICT, and 61 claims
with no annotated evidence. There were no mixed-label claims in this subset. Empty annotation maps
were assigned the operational evaluation label INSUFFICIENT; this means only that SciFact supplies
no annotated evidence for the claim, not that the scientific statement is false or inherently
unverifiable.

**This is not an evaluation on SciFact's official 300-claim held-out development set.** That set
remained reserved. The local data came from the official release archive with SHA-256
`11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`.

## Retrieval and generation setup

- One dense retrieval per claim, with `k=5`, over complete title-plus-abstract documents.
- `BAAI/bge-small-en-v1.5`, revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; prefixed claim queries, normalized 384-dimensional
  embeddings, and exact `faiss.IndexFlatIP` search. The cached corpus fingerprint was
  `bb15cae5d1852943d020806a38ae4a4154380d3f3ef898d4e9aecd92f7ac08ff`.
- The embedding limit was 512 tokens and 455 document representations were recorded as truncated.
  Generation still received the complete original abstracts without application-side truncation.
- One local Ollama call to `qwen3:4b-instruct-2507-q4_K_M`, a Q4_K_M quantization of the 4.0B Qwen3
  family. Ollama digest:
  `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`; runtime `0.23.2`.
  This quantized build is not claimed to be numerically identical to the BF16 checkpoint.
- Temperature 0, top-k 40, at most 512 generated tokens, an 8,192-token context, and no streaming.
  Prompt version `standard-rag-v1`; system-prompt SHA-256
  `783ed32ac72eeae0b18686a8d35768be3bdb69df49e05ee71129582718b3426f`.

The recorded environment was Python 3.12.14 on Windows 11, AMD64 Family 25 Model 117. An embedding
cache was used; this run therefore does not measure corpus embedding construction.

## Primary results

All 162 selected claims have one unique persisted attempt. Failures remain in classification
denominators as a distinct FAILURE output.

| Metric | Independently verified result |
| --- | ---: |
| Correct predictions | 108 / 162 |
| Accuracy | 0.666667 |
| Macro-F1 | 0.671140 |
| Structurally valid responses | 160 / 162 |
| Inference failures | 1 |
| Invalid responses | 1 |

| Gold class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| SUPPORT | 0.829787 | 0.590909 | 0.690265 | 66 |
| CONTRADICT | 0.604651 | 0.742857 | 0.666667 | 35 |
| INSUFFICIENT | 0.614286 | 0.704918 | 0.656489 | 61 |

Confusion matrix (rows are gold labels; columns are outputs):

| Gold \ output | SUPPORT | CONTRADICT | INSUFFICIENT | FAILURE |
| --- | ---: | ---: | ---: | ---: |
| SUPPORT | 39 | 5 | 21 | 1 |
| CONTRADICT | 2 | 26 | 6 | 1 |
| INSUFFICIENT | 6 | 12 | 43 | 0 |

Classification accuracy and retrieval Recall@5 measure different stages. Accuracy is the fraction
of all 162 claims assigned the annotation-derived verdict. Independently recomputed document
Recall@5 was **0.940594 macro-averaged over the 101 evidence-bearing claims**; Hit@5 was 0.950495.
Claims with no annotated evidence are excluded from those retrieval denominators. High retrieval
recall therefore did not imply high verdict accuracy.

## Citation overlap

The 160 valid responses contained 124 unique cited documents and 185 unique cited document-sentence
pairs. Exact overlap with SciFact annotations was:

| Level | Predicted | Exact matches | Gold denominator | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Document | 124 | 76 | 116 | 0.612903 | 0.655172 |
| Sentence | 185 | 80 | 198 | 0.432432 | 0.404040 |

These are annotation-overlap measures, not semantic faithfulness judgments. The preserved recall
denominators count gold items only for structurally valid responses. If the two failed attempts are
also retained in the citation-recall denominator, as they are for classification, document recall
is 76/118 = 0.644068 and sentence recall is 80/203 = 0.394089. The historical metrics are internally
consistent with their implementation, but should be described as recall conditional on a valid
response.

## Computational cost and failures

| Measurement | Count | Mean | Median |
| --- | ---: | ---: | ---: |
| Retrieval latency | 162 | 69.598 ms | 64.953 ms |
| Generation wall time | 161 | 126.215 s | 111.586 s |
| End-to-end latency | 162 | 127.358 s | 111.724 s |

The summed end-to-end attempt time was approximately 5.731 hours. Across the 161 calls with usage
metadata, Ollama measured 356,666 prompt tokens (mean 2,215.3) and 31,844 completion tokens (mean
197.8), or 388,510 tokens in total. All 162 retrievals and model calls were attempted; one model
call produced no generation metadata because it timed out.

- Claim 953 was an inference failure after the Ollama endpoint timed out. Its gold SUPPORT document
  was retrieved at rank 1.
- Claim 1295 produced a CONTRADICT JSON object without the citation required for that verdict and
  was rejected as an invalid response. Its gold CONTRADICT document was also at rank 1.

## Reproducibility and limitations

The run began at 2026-10-17 11:52:46 UTC and completed at 17:36:40 UTC. Its metadata preserves the
selected IDs, seed, model identities, prompt hash, decoding settings, corpus fingerprint, cache
state, environment, and failure records. The prediction file is complete and duplicate-free. No
metric discrepancy was found between independent recomputation and the archived metrics.

The result is a single internal-validation run with no uncertainty estimate and is not a held-out
generalization result. SciFact annotations are not exhaustive scientific relevance or truth
judgments. A retrieved gold document does not establish that the model selected the annotated
sentences or reasoned from them correctly; conversely, a verdict without a gold-document hit is not
automatically ungrounded because non-gold evidence may be relevant. Exact citation overlap does not
test entailment. Local quantization, runtime, hardware, a 512-token embedding limit, and the one-pass
evidence budget further limit generalization and exact replication.
