# Milestone 2 retrieval summary

This is a preserved summary of the historical retrieval experiment, not a new benchmark run.
It is based on the verified local artifact `outputs/retrieval/20261016T012109Z/` produced on
2026-10-16. The original, ignored experiment outputs remain unchanged locally.

## Purpose and setup

The experiment compared document-level SciFact retrieval backends to choose a common retriever
for later claim-verification systems. Each of the 5,183 SciFact corpus records was represented as
its title plus complete ordered abstract. A deterministic, train-derived validation split used seed
42 and a 0.2 fraction: 162 of 809 training claims (101 evidence-bearing and 61 no-evidence).
Recall and Hit denominators include only evidence-bearing claims; the official 300-claim development
set was not used for selection or prompt tuning.

- BM25: `rank-bm25` `BM25Okapi` with a lowercase regex tokenizer.
- Dense: `BAAI/bge-small-en-v1.5` at immutable revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; CPU embeddings, normalized float32 vectors, and
  exact `faiss.IndexFlatIP` cosine search. Claims used BGE's retrieval query prefix.
- Hybrid: reciprocal-rank fusion of the above rankings with a top-50 candidate pool and `c=60`.

## Historical results

| Method | Recall@3 | Recall@5 | Recall@10 | Hit@5 | Mean warm query latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.800 | 0.847 | 0.886 | 0.861 | 47.3 ms |
| Dense BGE + FAISS | 0.907 | 0.941 | 0.980 | 0.950 | 28.0 ms |
| Hybrid RRF | 0.866 | 0.886 | 0.913 | 0.901 | 75.2 ms |

Dense retrieval was selected because it had the highest recorded Recall@3/5/10 and Hit@5 while
also being faster than the measured BM25 and RRF runs after initialization. Its corpus embedding
cache was reused in Milestone 3; constructing embeddings and the FAISS index is deliberately
reported separately from warm query latency.

## Findings and caveats

The BGE model had a maximum sequence length of 512 and the benchmark recorded 455 corpus document
representations that were truncated for embedding. This did **not** change source documents: later
generation receives the original complete SciFact abstract sentences. Results are document-level
retrieval measurements, not evidence-faithfulness or claim-verdict results. SciFact annotations are
not exhaustive relevance judgments, claims without annotated evidence are excluded from retrieval
metric denominators, and the small internal split is not the final official development evaluation.

Only this concise report is intended for Git tracking. Model weights, embedding-cache arrays, FAISS
indexes, raw SciFact files, and large per-claim rankings remain local and ignored.
