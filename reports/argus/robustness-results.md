# ARGUS Retrieval Robustness Evaluation

## Experimental Setup

We evaluated Standard RAG, Iterative RAG, and ARGUS on 162 SciFact internal-validation claims using controlled retrieval perturbations.

Four noise levels (0–3) were tested. Each level represents the requested number of top-k retrieval-result replacements per retrieval call.

Replacements were selected from lower-ranked candidates of the same FAISS search. Already retrieved gold-annotated evidence documents were protected from replacement.

All 12 primary runs completed 162/162 claims. Two additional ARGUS v4.1 runs were completed at noise levels 0 and 3.

## Results

### Classification Accuracy

| Architecture | Noise 0 | Noise 1 | Noise 2 | Noise 3 |
|---|---:|---:|---:|---:|
| Standard RAG | 73.5% | 73.5% | 76.5% | 82.7% |
| Iterative RAG | 72.8% | 75.9% | 74.1% | 67.9% |
| ARGUS original | 68.5% | 67.3% | 63.0% | 58.6% |
| ARGUS v4.1 | 74.7% | — | — | 75.3% |

### Macro-F1

| Architecture | Noise 0 | Noise 1 | Noise 2 | Noise 3 |
|---|---:|---:|---:|---:|
| Standard RAG | 0.737 | 0.738 | 0.778 | 0.831 |
| Iterative RAG | 0.738 | 0.769 | 0.760 | 0.698 |
| ARGUS original | 0.693 | 0.689 | 0.644 | 0.604 |
| ARGUS v4.1 | 0.734 | — | — | 0.742 |

### Noise Injection Delivery at Level 3

| Architecture | Requested | Delivered |
|---|---:|---:|
| Standard RAG | 486 | 481 |
| Iterative RAG | 1,395 | 1,384 |
| ARGUS original | 972 | 962 |
| ARGUS v4.1 | 972 | 962 |

## Discussion

The original ARGUS configuration decreased from 68.5% to 58.6% accuracy as retrieval noise increased.

Iterative RAG decreased from 72.8% to 67.9% between noise levels 0 and 3.

Standard RAG unexpectedly improved from 73.5% to 82.7%.

The additional ARGUS v4.1 experiments achieved 74.7% accuracy without noise and 75.3% with noise level 3.

These results show that the architectures respond differently to retrieval perturbations. However, improved accuracy under this protocol does not establish greater resistance to genuine scientific misinformation.

## Limitations

1. Gold-annotated evidence already retrieved was protected from replacement.
2. Lower-ranked retrieved abstracts were not independently established to be misleading.
3. Actual replacement counts can be lower than requested.
4. ARGUS v4.1 was evaluated only at noise levels 0 and 3.
5. Original ARGUS and v4.1 are distinct experimental configurations; improvements cannot automatically be attributed to one particular component.
6. The 162 internal-validation claims are distinct from the 300 official-development claims used in the main architecture comparison.
7. These results are descriptive; no statistical significance is claimed.

## Reproducibility

The committed `robustness-artifacts/` directory contains the original `metadata.json`, `metrics.json`, and `predictions.jsonl` files for all 14 completed runs.

- `robustness-artifacts/runs/`: 12 main runs
- `robustness-artifacts/v41-runs/`: 2 ARGUS v4.1 runs

The metadata files record experiment settings, model information, corpus fingerprints, claim selections, and prompt configurations.

