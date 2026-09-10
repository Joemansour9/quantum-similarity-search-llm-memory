# Embedding-Model Generalization Check — 2026-07-30

Simulation-only (no QPU quota used). Tests whether two findings established with all-MiniLM-L6-v2
embeddings — **(1) PCA projection substantially outperforms random projection**, and **(2) PCA
tracks classical cosine similarity under query noise while random projection shows a large,
isolated quantum-specific weakness** — replicate with a different, larger embedding model.

## Setup

Re-embedded the exact same 48 documents from the hard corpus's seed=0 draw (confusable
comp.*/rec.autos+motorcycles/sci.electronics topics from 20 Newsgroups) using
**all-mpnet-base-v2** instead of all-MiniLM-L6-v2 — 768-dim vs 384-dim, unit-normalized
(confirmed norm≈1.0), per-dim std=0.0334 (vs MiniLM's 0.0488).

Text extraction: replicated `make_hard_real_corpus`'s exact `rng.choice()` selection logic
against the raw text pool (not the embeddings) to guarantee identical documents, same item keys
(`item_i_cX`), same order.

Since this uses one fixed corpus rather than the original's 5 corpus seeds, query count was
increased to n=50 per cell (matching the original's 5×10=50 total draws) to keep statistical
power comparable.

## Result 1 — PCA-over-random advantage: replicates cleanly

| n_qubits | random | PCA | gap |
|---|---|---|---|
| 4 | 0.160 | 0.820 | 0.660 |
| 6 | 0.200 | 0.840 | 0.640 |
| 8 | 0.240 | 0.880 | 0.640 |
| 10 | 0.260 | 0.900 | 0.640 |
| 12 | 0.320 | 0.920 | 0.600 |
| 14 | 0.360 | 0.940 | 0.580 |

Same qualitative pattern as MiniLM on the hard corpus: PCA near-ceiling from low qubit counts,
random struggling throughout, gap large and consistent (0.58-0.66) across every qubit count
tested. The advantage is not an artifact of MiniLM's specific embedding geometry.

## Result 2 — Query-noise dissociation: replicates cleanly, quantified precisely

n_qubits=4, noise multiplier swept 0.2-20 (scaled by mpnet's own per-dim std):

| mult | random_q | pca_q | classical |
|---|---|---|---|
| 0.2 | 0.900 | 1.000 | 1.000 |
| 1.2 | 0.160 | 0.820 | 1.000 |
| 2.0 | 0.020 | 0.980 | 1.000 |
| 4.0 | 0.020 | 1.000 | 1.000 |
| 6.0 | 0.000 | 1.000 | 1.000 |
| 8.0 | 0.020 | 0.860 | 0.880 |
| 10.0 | 0.020 | 0.660 | 0.640 |
| 12.0 | 0.020 | 0.500 | 0.480 |
| 16.0 | 0.020 | 0.360 | 0.360 |
| 20.0 | 0.000 | 0.300 | 0.300 |

**Mean |PCA − classical| deviation across all 10 noise levels: 0.026.**
**Mean |random − classical| deviation across all 10 noise levels: 0.648.**

25x difference. PCA tracks classical almost exactly at every noise level except one
(mult=1.2: pca=0.820 vs classical=1.000, a real 0.18 gap — the single deviation worth flagging,
not smoothed over). Random projection is disconnected from classical throughout, collapsing to
near-zero by mult=2.0 while classical stays at 1.000 until mult=8.0.

## Conclusion

Both headline findings hold with all-mpnet-base-v2, a different model with double the embedding
dimension (768 vs 384) and different training data/objective than all-MiniLM-L6-v2. This is
evidence the findings reflect something about the retrieval task and encoding method rather than
being specific to one embedding model's geometry — though this is still one alternative model
on one fixed 48-document corpus draw (seed=0 only, not swept across multiple corpus draws the
way the original MiniLM study was), so treat as a single strong generalization data point, not
an exhaustive robustness sweep.

## Files

- `scratch_mpnet_seed0_texts.pkl` — the 48 extracted texts + keys (verified identical documents
  to the MiniLM seed=0 corpus).
- `scratch_mpnet_corpus.pkl` — mpnet embeddings for those 48 texts, keyed the same way.
- `scratch_mpnet_qubit_sweep_results.pkl`, `scratch_mpnet_noise_sweep_results.pkl` — raw sweep
  results backing the tables above.
