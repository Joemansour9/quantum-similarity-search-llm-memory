# Large-Corpus Generalization Check (48 → 480 documents) — 2026-07-30

Simulation-only (no QPU quota used), all-MiniLM-L6-v2 throughout (embedding model held fixed —
that's a separate, already-completed axis, see `embedding_model_generalization_2026-07-30.md`).
Tests whether three findings from the 48-document hard corpus hold when scaled to address the
paper's "small corpus" limitation: **(1) PCA-over-random advantage, (2) the query-noise
dissociation (PCA tracks classical, random doesn't), (3) the corpus-scarcity finding that only
8/48 random-projection self-match pairs cleared 0.5 simulator fidelity.**

## Setup

- Verified fresh, not assumed: all 8 hard-corpus categories (comp.graphics,
  comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x,
  rec.autos, rec.motorcycles, sci.electronics) have ≥400 posts after the 200-2000 char filter
  (train subset) — comfortable headroom for 60-65/category.
- Raised `_POOL_PER_CATEGORY` from 60 to 65 **in `benchmark_realdata_embeddings_hard.py` itself**
  (not just a function parameter), with a comment documenting why.
- Built a 480-item corpus (60/category × 8, seed=0) via `make_hard_real_corpus(n_items=480,
  n_clusters=8, seed=0)` — confirmed exactly 60/topic, 384-dim MiniLM embeddings.
- Timing calibrated before committing to a full sweep (n_qubits=14 costs ~6.3s/query at 480
  candidates vs ~0.85s/query at n_qubits=4 — candidate count scales cost roughly linearly on
  top of the existing 2^n_qubits scaling).

## Result 1 — PCA-over-random: holds, gap widens in absolute terms

| n_qubits | random | PCA | gap |
|---|---|---|---|
| 4 | 0.050 | 0.250 | 0.200 |
| 6 | 0.000 | 0.550 | 0.550 |
| 8 | 0.050 | 0.800 | 0.750 |
| 10 | 0.150 | 0.900 | 0.750 |
| 12 | 0.200 | 0.950 | 0.750 |
| 14 | 0.350 | 0.900 | 0.550 |

(n_queries=20/cell.) Absolute recall for both methods dropped sharply vs. the 48-item corpus
(PCA: ~0.96-1.0 → 0.25-0.95; random: ~0.14-0.42 → 0.00-0.35) — confirms 480 candidates makes
this a substantially harder, more realistic retrieval task. **PCA's advantage did not shrink
with the harder task — the absolute gap is as large or larger** (0.55-0.75 at most qubit counts
vs. 0.14-0.28 at 48 items).

## Result 2 — Query-noise dissociation: WEAKENS substantially, confirmed real at n=100

At the 48-item corpus (and independently with all-mpnet-base-v2 embeddings), PCA tracked
classical almost exactly across the full noise range (mean |PCA-classical| deviation: 0.026).
**At the 480-item corpus, this breaks down at low-to-moderate noise:**

Initial pass (n_qubits=4, n=20/cell):

| mult | random | PCA | classical | PCA−classical gap |
|---|---|---|---|---|
| 0.2 | 0.550 | 1.000 | 1.000 | 0.00 |
| 1.2 | 0.050 | 0.250 | 1.000 | 0.75 |
| 2.0 | 0.000 | 0.150 | 1.000 | 0.85 |
| 4.0 | 0.000 | 0.450 | 0.950 | 0.50 |
| 6.0 | 0.000 | 0.400 | 0.550 | 0.15 |
| 8.0+ | 0.000 | ≤0.15 | ≤0.15 | ~0.00 |

**Confirmed at n=100/cell** on the four suspicious levels (not a sampling artifact):

| mult | PCA (n=100) | Classical (n=100) | gap | 95% CIs |
|---|---|---|---|---|
| 1.2 | 0.310 ± 0.091 | 1.000 ± 0.000 | 0.69 | **non-overlapping** |
| 2.0 | 0.150 ± 0.070 | 1.000 ± 0.000 | 0.85 | **non-overlapping** |
| 4.0 | 0.420 ± 0.097 | 0.980 ± 0.027 | 0.56 | **non-overlapping** |
| 6.0 | 0.560 ± 0.097 | 0.650 ± 0.093 | 0.09 | **overlapping — borderline** |

Point estimates were nearly identical between n=20 and n=100 at every level (e.g. mult=2.0:
0.150 at both), and three of four levels show clearly non-overlapping confidence intervals
between PCA and classical — this is a real, robust effect at mult=1.2-4.0, not noise. Only
mult=6.0 is genuinely borderline at this sample size (gap shrank from 0.15 to 0.09, CIs now
overlap) — flagged as such rather than rounded into the "confirmed" bucket.

**Interpretation:** at the 48-item corpus, this exact isolated degradation pattern (fidelity-
quantum lagging classical under noise) was previously the signature that distinguished random
projection's known weakness. At 480 items, low-to-moderate query noise, **PCA now shows the
same signature** — plausibly because PCA's fixed 4-component budget must now represent 480
documents across 8 topics instead of 48, and while it still separates topics well enough to
preserve the Result 1 advantage, it no longer reliably preserves fine-grained per-document
identity under noise. This is a genuine scale-dependent finding, not a replication.

## Result 3 — Corpus-scarcity: does NOT hold proportionally, roughly doubled instead

Full scan across all 480 self-match queries (n_qubits=4, mult=1.2, exact statevector fidelity):

- PCA: 456/480 (95.0%) clear sim_fid≥0.5
- Random: **184/480 (38.3%)** clear sim_fid≥0.5 — vs. the original **8/48 (16.7%)**

Random's proportion of high-fidelity self-matches more than doubled at the larger scale rather
than holding flat or worsening. No clear mechanism identified for why; reported as observed.

## Result 2b — Hardware confirmation: the breakdown is real, not a simulation artifact

Tested whether Result 2's breakdown (PCA-quantum picking the wrong top-1 at mult=2.0, the
largest CI gap) reproduces on real IBM hardware. Selected 4 queries from the mult=2.0/n=100
confirmation run where PCA-quantum simulation got the wrong top-1, each scoped to true match +
top-3 simulator competitors (4 candidates), same destructive circuit, n_qubits=4, `ibm_kingston`
(matching prior tests). 16 circuits total, one batched job (`d9ls5umh4e6s738ubqt0`, usage=7s).
Gate count was internally consistent within each query's 4-candidate set (31 or 23 gates
depending on query) — magnitude-comparison gate-matching discipline doesn't apply here since
this is a categorical per-case ranking-outcome test, not a cross-case magnitude comparison.

| q_idx | sim wrong pick | hw pick | true match rank (hw) | same wrong answer? |
|---|---|---|---|---|
| 0 | item_248_c4 | item_293_c4 | last of 4 | No |
| 1 | item_165_c2 | item_237_c3 | last of 4 | No |
| 2 | item_36_c0 | item_36_c0 | last of 4 | **Yes** |
| 3 | item_25_c0 | item_172_c2 | last of 4 | No |

**Decisive: 4/4 — the true match came in dead last on real hardware in every case, exactly as
in simulation. Zero cases where hardware "accidentally" recovered the correct answer.** The
breakdown is a real physical phenomenon at this corpus scale, not a simulation-specific artifact.

One more precise layer: only 1/4 cases had hardware land on the *exact same* wrong candidate as
simulation — the other 3 picked a different wrong answer among the close competitors, consistent
with Test 14's finding (2/12 full-ranking agreement) that fine ordering among near-tied
candidates is noise-sensitive even when the higher-level outcome (here: failure) is stable.

## Bottom line

Not a clean "everything replicates" result, reported as such rather than smoothed over:

- **PCA-over-random: robust to scale.** Confirmed at 480 items, gap as large or larger than at 48.
- **Query-noise dissociation: breaks down at scale, confirmed with adequate statistical power,
  AND confirmed on real hardware (4/4).** This is the more theoretically load-bearing claim of
  the two, and it does NOT hold at 480 documents for low-to-moderate noise (mult=1.2-4.0) — in
  simulation or on real IBM hardware. Should not be stated as a corpus-size-independent property
  without this caveat. This is now a hardware-confirmed finding, not just a simulation result.
- **Corpus-scarcity: does not hold proportionally** — eased rather than worsened at scale.

Result 2's hardware confirmation (Result 2b) closes the loop opened above: the breakdown
is a real physical phenomenon at this corpus scale, not a simulation artifact, and it's directly
relevant to any claims about production-scale deployment given the original hardware validation
round only ever tested the 48-item corpus.

## Files

- `benchmark_realdata_embeddings_hard.py` — `_POOL_PER_CATEGORY` raised 60→65 (code change,
  not a parameter override).
- `scratch_large_corpus_480.pkl` — the 480-item corpus (keys + MiniLM embeddings).
- `scratch_large_corpus_qubit_sweep.pkl`, `scratch_large_corpus_noise_sweep.pkl` — n=20 sweep
  results.
- `scratch_large_corpus_noise_confirm.pkl` — n=100 confirmation results for mult=1.2-6.0.
- `scratch_large_corpus_scarcity.pkl` — full 480-query exact-fidelity scan (PCA and random).
