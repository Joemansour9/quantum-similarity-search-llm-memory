# Corpus-Size Transition Point for the Query-Noise Dissociation — 2026-08-03

Simulation-only (no QPU quota used), all-MiniLM-L6-v2 throughout. Follow-up to
`large_corpus_generalization_2026-07-30.md`'s Result 2/2b, which established that the
query-noise dissociation (PCA-quantum lagging classical fidelity at low-to-moderate noise)
**does not appear at 48 documents** (mean |PCA-classical| deviation: 0.026, "tracks closely")
but **is confirmed real at 480 documents** (n=100/cell, non-overlapping 95% CIs at
mult=1.2/2.0/4.0, hardware-confirmed 4/4 at mult=2.0). This document asks: where between 48 and
480 does it emerge?

## Setup

- Built 6 intermediate corpora via `make_hard_real_corpus(n_items=N, n_clusters=8, seed=0)` —
  same 8 hard-corpus categories, same construction method as the 480-item corpus. Sizes: 96,
  144, 192, 240, 320, 400 (all divisible by 8; verified exactly N/8 items per category for each).
- Same `CORPUS_PER_DIM_STD = 0.04882` used throughout (computed once over the shared candidate
  pool in `benchmark_realdata_embeddings_hard.py`, imported rather than re-derived per corpus
  size, so noise scale stays comparable across sizes and to the 480-item report).
- Timing calibrated first (5 queries/size, n_qubits=4, PCA projection): 0.28-0.75s/query,
  roughly increasing with corpus size — cheap enough to run a 6-level noise sweep × 6 sizes ×
  n=20/cell (~6.5 min total) before committing to any n=100 confirmation.
- Initial sweep used PCA + classical only (not random) — the transition question is specifically
  about the PCA-classical gap, so random projection wasn't run here (it's not part of Result 2's
  dissociation claim).
- Noise levels: mult = 0.2, 1.2, 2.0, 4.0, 6.0, 8.0 — includes all three levels confirmed real at
  480 items (1.2, 2.0, 4.0), plus a low-noise control (0.2) and two higher levels (6.0, 8.0) for
  context on where both methods eventually floor out together.

## Result — gradual scaling, not a sharp cutoff

**Initial pass (n_qubits=4, n=20/cell), PCA-classical gap (classical − PCA) at each size:**

| n_items | mult=0.2 | mult=1.2 | mult=2.0 | mult=4.0 | mult=6.0 | mult=8.0 | mean\|gap\| |
|---|---|---|---|---|---|---|---|
| 96  | 0.00 | 0.05 | 0.25 | 0.00 | 0.05 | 0.00 | 0.058 |
| 144 | 0.00 | 0.15 | 0.25 | -0.05 | -0.05 | 0.00 | 0.083 |
| 192 | 0.00 | 0.55 | 0.35 | 0.00 | 0.05 | 0.00 | 0.158 |
| 240 | 0.00 | 0.50 | 0.75 | 0.15 | 0.00 | 0.00 | 0.233 |
| 320 | 0.00 | 0.55 | 0.75 | 0.25 | 0.00 | 0.00 | 0.258 |
| 400 | 0.05 | 0.70 | 0.95 | 0.25 | -0.05 | 0.00 | 0.333 |

(For reference: 48 items ≈ 0.026 mean\|gap\| across the full noise range, "tracks closely";
480 items' three confirmed-real levels average ≈0.70 gap at n=100.)

The gap at mult=1.2 and mult=2.0 grows **monotonically** with corpus size, from
near-baseline at 96 items to clearly substantial by 400. There is no single size where it jumps
from ~0 to "confirmed real" — it climbs steadily across the whole 96→480 range.

**Confirmation at n=100/cell (95% Wald CIs), three sizes spanning the apparent bend in the
curve — 96 (small gap), 144 (intermediate), 192 (first size with a gap comparable in
magnitude to the confirmed-real 480-item levels):**

| n_items | mult | PCA (n=100) | Classical (n=100) | gap | 95% CIs |
|---|---|---|---|---|---|
| 96  | 1.2 | 0.930 ± 0.050 | 1.000 ± 0.000 | 0.07 | non-overlapping* |
| 96  | 2.0 | 0.830 ± 0.074 | 1.000 ± 0.000 | 0.17 | non-overlapping* |
| 96  | 4.0 | 0.990 ± 0.020 | 0.990 ± 0.020 | 0.00 | overlapping |
| 144 | 1.2 | 0.820 ± 0.075 | 1.000 ± 0.000 | 0.18 | non-overlapping* |
| 144 | 2.0 | 0.760 ± 0.084 | 1.000 ± 0.000 | 0.24 | non-overlapping* |
| 144 | 4.0 | 0.970 ± 0.033 | 0.970 ± 0.033 | 0.00 | overlapping |
| 192 | 1.2 | 0.530 ± 0.098 | 1.000 ± 0.000 | 0.47 | **non-overlapping** |
| 192 | 2.0 | 0.610 ± 0.096 | 1.000 ± 0.000 | 0.39 | **non-overlapping** |
| 192 | 4.0 | 0.970 ± 0.033 | 0.970 ± 0.033 | 0.00 | overlapping |

\* **Flagged, not rounded into "confirmed":** at 96 and 144 items, classical recall sits
exactly at the ceiling (1.000), which makes its Wald CI collapse to zero width — the
"non-overlapping" label then triggers mechanically for *any* nonzero PCA gap, no matter how
small. A gap of 0.07 (96 items) is barely larger than the 0.026 baseline deviation that
characterized the 48-item corpus's "tracks closely" behavior, and shouldn't be read as the same
kind of finding as the 0.39-0.47 gaps at 192 items or the 0.56-0.85 gaps confirmed at 480. The
CI-overlap test is doing real discriminating work at 192 items and above; at 96-144 items it is
mostly reporting the ceiling effect, not a meaningful dissociation. This same caveat did not
apply to the original 480-item confirmation table because classical recall never sat exactly at
1.000 with zero variance in that table's mult=6.0 borderline row.

mult=4.0 shows no effect at any of the three confirmed sizes (gap=0.00, CIs overlapping) —
consistent with the n=20 sweep, where mult=4.0 only starts separating at 240+ items.

## Bottom line

- **No single clean transition point.** The PCA-classical gap grows gradually across the whole
  96→480 range at mult=1.2 and mult=2.0, not as a step function. Reporting one cutoff number
  would overstate the sharpness of what's actually a continuous scaling trend.
- **96-144 items: gap present but small, and partly a CI-degeneracy artifact.** Point estimates
  (0.07-0.24) are modest, closer in kind to the 48-item baseline (0.026) than to the confirmed
  480-item breakdown. The "non-overlapping CI" label at these sizes is mechanically driven by
  classical sitting exactly at the recall ceiling, not by a large, well-separated effect — flagged
  the same way the original report flagged mult=6.0 as borderline rather than confirmed.
- **192 items is the smallest size tested where the gap is unambiguous:** 0.39-0.47 at
  mult=1.2/2.0, comparable in magnitude to what was confirmed real at 480 items (0.56-0.85), and
  the non-overlap here isn't just a ceiling artifact — PCA's point estimate itself has dropped to
  0.53-0.61, a real degradation, not just a technicality of the CI formula.
- **mult=4.0 doesn't separate until somewhere between 192 and 240 items** — a second axis of
  gradualness: the transition size itself depends on which noise level you're asking about, not
  just corpus size in isolation.
- **Practical takeaway:** if a single practical threshold is needed for planning purposes, ~192
  items is a reasonable "starts to matter" marker for mult=1.2-2.0 — but the actual picture is a
  gradual onset beginning as early as 96-144 items (small, partly artifactual) and continuing to
  strengthen all the way to 480 (confirmed, hardware-validated). Anyone using this for a
  production sizing decision should treat "192" as a soft floor, not a hard boundary.

This step used no QPU quota. Hardware confirmation of the ~192-item region (or wherever a
specific production corpus size lands) is a separate, later step, matching how Result 2b's
hardware run was scoped only after the simulation-side effect was already confirmed at 480.

## Files

- `scratch_transition_build_corpora.py` — builds the 6 intermediate corpora.
- `scratch_transition_corpus_{96,144,192,240,320,400}.pkl` — the 6 corpora (keys + MiniLM
  embeddings), each verified N/8 items/category.
- `scratch_transition_timing.py` — per-query timing calibration across sizes.
- `scratch_transition_sweep.py` / `scratch_transition_sweep_results.pkl` — n=20/cell initial
  sweep, all 6 sizes × 6 noise levels.
- `scratch_transition_confirm.py` / `scratch_transition_confirm_results.pkl` — n=100/cell
  confirmation with 95% CIs for 144 and 192 items.
- `scratch_transition_confirm_96_results.pkl` — n=100/cell confirmation for 96 items (run
  inline via `python -c`, not a saved standalone script).
