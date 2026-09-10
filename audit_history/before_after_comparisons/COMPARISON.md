<title>Leaky vs Leak-Free — Every Comparison, Side by Side</title>

# Leaky vs. leak-free — every comparison this project ever ran

"Leaky" = the actual, currently-live methodology (`agent_memory.py`'s `query()` fits PCA on a batch
that includes the query itself). "Leak-free" = the query held out of the PCA fit, projected through
the resulting fixed directions afterward. Nine result files feed this document; **seven of the nine
had never been rendered into a table anywhere before this document.** Source `.pkl` files sit
alongside this one in this folder; the scripts that produced them are in
`../investigation/leakage_investigation/`.

## 1. 480-item corpus, qubit sweep, mult=1.2 (`scratch_pca_leakage_results.pkl`)
*Written up in `co_author_feedback_verification_2026-08-14.md`, Task 2.*

| n_qubits | PCA leaky | PCA leak-free | PCA Δ | random leaky | random leak-free | gap leaky | gap leak-free | **gap Δ** |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.250 | 0.250 | 0.000 | 0.050 | 0.050 | 0.200 | 0.200 | 0.000 |
| 6 | 0.550 | 0.400 | **−0.150** | 0.000 | 0.000 | 0.550 | 0.400 | **−0.150** |
| 8 | 0.800 | 0.750 | −0.050 | 0.050 | 0.100 | 0.750 | 0.650 | **−0.100** |
| 10 | 0.900 | 0.850 | −0.050 | 0.150 | 0.150 | 0.750 | 0.700 | **−0.050** |
| 12 | 0.950 | 0.950 | 0.000 | 0.200 | 0.200 | 0.750 | 0.750 | 0.000 |
| 14 | 0.900 | 0.900 | 0.000 | 0.350 | 0.350 | 0.550 | 0.550 | 0.000 |

At this specific corpus/noise level, n_qubits=4 (the headline setting) appears unaffected. **This
does not hold at other corpora or noise levels — see tables 3–7 below.**

## 2. Three other corpora, qubit sweep (`scratch_pca_leakage_remaining_results.pkl`)
*Documented in `co_author_feedback_verification_2026-08-14.md`'s same-day extension.*

**Synthetic corpus (Table 1, 24 items):**

| n_qubits | PCA leaky | PCA leak-free | PCA Δ | gap leaky | gap leak-free | gap Δ |
|---|---|---|---|---|---|---|
| 4 | 0.640 | 0.560 | **−0.080** | 0.140 | 0.040 | **−0.100** |
| 6 | 0.800 | 0.860 | +0.060 | 0.180 | 0.200 | +0.020 |
| 8 | 0.960 | 0.980 | +0.020 | 0.280 | 0.300 | +0.020 |
| 10 | 0.980 | 0.960 | −0.020 | 0.220 | 0.180 | −0.040 |
| 12 | 1.000 | 0.960 | −0.040 | 0.160 | 0.100 | −0.060 |
| 14 | 1.000 | 0.960 | −0.040 | 0.160 | 0.100 | −0.060 |

**Real-easy corpus (Table 2, 24 items):** PCA sits at the 1.000 recall ceiling in both conditions
at every qubit count — no room for the leak to show up either way; uninformative, not evidence of
"unaffected."

**Real-hard 48-item corpus (Section 5.1's own numbers):**

| n_qubits | PCA leaky | PCA leak-free | PCA Δ | gap leaky | gap leak-free | gap Δ |
|---|---|---|---|---|---|---|
| **4** | **0.880** | **0.660** | **−0.220** | **0.720** | **0.480** | **−0.240** |
| 6 | 0.880 | 0.920 | +0.040 | 0.720 | 0.800 | +0.080 |
| 8 | 0.920 | 0.940 | +0.020 | 0.640 | 0.720 | +0.080 |
| 10 | 0.980 | 1.000 | +0.020 | 0.640 | 0.620 | −0.020 |
| 12 | 0.980 | 1.000 | +0.020 | 0.640 | 0.620 | −0.020 |
| 14 | 1.000 | 1.000 | 0.000 | 0.660 | 0.580 | −0.080 |

**n_qubits=4 on THIS corpus is the largest leakage effect found in the entire Aug-14 investigation**
(−0.220 absolute, ~4.8 standard errors) — the opposite of what table 1 above suggested about
n_qubits=4 in general.

## 3. mpnet embedding model, 48-item corpus — not previously documented (`scratch_pca_leakage_mpnet*.pkl`)

**Qubit sweep, mult=1.2:**

| n_qubits | PCA leaky | PCA leak-free | PCA Δ | random leaky | random leak-free |
|---|---|---|---|---|---|
| 4 | 0.82 | 0.96 | **+0.14** | 0.16 | 0.16 |
| 6 | 0.84 | 0.98 | +0.14 | 0.20 | 0.22 |
| 8 | 0.88 | 0.98 | +0.10 | 0.24 | 0.28 |
| 10 | 0.90 | 1.00 | +0.10 | 0.26 | 0.28 |
| 12 | 0.92 | 1.00 | +0.08 | 0.32 | 0.34 |
| 14 | 0.94 | 1.00 | +0.06 | 0.36 | 0.38 |

At mult=1.2 specifically, removing the leak *increases* mpnet's PCA recall — consistent with the
small-N "mixed-sign" pattern already noted for the 48-item MiniLM corpus. **This reverses at
higher noise — see the noise sweep below.**

**Noise sweep, n_qubits=4 (all 10 multipliers):**

| mult | classical | PCA leaky | PCA leak-free | **PCA Δ** | random leaky | random leak-free |
|---|---|---|---|---|---|---|
| 0.2 | 1.00 | 1.00 | 1.00 | 0.00 | 0.90 | 0.90 |
| 1.2 | 1.00 | 0.82 | 0.96 | +0.14 | 0.16 | 0.16 |
| 2.0 | 1.00 | 0.98 | 0.72 | **−0.26** | 0.02 | 0.06 |
| 4.0 | 1.00 | 1.00 | 0.32 | **−0.68** | 0.02 | 0.00 |
| 6.0 | 1.00 | 1.00 | 0.20 | **−0.80** | 0.00 | 0.00 |
| 8.0 | 0.88 | 0.86 | 0.12 | **−0.74** | 0.02 | 0.00 |
| 10.0 | 0.64 | 0.66 | 0.08 | **−0.58** | 0.02 | 0.00 |
| 12.0 | 0.48 | 0.50 | 0.06 | **−0.44** | 0.02 | 0.00 |
| 16.0 | 0.36 | 0.36 | 0.02 | −0.34 | 0.02 | 0.00 |
| 20.0 | 0.30 | 0.30 | 0.02 | −0.28 | 0.00 | 0.02 |

**The leak's effect is substantially larger at mult≥4.0 than at mult=1.2** — the same pattern
observed in the MiniLM-embedded corpus, now confirmed in a different embedding model.

## 4. MiniLM 48-item corpus, n_qubits=8, full noise sweep — not previously documented
(`scratch_pca_leakage_minilm_noisesweep_nq8_results.pkl`)

| mult | classical | PCA leaky | PCA leak-free | **PCA Δ** |
|---|---|---|---|---|
| 0.2 | 1.00 | 1.00 | 1.00 | 0.00 |
| 1.2 | 1.00 | 0.92 | 0.94 | +0.02 |
| 2.0 | 1.00 | 1.00 | 0.82 | **−0.18** |
| 4.0 | 0.96 | 0.96 | 0.34 | **−0.62** |
| 6.0 | 0.86 | 0.86 | 0.20 | **−0.66** |
| 8.0 | 0.58 | 0.62 | 0.16 | **−0.46** |
| 10.0 | 0.46 | 0.46 | 0.14 | −0.32 |
| 12.0 | 0.34 | 0.36 | 0.14 | −0.22 |
| 16.0 | 0.22 | 0.18 | 0.08 | −0.10 |
| 20.0 | 0.16 | 0.16 | 0.06 | −0.10 |

Same shape as table 3, different corpus/qubit count: small or positive effect at low noise, large
negative effect from mult=4.0 onward.

## 5. 480-item corpus, query-noise sweep, n_qubits=4, n=20/cell — not previously documented
(`scratch_pca_leakage_480_noisesweep_results.pkl`)

| mult | classical | PCA leaky | PCA leak-free | **PCA Δ** |
|---|---|---|---|---|
| 0.2 | 1.00 | 1.00 | 1.00 | 0.00 |
| 1.2 | 1.00 | 0.25 | 0.25 | 0.00 |
| 2.0 | 1.00 | 0.15 | 0.05 | **−0.10** |
| 4.0 | 0.95 | 0.45 | 0.00 | **−0.45** |
| 6.0 | 0.55 | 0.40 | 0.00 | **−0.40** |
| 8.0 | 0.15 | 0.15 | 0.00 | −0.15 |
| 10.0+ | ≤0.05 | ≤0.05 | 0.00 | ≤−0.05 |

This is the exact corpus/qubit count behind the paper's own Result 2, and the exact noise levels
Result 2's headline table reports. **mult=4.0 and mult=6.0 are the two levels the paper actually
cites (0.42 and 0.56) — table 6 below reruns them at publication-grade n=100 to confirm this n=20
pattern precisely.**

## 6. 480-item corpus, mult=4.0 & mult=6.0, n=100/cell — publication-grade rerun, not previously documented
(`scratch_pca_leakage_480_n100_confirm_results.pkl`) — reruns Result 2's two headline noise levels at publication-grade sample size

| mult | | classical | PCA leaky | PCA leak-free | random leaky | random leak-free |
|---|---|---|---|---|---|---|
| **4.0** | recall | 0.98 ± 0.027 | **0.42 ± 0.097** (published) | **0.03 ± 0.033** | 0.00 ± 0.00 | 0.01 ± 0.020 |
| **6.0** | recall | 0.65 ± 0.093 | **0.56 ± 0.097** (published, "borderline") | **0.00 ± 0.00** | 0.00 ± 0.00 | 0.01 ± 0.020 |

Both leaky rows reproduce the published Result 2 numbers exactly (0.42, 0.56), confirming this is a
clean rerun, not a different setup. Leak-free:

- **mult=4.0: PCA recall drops from 0.42 to 0.03** — a 14x reduction. Gap vs. classical widens from
  0.56 (leaky) to **0.95** (leak-free) — both non-overlapping, so the qualitative "PCA lags
  classical" finding survives and strengthens, but the *magnitude* originally published overstates
  how well PCA does by more than an order of magnitude.
- **mult=6.0: PCA recall drops from 0.56 to 0.00.** The published gap (0.09, overlapping CIs,
  "genuinely borderline") becomes a leak-free gap of **0.65, cleanly non-overlapping (0.65±0.093 vs
  0.00±0.00)** — the single most decisive result of the four noise levels once corrected, not the
  most doubtful one. The "borderline" label in the published Result 2 table is an artifact of the
  leaky methodology at this specific noise level, not a property of the underlying effect.

## 7. Six transition-boundary corpus sizes, mult=1.2 & 2.0 — not previously documented
(`scratch_pca_leakage_transition_sizes_results.pkl`)

| size | mult | classical | PCA leaky | PCA leak-free | **PCA Δ** |
|---|---|---|---|---|---|
| 96 | 1.2 | 1.00 | 0.95 | 0.60 | **−0.35** |
| 96 | 2.0 | 1.00 | 0.75 | 0.30 | **−0.45** |
| 144 | 1.2 | 1.00 | 0.85 | 0.60 | **−0.25** |
| 144 | 2.0 | 1.00 | 0.75 | 0.25 | **−0.50** |
| 192 | 1.2 | 1.00 | 0.45 | 0.45 | 0.00 |
| 192 | 2.0 | 1.00 | 0.65 | 0.20 | **−0.45** |
| 240 | 1.2 | 1.00 | 0.50 | 0.30 | **−0.20** |
| 240 | 2.0 | 1.00 | 0.25 | 0.15 | −0.10 |
| 320 | 1.2 | 1.00 | 0.45 | 0.25 | **−0.20** |
| 320 | 2.0 | 1.00 | 0.25 | 0.10 | −0.15 |
| 400 | 1.2 | 1.00 | 0.30 | 0.15 | −0.15 |
| 400 | 2.0 | 1.00 | 0.05 | 0.15 | +0.10 |

Leakage inflates PCA recall at nearly every one of the twelve (size × mult) cells the
transition-boundary study relies on for its "gradual, not sharp" curve — meaning that entire curve
(and its real-hardware confirmation in `../investigation/transition_boundary_followup/`) is drawn
through leaky points throughout, never rechecked leak-free as a whole curve.

## 8–9. Real-hardware circuit-level angle-shift traces (not recall numbers)
(`scratch_hw_circuit_leakage_results.pkl`, `scratch_n18_leakage_results.pkl`)

Rather than recall deltas, these measure how much the actual Ry rotation angles sent to real IBM
hardware shift between the leaky and leak-free PCA fit, for specific already-submitted hardware
jobs (see `repo/hardware/README.md`'s own cross-reference section for which jobs).

| trace | pairs checked | leaked | basis-unstable (PCA picked different top-k directions, not just different angles) | mean of mean-shift | largest single shift |
|---|---|---|---|---|---|
| Trace 1 (Tests 1, 10, 11, Priority-1/n5) | 25 | 25/25 | 25/25 | ~1.0–1.5 rad | 2.41 rad (76.8% of π) |
| Trace 2 (n=18 gate-matched set) | 18 | 18/18 | 18/18 | 1.013 rad | 2.360 rad (75.1% of π) |

Every single PCA-fitted circuit ever submitted to real IBM Quantum hardware in this project's
original validation round was leaky, and in every case the leak didn't just nudge the rotation
angles slightly — it changed *which* principal directions PCA picked as its top components. This is
the hardware-side counterpart to the recall-number tables above: the same bug that inflates
simulated recall also means every real-hardware circuit in `repo/hardware/` was built from a
PCA basis that wouldn't have been chosen if the query had been properly excluded from the fit.
