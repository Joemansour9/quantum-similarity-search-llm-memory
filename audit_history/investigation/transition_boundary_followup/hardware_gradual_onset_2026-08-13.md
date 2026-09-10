# Does the Corpus-Scale Breakdown's Onset Show Up as Gradual on Hardware Too? — 2026-08-13

Real IBM Quantum hardware (`ibm_kingston`). Follow-up to
`corpus_size_transition_2026-08-03.md`, which found (in simulation) that the PCA-classical
query-noise dissociation doesn't have a sharp corpus-size cutoff — it grows gradually from a
small, partly-artifactual gap at 96-144 items to a clearly substantial one by 192 items,
continuing to strengthen through 480. This document tests whether that gradualness is also
visible on real hardware, using the same wrong-top-1 reproduction design as
`hardware_confirmation_480item_2026-08-12.md`, now run on the 192-item corpus
(`scratch_transition_corpus_192.pkl`, reused unchanged) at all four noise levels already
established at 480 items (mult=1.2, 2.0, 4.0, 6.0), n=12 each — a fully symmetric
2-corpus-size × 4-noise-level comparison. Testing ran in two passes: mult=1.2/2.0 first, then
mult=4.0/6.0 to close the asymmetry left by only testing two levels.

## Setup

- Quota confirmed before starting: 178s (matched expectation exactly against the portal).
- Same design throughout: destructive circuit, n_qubits=4, PCA projection, ibm_kingston,
  queries selected as cases where local exact-statevector simulation already produced the wrong
  top-1, each scoped to true match + top-3 simulator competitors (4 candidates/query), n=12,
  1024 shots, batched one job per noise level.
- 192-item corpus wrong-top-1 rates were noticeably lower than at 480 items, meaning more
  queries had to be scanned to fill n=12: 24 scanned for mult=1.2 (vs 480-item's 16), 33 scanned
  for mult=2.0 (vs 480-item's — mult=2.0 wasn't originally scanned fresh at 480 items, but its
  n=100 simulation recall of 0.15 implies a much higher wrong-rate there than 192's implied
  ~36-40%). This by itself is a simulation-side echo of the gradual-onset finding, now visible in
  how *hard it is to even find* failures at the smaller size.

## Result

**192-item corpus, mult=1.2, n=12** (job `d9ufha343mgs73estpa0`, gate counts 15/23/31 mixed —
clipping-driven simplification, same pattern seen at higher noise on the 480-item batches):

| q_idx | sim wrong pick | hw pick | true match rank (hw) | same wrong answer? |
|---|---|---|---|---|
| 0 | item_15_c0 | item_0_c0 | **1/4 (hw fixed it)** | No |
| 1 | item_13_c0 | item_13_c0 | 4/4 | **Yes** |
| 4 | item_1_c0 | item_1_c0 | 4/4 | **Yes** |
| 8 | item_9_c0 | item_9_c0 | 4/4 | **Yes** |
| 10 | item_58_c2 | item_91_c3 | 3/4 | No |
| 12 | item_106_c4 | item_106_c4 | 3/4 | **Yes** |
| 14 | item_18_c0 | item_18_c0 | 2/4 | **Yes** |
| 15 | item_69_c2 | item_69_c2 | 4/4 | **Yes** |
| 16 | item_5_c0 | item_5_c0 | 4/4 | **Yes** |
| 18 | item_90_c3 | item_90_c3 | 2/4 | **Yes** |
| 19 | item_3_c0 | item_19_c0 | **1/4 (hw fixed it)** | No |
| 23 | item_110_c4 | item_108_c4 | 4/4 | No |

**10/12 reproduced, 2/12 hardware fixed it, 8/12 same wrong candidate as simulation.**

**192-item corpus, mult=2.0, n=12** (job `d9ufjngu5hac73ah5phg`):

| q_idx | sim wrong pick | hw pick | true match rank (hw) | same wrong answer? |
|---|---|---|---|---|
| 2 | item_51_c2 | item_2_c0 | **1/4 (hw fixed it)** | No |
| 3 | item_13_c0 | item_176_c7 | 3/4 | No |
| 6 | item_9_c0 | item_6_c0 | **1/4 (hw fixed it)** | No |
| 9 | item_6_c0 | item_13_c0 | 3/4 | No |
| 13 | item_168_c7 | item_13_c0 | **1/4 (hw fixed it)** | No |
| 16 | item_106_c4 | item_106_c4 | 4/4 | **Yes** |
| 18 | item_57_c2 | item_50_c2 | 4/4 | No |
| 24 | item_61_c2 | item_61_c2 | 3/4 | **Yes** |
| 25 | item_112_c4 | item_110_c4 | 4/4 | No |
| 30 | item_65_c2 | item_65_c2 | 4/4 | **Yes** |
| 31 | item_59_c2 | item_59_c2 | 4/4 | **Yes** |
| 32 | item_36_c1 | item_36_c1 | 4/4 | **Yes** |

**9/12 reproduced, 3/12 hardware fixed it, 5/12 same wrong candidate as simulation.**

**192-item corpus, mult=4.0, n=12** (job `d9ufpbgu5hac73ah61p0`, gate counts 15/31 mixed). Note
on selection: this level's wrong-top-1 rate on the 192-item corpus is genuinely very low (~3%,
consistent with `corpus_size_transition_2026-08-03.md`'s finding that mult=4.0 shows no gap at
192 items in simulation — PCA and classical recall were statistically indistinguishable there,
0.970 vs 0.970 at n=100). Reaching n=12 required scanning 347 queries instead of the usual
15-40, and all four candidates' fidelities in the table below sit tightly clustered near the
noise floor (typically within ~0.05-0.10 of each other) — these are essentially rare coin-flip
misclassifications between two otherwise-equal methods, not clean breakdowns:

| q_idx | sim wrong pick | hw pick | true match rank (hw) | same wrong answer? |
|---|---|---|---|---|
| 4 | item_142_c5 | item_4_c0 | **1/4 (hw fixed it)** | No |
| 31 | item_92_c3 | item_71_c2 | 2/4 | No |
| 64 | item_80_c3 | item_80_c3 | 3/4 | **Yes** |
| 147 | item_149_c6 | item_149_c6 | 3/4 | **Yes** |
| 172 | item_40_c1 | item_122_c5 | 3/4 | No |
| 175 | item_180_c7 | item_34_c1 | 4/4 | No |
| 196 | item_121_c5 | item_121_c5 | 4/4 | **Yes** |
| 229 | item_165_c6 | item_165_c6 | 2/4 | **Yes** |
| 240 | item_157_c6 | item_24_c1 | 3/4 | No |
| 268 | item_151_c6 | item_125_c5 | 4/4 | No |
| 295 | item_65_c2 | item_103_c4 | **1/4 (hw fixed it)** | No |
| 346 | item_99_c4 | item_10_c0 | 2/4 | No |

**10/12 reproduced, 2/12 hardware fixed it, 4/12 same wrong candidate as simulation.**

**192-item corpus, mult=6.0, n=12** (job `d9ufqc535hes73fjsoe0`, gate count uniformly 31 —
back to normal since this level's wrong-top-1 rate is not unusually low, 12 found in 43 scanned):

| q_idx | sim wrong pick | hw pick | true match rank (hw) | same wrong answer? |
|---|---|---|---|---|
| 0 | item_101_c4 | item_101_c4 | 3/4 | **Yes** |
| 2 | item_64_c2 | item_64_c2 | 2/4 | **Yes** |
| 4 | item_142_c5 | item_45_c1 | 3/4 | No |
| 16 | item_72_c3 | item_16_c0 | **1/4 (hw fixed it)** | No |
| 21 | item_13_c0 | item_189_c7 | 3/4 | No |
| 22 | item_98_c4 | item_4_c0 | 4/4 | No |
| 24 | item_136_c5 | item_70_c2 | 4/4 | No |
| 25 | item_41_c1 | item_109_c4 | 4/4 | No |
| 31 | item_92_c3 | item_92_c3 | 2/4 | **Yes** |
| 33 | item_37_c1 | item_189_c7 | 2/4 | No |
| 38 | item_153_c6 | item_8_c0 | 3/4 | No |
| 42 | item_93_c3 | item_42_c1 | **1/4 (hw fixed it)** | No |

**10/12 reproduced, 2/12 hardware fixed it, 3/12 same wrong candidate as simulation.**

## The full 2×4 picture

| corpus size | mult=1.2 | mult=2.0 | mult=4.0 | mult=6.0 |
|---|---|---|---|---|
| 48 | *(not directly comparable — see note)* | *(n/c)* | *(n/c)* | *(n/c)* |
| 192 | 10/12 | 9/12 | 10/12 | 10/12 |
| 480 | 12/12 | 12/12 | 10/12 | 10/12 |

**Note on the 48-item point:** it isn't included as a matched wrong-top-1 test, because that
design doesn't really apply there — at 48 items, mult=1.2, PCA-quantum recall in simulation is
already 0.88-1.00 (see `pca_vector_classical_baseline_2026-08-04.md`'s qubit-sweep table), so
there are barely any wrong-top-1 cases to select in the first place. The closest existing
hardware evidence at 48 items is `hardware_validation_2026-07-30.md` Test 14's
ranking-preservation check: **12/12 top-1 agreement between hardware and simulator** across a
general (not wrong-top-1-selected) n=12 sample — consistent with "works," but answering a
slightly different question (general agreement vs. does-hardware-reproduce-a-known-failure).
Flagged rather than glossed over: the three points aren't perfectly matched-design, only the
192-and-480 points are.

**mult=1.2 and mult=2.0: hardware reproduction is real but measurably weaker at 192 than at 480
(10/12 vs 12/12; 9/12 vs 12/12).** This is consistent with — not a hardware-specific new finding
beyond — `corpus_size_transition_2026-08-03.md`'s simulation conclusion that 192 items is a
"soft floor," not a sharp cutoff: the breakdown is present and already substantial there, but
measurably less unanimous than at 480, both in how often hardware reproduces it (2-3/12 flips at
192 vs 0/12 at 480) and in same-wrong-candidate agreement (8/12 and 5/12 at 192, vs 480's 7/12
and 6/12).

**mult=4.0 and mult=6.0 tell a different, and genuinely informative, story: 192 and 480 landed
at the *same* reproduction rate (10/12 both), but for opposite reasons.** At mult=6.0, both
corpus sizes show a real, substantial breakdown in simulation (192: PCA≈0.56-ish rare-failure
regime per the earlier confirmation; 480: PCA 0.560±0.097 vs classical 0.650±0.093, "borderline"
but real) — so 10/12 at both sizes reflects a genuinely comparable effect holding up similarly
at both scales. At mult=4.0, by contrast, 192 items shows **no effect at all** in simulation
(PCA and classical both 0.970±0.033, statistically indistinguishable) — the wrong-top-1 cases
found there are rare coin-flip misclassifications between two equally-good methods (wrong-top-1
rate ~3%, fidelities tightly clustered near the noise floor), not the same phenomenon as 480's
real 0.56-gap breakdown (wrong-top-1 rate closer to 42%, per the earlier confirmation). **The
matching 10/12 rate at mult=4.0 is a coincidence of two different underlying regimes, not
evidence that mult=4.0's onset is close to 192 items** — the simulation-side finding that
mult=4.0 only starts separating somewhere between 192 and 240 items stands, and this hardware
result doesn't move that boundary. Flagged explicitly so the matching numbers in the 2×4 table
aren't misread as "no scale dependence at mult=4.0."

**Bottom line: the gradual-onset picture from simulation holds up on real hardware where the
effect is genuinely present at both sizes (mult=1.2, 2.0, 6.0) — weaker at 192, stronger at
480, same direction throughout. At mult=4.0, hardware isn't testing the same phenomenon at both
sizes at all, since simulation itself found no effect at 192 for that noise level — the equal
10/12 rate there is not a counterexample to gradualness, it's a different question with a
coincidentally similar answer.**

## Quota

Started this task at 178s (confirmed exactly). First pass (mult=1.2, mult=2.0) consumed 30s;
second pass (mult=4.0, mult=6.0) consumed a further 30s (15s + 15s, despite mult=4.0 needing an
8x-larger scan to find candidates — that scan cost is local simulation, not hardware quota).
**118s remaining at completion.** Comfortably enough for further batches if there's a next
priority — this document's own scope is now complete (all 4 levels × 2 corpus sizes done).

## Files

- `scratch_hw_build_192_mult1.2.py` / `scratch_hw_192_mult1.2_circuits.pkl` — selection + circuit build.
- `scratch_hw_submit_192_mult1.2.py` / `scratch_hw_192_mult1.2_results.pkl` — hardware submission + results.
- `scratch_hw_build_192_mult2.0.py` / `scratch_hw_192_mult2.0_circuits.pkl` — selection + circuit build.
- `scratch_hw_submit_192_mult2.0.py` / `scratch_hw_192_mult2.0_results.pkl` — hardware submission + results.
- `scratch_hw_build_192_mult4.0.py` / `scratch_hw_192_mult4.0_circuits.pkl` — selection + circuit build (scan cap raised to 500, needed 347 to reach n=12).
- `scratch_hw_submit_192_mult4.0.py` / `scratch_hw_192_mult4.0_results.pkl` — hardware submission + results.
- `scratch_hw_build_192_mult6.0.py` / `scratch_hw_192_mult6.0_circuits.pkl` — selection + circuit build.
- `scratch_hw_submit_192_mult6.0.py` / `scratch_hw_192_mult6.0_results.pkl` — hardware submission + results.
- `scratch_transition_corpus_192.pkl` — reused unchanged (192-item corpus, seed=0, from `corpus_size_transition_2026-08-03.md`).
