# Other Checks

Three supplementary checks, none documented in a markdown report.

## `scratch_three_baselines_qubit_sweep_hard.py` (superseded — see v2 in `main_reproduction/`)
An earlier attempt at the three-extra-classical-baselines comparison (SVD-no-centering, whitened
PCA, first-n-raw-dimensions) behind the paper's Table 13, on the 48-item hard corpus. This script
reconstructed the corpus using `_POOL_PER_CATEGORY = 60`, on the belief (documented in an earlier
version of `main_reproduction/README.md`'s caveats) that pool=60 was the value used to produce the
paper's published qubit-hard numbers. **That belief turned out to be wrong.** Its own saved output
(`scratch_three_baselines_qubit_sweep_hard_results.pkl`) does not match the paper's published Table
3 values — e.g. PCA Recall@1 = 0.80 at n_qubits=4 here, versus the paper's published 0.660; 0.98 vs.
0.920 at n_qubits=6 — confirming the pool=60 reconstruction draws a different, non-matching corpus.
The corrected version, `scratch_three_baselines_qubit_sweep_hard_v2.py`, instead uses pool=65 (the
codebase's actual current default) and reproduces the published table exactly; it lives in
`repo/main_reproduction/results/table3_realdata_hard_and_table13_baselines/` since it is the current
source for Tables 3 and 13, not historical material. This v1 script and its non-matching output are
kept here purely as a record of the wrong assumption and how it was caught.

## `scratch_check_underpowered_mult6_n500.py`
Reruns the original mult=6.0/480-item "genuinely borderline" finding at n=500/cell, using the
original leaky methodology by design, to test whether "borderline" reflects an underpowered sample
(the original used n=100). Result, in `scratch_underpowered_mult6_n500_results.pkl`:
PCA=0.554±0.044, classical=0.600±0.043 — still overlapping. The borderline result persists at five
times the original sample size; it is not a power artifact. This check is independent of the
leakage question — see `../before_after_comparisons/COMPARISON.md` for the effect of removing the
leak from this same cell, which is substantially larger than the sample-size effect tested here.

## `scratch_check_rzz_ablation.py` (superseded — see the out-of-sample rerun in `main_reproduction/`)
Tests whether the RZZ entangling layer contributes to Recall@1, as distinct from raw fidelity,
against a plain product-state (no entangler) encoding. Uses the project's standard leaky
PCA-fitting convention by design, since this question is independent of the leakage issue. A later
out-of-sample rerun, `scratch_verify_rzz_ablation_oos.py`, was done to close a gap with the paper's
blanket out-of-sample claim; it is the *current* source for the Methods §3.1 entangling-ablation
prose and lives in `repo/main_reproduction/results/methods_rzz_entangling_ablation/`, not here. Its
qualitative conclusion is meaningfully different from the leaky result below — see that folder's
README for the corrected numbers.

**Status as of 2026-09-01:** the script was rewritten — extended from a single fixed n_qubits=4 to
sweep both n_qubits=4 and n_qubits=8 — and run, with output saved
(`scratch_rzz_ablation_result.txt`, `scratch_rzz_ablation_results.pkl`). The version and results in
this folder reflect that run. Its filename ("Check 2") implies a "Check 1" that does not exist in
this project.

**Result: the entangling layer's contribution is noise-level-dependent, not uniform.** At low noise
(mult=1.2), RZZ underperforms the plain product-state (Ry-only) encoding — Recall@1 0.880 vs 1.000
at n_qubits=4, 0.920 vs 1.000 at n_qubits=8: the entangling layer costs 0.08–0.12 Recall@1 at this
noise level rather than adding to it. At mult=4.0 and above, the two encodings are identical or
within a few points of each other at every level tested, both qubit counts: the entangling layer
contributes nothing distinguishable to retrieval quality once noise is moderate to high. At the
lowest noise level tested (mult=0.2), both encodings are already at the 1.000 ceiling, so neither
construction is distinguishable there either. Across this sweep, RZZ's entangling layer is not
shown to improve Recall@1 at any tested condition, and measurably reduces it at mult=1.2.
