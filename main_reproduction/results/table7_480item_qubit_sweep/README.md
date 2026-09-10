# Table 7 — 480-item corpus qubit sweep

Feeds the paper's **Table 7** (Section 7): Recall@1 vs. n-qubits, Random vs. PCA, on the 480-item
version of the hard corpus, with Wilson 95% confidence intervals. This is also the source for the
**"Fidelity kernel" column of Table 14** (Section 9) — see `table14_and_discussion_kernel_crossover/`
for that cross-reference.

## Which file to use

For the point estimates: `scratch_table7_480qubitsweep_CLEAN.pkl` — read the `'leak_free_only'`
key, a dict keyed by `(n_qubits, 'pca')`/`(n_qubits, 'random')` for n_qubits in {4,6,8,10,12,14}.

For the Wilson confidence intervals printed alongside each cell in the paper's table:
`scratch_wilson_ci_480_qubit_sweep_results.pkl`, produced by `scratch_wilson_ci_480_qubit_sweep.py`
(both in this folder). This script's own comments confirm its input is the `'leak_free'` branch of
`scratch_pca_leakage_results.pkl` — the same computation as the CLEAN file above, just with
confidence intervals added.

## Fitting convention

Out-of-sample (leak-free): the query is excluded from the candidate batch used to fit the PCA
projection. `scratch_table7_480qubitsweep_CLEAN.pkl` was extracted from a dual-mode pkl
(`scratch_verify_pca_leakage.py` → `scratch_pca_leakage_results.pkl`) that originally stored both a
leaky and a leak-free computation together; the leaky branch has been stripped from this extract.
The full dual-mode original lives in
`repo/audit_history/before_after_comparisons/scratch_pca_leakage_results.pkl`.
