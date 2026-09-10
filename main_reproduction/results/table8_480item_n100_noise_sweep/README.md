# Table 8 — 480-item corpus, n_qubits=4, n=100 confirmatory noise sweep

Feeds the paper's **Table 8** (Section 7): Recall@1 for PCA and Classical across query-noise
multipliers {0.2, 1.2, 2.0, 4.0, 6.0}, at n=100 query-runs per cell (5× the project's usual n=20,
run specifically to confirm a pattern that already looked significant at the smaller sample size),
with Wilson 95% confidence intervals.

## Which file to use

Two files, because this table was confirmed in two separate passes:
- `scratch_table8_480n100_lowmult_CLEAN.pkl` — multipliers 0.2, 1.2, 2.0.
- `scratch_table8_480n100_highmult_CLEAN.pkl` — multipliers 4.0, 6.0.

Each has a `'leak_free_only'` key: a dict keyed by noise multiplier, each holding `('pca',)` /
`('random',)` / `'classical'` entries as `(value, confidence_interval_halfwidth)` pairs.

## Fitting convention

Out-of-sample (leak-free): the query is excluded from the fitting batch. Both files were extracted
from dual-mode pkls that originally stored a leaky and a leak-free computation together (producing
scripts `scratch_verify_pca_leakage_480_n100_confirm_lowmult.py` and
`scratch_verify_pca_leakage_480_n100_confirm.py`, both also in this folder); the leaky branch has
been stripped from each extract. The full dual-mode originals live in
`repo/audit_history/before_after_comparisons/` (`scratch_pca_leakage_480_n100_confirm_lowmult_results.pkl`
and `scratch_pca_leakage_480_n100_confirm_results.pkl`).
