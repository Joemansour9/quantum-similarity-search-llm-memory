# Table 4 — Query-noise sweep, real (easy) + synthetic corpora

Feeds the paper's **Table 4** (Section 5.2): Recall@1 across query-noise multipliers
{0.2, 1.2, 3.0, 5.0}, for Random, PCA, and Classical, on both the synthetic corpus and the real
"easy" corpus (n_qubits=8 fixed, the section's default).

## Which file to use

`scratch_pca_leakage_noiseeasy_synthetic_results.pkl` — a dict with top-level keys `'synthetic'`
and `'real_easy'`, each keyed by noise multiplier and then by method. This file is *not* an
extract of a shared dual-mode file like several other tables' sources — the producing script,
`scratch_verify_pca_leakage_noiseeasy_synthetic.py` (also in this folder), computes only the
out-of-sample version directly; there is no leaky branch to strip out, because no leak-free version
of this table existed anywhere in the project until this script was written.

## Fitting convention

Out-of-sample (leak-free): both Random and PCA are fit with the query excluded from the fitting
batch. The Classical column is raw-embedding cosine similarity, which has no projection step and so
is unaffected by the leakage question either way; it was recomputed fresh here too, and the script
cross-checks it against the previously-published Classical numbers as a sanity check
(`classical_cross_check_all_match` field in the pkl).
