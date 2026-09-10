# Table 9 — PCA-vs-classical gap across intermediate corpus sizes

Feeds the paper's **Table 9** (Section 7.1): the PCA-vs-classical Recall@1 gap at n_qubits=4, noise
multipliers 1.2 and 2.0, across six intermediate corpus sizes (96, 144, 192, 240, 320, 400
documents) bridging the 48-item and 480-item corpora used elsewhere. This is also the source for
**Figure 5** (the plot of this same gap-vs-corpus-size relationship).

## Which file to use

`scratch_table9_transition_sizes_CLEAN.pkl` — read the `'leak_free_only'` key, a dict keyed by
`(n_items, multiplier)` (e.g. `(96, 1.2)`), each holding `('pca',)` / `('random',)` / `'classical'`
entries as `(value, confidence_interval_halfwidth)` pairs. Classical sits at 1.000 in every cell —
raw-embedding cosine retrieval is unaffected by corpus size at this noise level.

For the Wilson confidence intervals printed in the paper's table: `scratch_wilson_ci_transition_sizes_results.pkl`,
produced by `scratch_wilson_ci_transition_sizes.py` (both in this folder).

## Fitting convention

Out-of-sample (leak-free): the query is excluded from the fitting batch at every corpus size. This
file was extracted from a dual-mode pkl (`scratch_verify_pca_leakage_transition_sizes.py` →
`scratch_pca_leakage_transition_sizes_results.pkl`) that originally stored both a leaky and a
leak-free computation side by side; the leaky branch has been stripped from this extract. The full
dual-mode original lives in
`repo/audit_history/before_after_comparisons/scratch_pca_leakage_transition_sizes_results.pkl`.
