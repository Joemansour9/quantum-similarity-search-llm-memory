# Table 5 — Real (hard) corpus, n-qubits=8, extended query-noise sweep

Feeds the paper's **Table 5** (Section 5.2): Recall@1 across the full query-noise sweep
(multipliers 0.2, 1.2, 4.0, 6.0, 8.0, 10.0, 16.0, 20.0) for Random, PCA, and Classical, on the
48-item hard corpus at n_qubits=8. This is also the source for Figure 3 (the plot of this same
sweep) and is referenced again in Section 6's mean-deviation-ratio prose for the MiniLM embedding
model.

## Which file to use

`scratch_table5_minilm_noisesweep_nq8_CLEAN.pkl` — load it and read the `'leak_free_only'` key, a
dict keyed by noise multiplier, each holding `('pca',)` / `('random',)` / `'classical'` entries.
Table 5 uses only 8 of the 10 multipliers this file actually covers (it also has 2.0 and 12.0,
computed but not printed in the paper's table).

## Fitting convention

Out-of-sample (leak-free): extracted from a dual-mode pkl that originally computed both the leaky
and leak-free versions side by side for direct comparison. The leaky branch has been stripped from
this extract. The producing script, `scratch_verify_pca_leakage_minilm_noisesweep_nq8.py` (also in
this folder), shows the exact fitting call: `fit_batch = candidate_batch` (query excluded) for the
leak-free branch. The full leaky-vs-clean comparison lives in
`repo/audit_history/before_after_comparisons/scratch_pca_leakage_minilm_noisesweep_nq8_results.pkl`.
