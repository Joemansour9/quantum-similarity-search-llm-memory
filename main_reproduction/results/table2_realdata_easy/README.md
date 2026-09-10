# Table 2 — Real (easy) corpus qubit sweep

Feeds the paper's **Table 2** (Section 5.1): Recall@1 vs. n-qubits, Random vs. PCA projection, on
the real 24-item "easy" corpus (8 topically-distinct 20 Newsgroups categories, 3 posts each,
all-MiniLM-L6-v2 embeddings).

## Which file to use

`scratch_table1_table2_table6minilm_CLEAN.pkl` — a Python dict pickle. Load it and read the
`'leak_free_only'` key, then the `'real_easy'` sub-key. That gives you a dict keyed by
`(n_qubits, 'pca')` / `(n_qubits, 'random')` for n_qubits in {4,6,8,10,12,14}, matching Table 2's
Random and PCA columns exactly.

**This same file also backs two other tables** — `table1_synthetic/`'s Table 1 (via its
`'synthetic'` key) and `table6_mpnet_generalisation/`'s MiniLM row of Table 6 (via its
`'real_hard_48'` key). It's the same file, copied into all three folders; only the `'real_easy'`
key is relevant here.

## Fitting convention

Out-of-sample (leak-free): the query vector is excluded from the batch used to fit the PCA
projection. This file was extracted from a dual-mode pkl that originally stored both the leaky and
leak-free computation side by side; the leaky branch has been stripped out entirely. The full
leaky-vs-clean comparison lives in
`repo/audit_history/before_after_comparisons/scratch_pca_leakage_remaining_results.pkl` and its
producing script `scratch_verify_pca_leakage_remaining.py` (also copied into this folder for
reference — it documents the exact fitting call).
