# Table 3 (real-hard qubit sweep) and Table 13 (fairer classical baselines)

This is the paper's **headline-result folder**. It feeds two tables:
- **Table 3** (Section 5.1): Recall@1 vs. n-qubits, Random vs. PCA projection, on the real 48-item
  "hard" corpus (8 deliberately confusable 20 Newsgroups categories, 6 posts each).
- **Table 13** (Section 9 Discussion): the same corpus and qubit sweep, extended with three further
  classical projection baselines — truncated SVD without centering, whitened PCA, and the first-n
  raw embedding dimensions — used to test whether PCA's advantage is attributable to identifying
  dominant variance directions specifically, or to some other property of PCA.

## Which file to use

`scratch_three_baselines_qubit_sweep_hard_v2_results.pkl` — the single result file for both tables.
It's a dict keyed by `(n_qubits, method)` for method in `{'random', 'pca', 'svd', 'whiten',
'firstn'}` and n_qubits in {4,6,8,10,12,14}. Table 3 uses just the `'random'`/`'pca'` columns;
Table 13 uses all five. The producing script, `scratch_three_baselines_qubit_sweep_hard_v2.py`, is
in this folder too — its docstring explains the corpus reconstruction in detail (see below).

## Fitting convention

Out-of-sample (leak-free) throughout — the script fits every method's projection on the candidate
batch alone, with the query excluded, for both the PCA/random cross-check and the three new
baselines.

## A corpus-reconstruction detail worth knowing

`benchmark_realdata_embeddings_hard.py`'s `_POOL_PER_CATEGORY` parameter controls how many
documents-per-topic the hard corpus is drawn from before subsampling to 6/topic. Its current value
is **65**. This script uses that current value, and its results match the paper's published Table 3
numbers exactly (validated by the script itself before it computes anything new). An earlier
attempt at this same three-baselines comparison assumed the *published* numbers came from an older
`_POOL_PER_CATEGORY = 60`, reconstructed that older corpus, and got numbers that did **not** match
the published table (e.g. PCA Recall@1 = 0.80 at n_qubits=4, vs. the published 0.660). That earlier,
non-matching attempt is archived — with its own output showing the mismatch — at
`repo/audit_history/investigation/other_checks/scratch_three_baselines_qubit_sweep_hard.py`
("v1" of this script), kept there as a record of the wrong assumption and how it was caught.
