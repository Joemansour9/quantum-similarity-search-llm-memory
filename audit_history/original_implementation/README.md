# Original Implementation

The four core library files, retained here as the "before" side of this correction history.

**Note on scope.** These files are identical, byte-for-byte, to the copies in
`repo/main_reproduction/pipeline/`. This is not an inconsistency: the PCA query-leakage issue
documented in `../investigation/` and quantified in `../before_after_comparisons/` was never
patched into this code. Exactly one version of `agent_memory.py`, `q_encoder.py`, `q_graph.py`, and
`q_search.py` exists in this project; it is simultaneously the original, unpatched implementation
and the current, in-use implementation. This copy exists for labeling and provenance — to fix what
"original" means as of this correction history — not to preserve a separate codebase.

One file is deliberately absent from this folder: `benchmark_realdata_embeddings_hard.py`. Its
`_POOL_PER_CATEGORY` constant was changed in place (60 → 65) partway through the project, and no
version control exists on this disk to recover the pre-change file. The original implementation of
that file no longer exists as a file; only its frozen output survives, in
`repo/main_reproduction/results/table3_realdata_hard_and_table13_baselines/` (renamed from
`section5_1_realdata_hard/`; the old name survives only in the
`stale_main_reproduction_results_2026-09-08/` archive alongside it in this folder). The current, pool=65 version is kept
here as `benchmark_realdata_embeddings_hard_CURRENT_pool65.py`, named to distinguish it from the
version that produced the numbers in that results folder.

The leak itself is located in `agent_memory.py`'s `query()` method, at the
`full_batch = np.vstack([...])` line. `q_encoder.py`'s and `agent_memory.py`'s own docstrings
describe what the projection refit does and does not depend on, but do not note that the batch used
for the fit includes the query being evaluated.
