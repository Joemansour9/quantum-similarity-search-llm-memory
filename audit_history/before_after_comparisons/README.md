# Before/After Comparisons

See `COMPARISON.md` for every file in this folder rendered as readable tables, with the pattern
across all of them stated explicitly: the leak's effect grows with query-noise strength — small or
absent at mult=1.2, severe at mult≥4.0, in every corpus and embedding model tested.

## Files in this folder

| File | Contents |
|---|---|
| `scratch_pca_leakage_results.pkl` | 480-item corpus, qubit sweep, mult=1.2. |
| `scratch_pca_leakage_remaining_results.pkl` | Synthetic / real-easy / real-hard-48 corpora, qubit sweep. |
| `scratch_pca_leakage_mpnet_results.pkl` | mpnet 48-item corpus, single point (n_qubits=4, mult=1.2). |
| `scratch_pca_leakage_mpnet_qubitsweep_results.pkl` | mpnet 48-item corpus, full qubit sweep. |
| `scratch_pca_leakage_mpnet_noisesweep_results.pkl` | mpnet 48-item corpus, full noise sweep. |
| `scratch_pca_leakage_minilm_noisesweep_nq8_results.pkl` | MiniLM 48-item corpus, n_qubits=8, full noise sweep. |
| `scratch_pca_leakage_480_noisesweep_results.pkl` | 480-item corpus, n_qubits=4, full noise sweep, n=20/cell. |
| `scratch_pca_leakage_480_n100_confirm_results.pkl` | 480-item corpus, mult=4.0 and 6.0, n=100/cell — the publication-grade rerun of Result 2's two headline noise levels. |
| `scratch_pca_leakage_transition_sizes_results.pkl` | Six transition-boundary corpus sizes (96–400), mult 1.2 and 2.0. |
| `scratch_hw_circuit_leakage_results.pkl` | Real-hardware angle-shift trace, 25 query/candidate pairs across 4 hardware jobs. |
| `scratch_n18_leakage_results.pkl` | Real-hardware angle-shift trace, the n=18 gate-matched PCA-vs-random comparison set. |

Only the first two files were previously rendered into a report
(`co_author_feedback_verification_2026-08-14.md`). The remaining seven — the mpnet, MiniLM,
480-item-noise-sweep, n100-confirm, and transition-sizes files — existed only as raw pickle data
until `COMPARISON.md` was written.
