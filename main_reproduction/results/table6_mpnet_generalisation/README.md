# Table 6 and Section 6 — Second embedding model (mpnet) generalisation

Feeds the paper's **Table 6** (Section 6): the qubit-efficiency gap (PCA Recall@1, Random Recall@1,
and the gap between them, each as a range over n_qubits=4–14) compared between the original
embedding model (all-MiniLM-L6-v2, d=384) and a second one (all-mpnet-base-v2, d=768), on the same
48-item hard corpus. It also feeds **Section 6's prose** — the specific mean-|quantum−classical|
deviation numbers and PCA-vs-random ratios quoted in the text and plotted in **Figure 4**.

This folder has four result files because Table 6 and its surrounding prose draw on four separate
computations. Here's what each one is and which specific number it's for — you shouldn't need to
open any of them in Python to know which file you want:

| File | What it contains | What it's for |
|---|---|---|
| `scratch_table1_table2_table6minilm_CLEAN.pkl` | **Shared file, also used by two other folders.** Load it and read `['leak_free_only']['real_hard_48']` — a dict keyed by `(n_qubits, 'pca')`/`(n_qubits, 'random')`. The other two keys this file contains, `'synthetic'` and `'real_easy'`, belong to Tables 1 and 2 respectively and are **not relevant here.** | Table 6's **MiniLM row** (PCA range 0.66–1.00, Random range 0.12–0.42) |
| `scratch_table6_mpnet_CLEAN.pkl` | Load it and read `['leak_free_only']`, a dict keyed by `(n_qubits, 'pca')`/`(n_qubits, 'random')` for the mpnet embedding on the same 48-item corpus. | Table 6's **mpnet row** (PCA range 0.96–1.00, Random range 0.16–0.38) |
| `scratch_sec6_mpnet_noisesweep_nq8_CLEAN.pkl` | mpnet, n_qubits=8, full query-noise sweep, leak-free only. | Section 6 prose: mpnet's PCA=0.980/classical=1.000 at mult=1.2; PCA=0.280/classical=1.000 at mult=6.0; mean deviation 0.326 (PCA) / 0.602 (random); the "1.85×" ratio in Figure 4. |
| `scratch_sec6_minilm_noisesweep_nq4_CLEAN.pkl` | MiniLM, n_qubits=4, full query-noise sweep, leak-free only. | Section 6 prose: MiniLM's mean deviation 0.384 at n_qubits=4, and the "1.40×" ratio compared against the "1.92×" ratio at n_qubits=8 (that one comes from `table5_hard48_nq8_full_noise_sweep/`, not from a file in this folder). |

The three `.py` scripts alongside these — `scratch_verify_pca_leakage_mpnet_qubitsweep.py`,
`scratch_verify_pca_leakage_mpnet_noisesweep_nq8.py`, `scratch_verify_pca_leakage_minilm_noisesweep_nq4.py`
— are the scripts that produced the mpnet/MiniLM-nq4 files above (before the leaky branch was
stripped out). Their own docstrings refer to an earlier internal report's "Result 1" / "Result 2"
labels — that report predates the paper's current Table/Section numbering and isn't archived near
this folder, so ignore those labels; the table above maps everything to the paper's actual Table 6
and Section 6 as it stands today.

## Fitting convention

Out-of-sample (leak-free) throughout, for both PCA and Random projection, in all four files. Each
was extracted from a dual-mode pkl that originally stored a leaky and a leak-free computation side
by side; the full leaky-vs-clean comparisons live in `repo/audit_history/before_after_comparisons/`
(`scratch_pca_leakage_remaining_results.pkl`, `scratch_pca_leakage_mpnet_qubitsweep_results.pkl`,
`scratch_pca_leakage_mpnet_noisesweep_nq8_results.pkl`, `scratch_pca_leakage_minilm_noisesweep_nq4_results.pkl`).
