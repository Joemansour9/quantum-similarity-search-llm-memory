# Figures 2–5

The four rendered figures embedded in the paper, plus the scripts that generated them.

| Script | PDF | Feeds |
|---|---|---|
| `scratch_regen_fig_qubit_sweep.py` | `fig_qubit_sweep.pdf` | Figure 2 — qubit-count sweep, PCA vs. random, across synthetic/real-easy/real-hard corpora (Tables 1, 2, 3). |
| `scratch_regen_fig_query_noise.py` | `fig_query_noise.pdf` | Figure 3 — query-noise sweep on the hard corpus, n_qubits=8 (Table 5). |
| `scratch_regen_fig_embedding_replication.py` | `fig_embedding_replication.pdf` | Figure 4 — mean |quantum−classical| deviation by projection method, MiniLM vs. mpnet, matched at n_qubits=8 (Table 6 / Section 6). |
| `scratch_regen_corpus_scale_transition.py` | `corpus_scale_transition.pdf` | Figure 5 — PCA-vs-classical gap vs. candidate-pool size (Table 9 / Section 7.1). |

## How to check these against the tables

Each script has its plotted data typed directly into the script as literal arrays (not read from a
pkl at plot time) — this is a rendering step, not a computation step. To verify a figure matches its
table, compare the arrays inside the script against the corresponding table folder listed above;
every one of the four matches its table's clean source values exactly, and every PDF's file
timestamp postdates its script's last edit (so no figure is stale relative to its own script).
