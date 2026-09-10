# Main Reproduction

**Status (2026-09-08):** Every table, figure, and prose numeric claim in the current manuscript has been re-audited against its source file. All results in this directory use out-of-sample (leak-free) fitting.

This directory is the authoritative source for the manuscript's current simulation results.

Historical implementations, investigation artefacts, and superseded result files are archived in:

`repo/audit_history/`

## `results/` — the files behind each reported number

All result files referenced below correspond to the manuscript's final out-of-sample
implementation. Historical and comparison artefacts have been archived under
`repo/audit_history/`.

The table below provides the canonical mapping from manuscript elements to the files that generated them.

| Folder | Paper element | Source | Fitting |
|---|---|---|---|
| `background_variance_enrichment/` | Background §2.3 prose (PCA variance/enrichment percentages) | `scratch_verify_variance_enrichment.py`/`_results.pkl` | clean (no query involved, so no leak applies) |
| `table1_synthetic/` | Table 1 (synthetic qubit sweep) | `scratch_table1_table2_table6minilm_CLEAN.pkl`, key `'synthetic'` | clean |
| `table2_realdata_easy/` | Table 2 (real-easy qubit sweep) | same file, key `'real_easy'` | clean |
| `table3_realdata_hard_and_table13_baselines/` | Table 3 (real-hard qubit sweep, Random/PCA) and Table 13 (SVD/whitening/first-n baselines) | `scratch_three_baselines_qubit_sweep_hard_v2.py` / `..._results.pkl` | clean |
| `table4_noise_easy_and_synthetic/` | Table 4 (noise sweep, easy + synthetic) | `scratch_verify_pca_leakage_noiseeasy_synthetic.py` / `..._results.pkl` | clean (script has no leaky branch at all) |
| `table5_hard48_nq8_full_noise_sweep/` | Table 5 (hard corpus, n_qubits=8, full noise sweep) | `scratch_verify_pca_leakage_minilm_noisesweep_nq8.py` → `scratch_table5_minilm_noisesweep_nq8_CLEAN.pkl` | clean |
| `table6_mpnet_generalisation/` | Table 6 (MiniLM vs mpnet gap) + Section 6 prose (mean-deviation ratios) | MiniLM row: `scratch_table1_table2_table6minilm_CLEAN.pkl` key `'real_hard_48'`; mpnet row: `scratch_table6_mpnet_CLEAN.pkl`; prose: `scratch_sec6_mpnet_noisesweep_nq8_CLEAN.pkl`, `scratch_sec6_minilm_noisesweep_nq4_CLEAN.pkl` | clean |
| `table7_480item_qubit_sweep/` | Table 7 (480-item qubit sweep, Wilson CIs) | `scratch_table7_480qubitsweep_CLEAN.pkl` + `scratch_wilson_ci_480_qubit_sweep.py`/`_results.pkl` | clean |
| `table8_480item_n100_noise_sweep/` | Table 8 (480-item, n=100 confirmatory noise sweep) | `scratch_table8_480n100_lowmult_CLEAN.pkl` (mult 0.2/1.2/2.0) + `scratch_table8_480n100_highmult_CLEAN.pkl` (mult 4.0/6.0) | clean |
| `table9_corpus_transition_sizes/` | Table 9 (corpus sizes 96–400) | `scratch_table9_transition_sizes_CLEAN.pkl` + `scratch_wilson_ci_transition_sizes.py`/`_results.pkl` | clean |
| `table14_and_discussion_kernel_crossover/` | Table 14 (fidelity vs. cosine kernel crossover) and Section 9 Discussion prose (48-item cosine-kernel comparison; 480-item noise-sweep "0.345 vs 0.355") | cosine-kernel column: `scratch_verify_pca_vector_crossover_oos.py`/`_results.pkl`; fidelity-kernel column = same values as `table7_480item_qubit_sweep/` (noted in-file); 48-item Discussion prose: `scratch_verify_qubitsweep48_crossover_oos.py`/`_results.pkl`; 480-item noise-sweep prose: `scratch_verify_noisesweep480_crossover_oos.py`/`_results.pkl` | clean |
| `methods_rzz_entangling_ablation/` | Section 3.1 Methods prose (entangling-layer ablation) | `scratch_verify_rzz_ablation_oos.py`/`_results.pkl` | clean (out-of-sample rerun; supersedes the archived version in `audit_history/`) |
| `appendix_destructive_circuit_validation/` | Appendix Table `destructive-validation-appendix` and Section 8.1 prose (144-pair destructive-circuit shot-noise validation) | `scratch_verify_destructive_validation_all_staged_pairs_CORRECTED.py`/`_results.pkl` | clean for the 101 PCA pairs (out-of-sample rerun); the 43 random pairs are unaffected by the leak and carried over unchanged (see folder README) |
| `corpus_scarcity_finding/` | Section 7 prose (corpus-scarcity finding: fraction of pairs clearing a 0.5 fidelity threshold, 480- and 48-item corpora) | `scratch_verify_scarcity_oos.py`/`_results.pkl` | clean for PCA at both scales; the 48-item random figure is reported as a range (~17%–19%) since no source script survives to confirm which of two candidate values (8/48 published vs. 9/48 reconstructed) is correct (see folder README) |
| `figures/` | Figures 2–5 | the four `scratch_regen_*.py` scripts plus their rendered `.pdf` outputs | clean |

## `pipeline/` — the current library and benchmark drivers

The core library (`q_encoder.py`, `q_graph.py`, `q_search.py`, `agent_memory.py`) is
**leakage-agnostic**: `fit_pca_projection_scale()` fits on whatever batch of vectors it is handed.
Whether a given result is leaky or clean is a property of what the *calling script* passes as that
batch — every result script referenced above passes the candidate batch alone, with the query
excluded, which is what makes its output out-of-sample. This library code is unchanged from the
original implementation (byte-for-byte identical to `repo/audit_history/original_implementation/`);
the correction happened entirely in how the result-producing scripts call it, not in the library
itself.

`benchmark_realdata_embeddings_hard.py` uses `_POOL_PER_CATEGORY = 65`, matching the configuration
used for the manuscript results (see `results/table3_realdata_hard_and_table13_baselines/`).

Background §2.3's variance/enrichment percentages (21.7%/58.8% MiniLM, 21.0%/57.5% mpnet) are
computed and saved in `results/background_variance_enrichment/`. The Limitations §9.2
standard-error range (0.019–0.027, worst case 0.031) is a pure algebraic derivation from the
shot-noise formula (no corpus or data file needed) and has no separate saved result file.

### Hardware-sourced numbers (Tables 10–12)
Every hardware number (the gate-count comparison, ranking-preservation test, and multi-query
benchmark) is archived once, canonically, in `repo/hardware/corrected_results/` and
`repo/hardware/corrected_job_records/`. Not duplicated here.

## What is deliberately excluded from this folder

- Anything with a leaky branch, in whole or in part — see `repo/audit_history/` instead.
- Historical dual-mode (leaky + leak_free) comparison files are archived in
  `repo/audit_history/before_after_comparisons/`.
