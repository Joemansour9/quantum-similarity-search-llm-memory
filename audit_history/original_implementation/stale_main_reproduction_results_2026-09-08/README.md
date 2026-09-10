# Stale `main_reproduction/results/` Snapshot (superseded 2026-09-08)

This folder is a frozen copy of `main_reproduction/results/` as it stood before the 2026-09-08
three-way archive split (`main_reproduction/` / `hardware/` / `audit_history/`), preserved here
because at that point in the project's history it still mixed leaky and out-of-sample results
together, under folder names that have since been renamed. It is retained for provenance, not as a
current source — every number in it has a corrected, out-of-sample counterpart in the live
`main_reproduction/results/` tree.

Old name → current location:
- `table1_synthetic/`, `table2_realdata_easy/` → `main_reproduction/results/table1_synthetic/`,
  `table2_realdata_easy/` (same names, but this copy predates the leak-free correction: these
  subfolders here hold plain `.txt` logs, not the `_CLEAN.pkl` files the live folders use)
- `section5_1_realdata_hard/` → `main_reproduction/results/table3_realdata_hard_and_table13_baselines/`
- `embedding_model_generalization/` → `main_reproduction/results/table6_mpnet_generalisation/`
- `result123_480item/` → `main_reproduction/results/table7_480item_qubit_sweep/`,
  `table8_480item_n100_noise_sweep/`, and `corpus_scarcity_finding/` (this subfolder's
  `large_corpus_generalization_2026-07-30.md` is the source of the original, pre-correction
  "8/48 (16.7%)" corpus-scarcity figure referenced for comparison in
  `corpus_scarcity_finding/README.md`)

Do not cite numbers from this folder in the manuscript; cite the corresponding live folder listed
above instead.
