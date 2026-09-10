# Table 14 and Section 9 Discussion — Fidelity kernel vs. cosine kernel

Feeds the paper's **Table 14** (Section 9 Discussion): a comparison, at n_qubits=4–14 on the
480-item corpus (noise multiplier 1.2), between the quantum fidelity kernel, raw-classical cosine
similarity, and a classical baseline — cosine similarity computed directly on the same
PCA-projected vector, with no quantum step at all ("cosine kernel"). It also feeds two further
Discussion-section prose call-outs that use the same fidelity-kernel-vs-cosine-kernel comparison at
different scales.

**Important — this folder does not contain all of Table 14's data.** Table 14's "Fidelity kernel"
column is exactly the same numbers as the PCA column of `table7_480item_qubit_sweep/`
(the paper's own text points this out explicitly: the fidelity kernel *is* the quantum-encoded PCA
retrieval already reported in Table 7, just relabeled for this comparison). If you're looking for
that column's source file, go to `../table7_480item_qubit_sweep/`, not here — there is no
"fidelity kernel" file duplicated into this folder, and that's intentional, not a missing file.

## Which file to use

| File | What it's for |
|---|---|
| `scratch_pca_vector_crossover_oos_results.pkl` (script: `scratch_verify_pca_vector_crossover_oos.py`) | Table 14's **"Cosine kernel" and "Raw-classical" columns** at n_qubits=4–14, 480-item corpus, mult=1.2. |
| `scratch_qubitsweep48_crossover_oos_results.pkl` (script: `scratch_verify_qubitsweep48_crossover_oos.py`) | Discussion prose: the 48-item corpus's own cosine-kernel-vs-classical comparison ("trails classical at n_qubits=4 and 6, 0.780 and 0.940 against 1.000... matching exactly from n_qubits=8 onward"). |
| `scratch_noisesweep480_crossover_oos_results.pkl` (script: `scratch_verify_noisesweep480_crossover_oos.py`) | Discussion prose: the 480-item, n_qubits=4 full noise-sweep mean-deviation comparison ("mean deviation 0.345" fidelity kernel vs. "0.355" cosine kernel). |
| `scratch_wilson_ci_pca_vector_crossover_results.pkl` (script: `scratch_wilson_ci_pca_vector_crossover.py`) | Table 14's Wilson 95% confidence intervals, all three columns, n_qubits=4–14. First re-confirms all 18 point estimates against the two files above, then computes the CIs — both steps recorded in the saved pkl. |

## Fitting convention

Out-of-sample (leak-free) throughout: every fidelity/cosine-kernel computation in these three files
fits the PCA projection on the candidate batch alone, with the query excluded. None of these three
scripts had a leaky predecessor with matching methodology to compare against — the closest earlier
work is `repo/audit_history/investigation/leakage_investigation/scratch_pca_vector_classical_baseline.py`,
a related but leaky (query-included) classical baseline check that predates and motivated this
out-of-sample rerun; see that folder for the earlier, leaky version.
