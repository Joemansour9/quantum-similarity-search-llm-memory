# Investigation

The diagnostic work behind the project's methodology audit, organized into three largely
independent investigation threads.

## `leakage_investigation/`
The PCA query-leakage issue: where it originates, confirmation that it is universal, and its
measured size across every corpus, embedding model, and noise level used in this project. Also
includes the classical-simulability finding (`scratch_verify_classical_simulability.py`), which
does not correct any reported result but reframes the chain-topology fidelity kernel as classically
computable.

`pca_vector_classical_baseline_2026-08-04.md` predates the leakage discovery and addresses a
separate question: whether PCA's advantage can be explained by classical compression alone. Both
are grouped here as part of the broader question of whether the headline finding is real, not
because they are leakage checks specifically.

## `transition_boundary_followup/`
This is a scaling study, not a leakage check. A separate research question — at what corpus size
does the query-noise dissociation effect begin to appear — pursued across three reports
(`corpus_size_transition_2026-08-03.md`, `hardware_gradual_onset_2026-08-13.md`,
`transition_boundary_hardware_2026-08-13.md`), using the same, still-leaky methodology throughout.
Its own numbers are shown, retrospectively, to also carry the leakage effect — see the
transition-sizes table in `../before_after_comparisons/COMPARISON.md` — but that finding has not
been incorporated into these three reports.

## `other_checks/`
Three items, of two different kinds. Two supplementary checks from 2026-08-31/09-01, labeled
"Check 2" and "Check 3" in their filenames.

> Note: no corresponding "Check 1" file exists in the project.

Neither check has a markdown writeup.
`scratch_check_underpowered_mult6_n500.py` has a saved result — a statistical-power check using the
original leaky methodology by design, not a leakage check. `scratch_check_rzz_ablation.py` was
rewritten and run as of 2026-09-01, extended to n_qubits=4 and 8: the entangling layer
underperforms a plain product-state encoding at low noise (mult=1.2) and contributes nothing
distinguishable at moderate-to-high noise. Full numbers are in that subfolder's README.

The third item, `scratch_three_baselines_qubit_sweep_hard.py`, is not a "check" in the same sense
as the two above — it is a superseded implementation attempt (the "v1" of the Table 3/13 baseline
script now living in `main_reproduction/`), archived here because it reconstructed the wrong
historical corpus (`_POOL_PER_CATEGORY = 60` instead of 65) and its output does not match the
paper's published numbers. It is unrelated to, and (created 2026-09-07) postdates, the "Check
2"/"Check 3" naming convention.
