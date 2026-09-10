# Hardware Job Archive

Structured records of every real IBM Quantum job associated with this project's hardware
validation work. This directory is one of three top-level components of the repository, alongside
`main_reproduction/` (the current pipeline and result files) and `audit_history/` (the methodology
correction described in the paper's limitations section). Each is self-contained and
cross-referenced where relevant.

**A corrected (leak-free) version of this archive's hardware data now exists.** Every PCA-based
number in the "Original jobs" table below used a projection fit on a batch that included the query
itself (the leakage issue documented in `repo/audit_history/`). Of the 34 original jobs, 31 used the
leaky PCA-fitting convention and were re-submitted with out-of-sample (query-excluded) PCA fitting;
all other experimental conditions were held fixed. Those 31 corrected jobs,
their result files, and their mapping back to the original job they replace are documented in
"Corrected jobs" below and live in `corrected_job_records/` and `corrected_results/`. The manuscript's reported hardware-validation results are derived from the corrected jobs
unless explicitly stated otherwise. **Three original jobs have no corrected
counterpart**: two (`d9l17vrjf64c739isifg`, Test 2; `d9l8n22br2fc73e8bma0`, Test 6) never used PCA
at all (random projection has no fitting-batch leakage to correct), and one
(`d9sfqi7pemts73ctm3m0`, Tier 1a — mult=1.2, n=12, 480-item) *did* use the leaky PCA convention but
was never re-submitted; its numbers should still be read as leaky and uncorrected. No result
reported in the current manuscript relies on this job's values.

**Scope.** This archive documents 34 original jobs plus 31 corrected jobs (65 total), each
sourced directly from the account's IBM Quantum job records: job ID, backend, and creation
timestamp match exactly for every entry. Full per-job detail — exact ISO-8601 timestamps,
corpus/seed, n_qubits, circuit type, and surviving artifact filenames — is in
`job_records/<job_id>.json` (originals) and `corrected_job_records/<job_id>.json` (corrected). The
two reports describing the methodology and results behind the *original* jobs are in `reports/`:
`hardware_validation_2026-07-30.md` (Tests 1–14 and Priority 1–2, the 48-item corpus) and
`hardware_confirmation_480item_2026-08-12.md` (the 480-item corpus confirmations and the
n_qubits=14 spot-check). No equivalent narrative report exists yet for the corrected round; the
per-job records and result files are the authoritative source for those numbers until one is
written.

## Original jobs (leaky PCA-fitting convention)

> **Interpretation of labels**
>
> **verified** = leak-free vs. leaky rotation-angle shifts were computed directly for that job's query/candidate pairs.
>
> **code-pattern only** = the leaky fitting convention was established from the recorded call pattern, without a pair-specific rotation-angle reconstruction.

| Job ID | Backend | Date | Test / Table | PCA-fitting convention |
|---|---|---|---|---|
| `d9l0mrqbr2fc73e812hg` | ibm_fez | 2026-07-30 | Test 1 — CSWAP-PCA fidelity | query-included (leaky), **verified** |
| `d9l17rbjf64c739isibg` | ibm_fez | 2026-07-30 | Test 1 — destructive-PCA fidelity | query-included (leaky), **verified** |
| `d9l17vrjf64c739isifg` | ibm_fez | 2026-07-30 | Test 2 — destructive-random (near-floor pair) | N/A (random projection) |
| `d9l7shabr2fc73e8ao40` | ibm_kingston | 2026-07-30 | Test 3 — ranking preservation (4 PUBs) | query-included (leaky), **verified** |
| `d9l820rhdfks73ckkco0` | ibm_kingston | 2026-07-30 | Test 4 — gate-count midpoint (opt_level=0) | query-included (leaky), **verified** (reuses Test 1's circuit) |
| `d9l842rjf64c739j5q4g` | ibm_kingston | 2026-07-30 | Test 5 — multi-candidate retrieval (12 PUBs) | query-included (leaky), code-pattern only |
| `d9l8n22br2fc73e8bma0` | ibm_kingston | 2026-07-30 | Test 6 — high-fidelity random pair | N/A (random projection) |
| `d9l9kb3jf64c739j7ni0` | ibm_kingston | 2026-07-30 | Test 7 — n=4/method comparison (6 PUBs) | mixed: PCA leaky **verified** (3), random N/A (3) |
| `d9l9uh0ii2cc73eh7bl0` | ibm_kingston | 2026-07-30 | Test 8 — n≈18/method, batch 1 (14 PUBs) | mixed: PCA leaky **verified** (2 of 9), random N/A (5) |
| `d9l9vbabr2fc73e8d8o0` | ibm_kingston | 2026-07-30 | Test 8 — n≈18/method, batch 2 (14 PUBs) | mixed: PCA leaky **verified** (5), random N/A (9) |
| `d9la4sgii2cc73eh7j3g` | ibm_kingston | 2026-07-30 | Test 9 — gate-matched PCA fill (7 PUBs) | query-included (leaky), **verified** |
| `d9lac03jf64c739j8i50` | ibm_kingston | 2026-07-30 | Test 10 — n_qubits=8 replication (16 PUBs) | mixed: PCA leaky **verified** (8), random N/A (8) |
| `d9lal32br2fc73e8e0sg` | ibm_kingston | 2026-07-30 | Test 11 — n_qubits=6 intermediate (16 PUBs) | mixed: PCA leaky **verified** (8), random N/A (8) |
| `d9lavtibr2fc73e8ed5g` | ibm_kingston | 2026-07-30 | Test 12 — retrieval strengthening, batch 1 (16 PUBs) | query-included (leaky), code-pattern only |
| `d9lb080ii2cc73eh8h7g` | ibm_kingston | 2026-07-30 | Test 12 — retrieval strengthening, batch 2 (16 PUBs) | query-included (leaky), code-pattern only |
| `d9lblv8ii2cc73eh9a40` | ibm_kingston | 2026-07-30 | Priority 1 — n_qubits=5 unresolved window (16 PUBs) | mixed: PCA leaky **verified** (8), random N/A (8) |
| `d9lcqtjjf64c739jbkrg` | ibm_fez | 2026-07-30 | Priority 2 — CSWAP-vs-destructive, n=4 paired (8 PUBs) | query-included (leaky), **verified** (reuses Test 3/7's pairs) |
| `d9ls5umh4e6s738ubqt0` | ibm_kingston | 2026-07-31 | `large_corpus_generalization…md` Result 2b — hardware breakdown confirmation (480-item corpus) | query-included (leaky), code-pattern only |
| `d9sfqi7pemts73ctm3m0` | ibm_kingston | 2026-08-10 | `hardware_confirmation_480item…md` Tier 1a — mult=1.2, n=12 (480-item) | query-included (leaky), code-pattern only |
| `d9toqms98n5s7391tn30` | ibm_kingston | 2026-08-12 | `hardware_confirmation_480item…md` Tier 2 — n_qubits=14 spot-check, PCA-only (480-item) | query-included (leaky), code-pattern only |
| `d9tokp343mgs73es1aug` | ibm_kingston | 2026-08-12 | `hardware_confirmation_480item…md` Tier 1b — mult=4.0, n=12 (480-item) | query-included (leaky), code-pattern only |
| `d9uedg343mgs73essa60` | ibm_kingston | 2026-08-13 | `hardware_confirmation_480item…md` Tier 1c — mult=2.0, 8 new queries extending Result 2b to n=12 (480-item) | query-included (leaky), code-pattern only |
| `d9uejad35hes73fjr31g` | ibm_kingston | 2026-08-13 | `hardware_confirmation_480item…md` Tier 3 — mult=6.0, n=12 (480-item) | query-included (leaky), code-pattern only |
| `d9ufha343mgs73estpa0` | ibm_kingston | 2026-08-13 | `hardware_gradual_onset…md` — mult=1.2, n=12 (192-item) | query-included (leaky), code-pattern only |
| `d9ufjngu5hac73ah5phg` | ibm_kingston | 2026-08-13 | `hardware_gradual_onset…md` — mult=2.0, n=12 (192-item) | query-included (leaky), code-pattern only |
| `d9ufpbgu5hac73ah61p0` | ibm_kingston | 2026-08-13 | `hardware_gradual_onset…md` — mult=4.0, n=12 (192-item) | query-included (leaky), code-pattern only |
| `d9ufqc535hes73fjsoe0` | ibm_kingston | 2026-08-13 | `hardware_gradual_onset…md` — mult=6.0, n=12 (192-item) | query-included (leaky), code-pattern only |
| `d9ug1i8u5hac73ah6a7g` | ibm_kingston | 2026-08-13 | `transition_boundary_hardware…md` Tier 1 — mult=4.0, n=12 (216-item, headline scan-count result) | query-included (leaky), code-pattern only |
| `d9ug2ik98n5s7392qol0` | ibm_kingston | 2026-08-13 | `transition_boundary_hardware…md` Tier 2 — mult=1.2, n=12 (216-item) | query-included (leaky), code-pattern only |
| `d9ug37498n5s7392qp9g` | ibm_kingston | 2026-08-13 | `transition_boundary_hardware…md` Tier 2 — mult=2.0, n=12 (216-item) | query-included (leaky), code-pattern only |
| `d9ug5r498n5s7392qrpg` | ibm_kingston | 2026-08-13 | `transition_boundary_hardware…md` Tier 2 — mult=6.0, n=12 (216-item) | query-included (leaky), code-pattern only |
| `d9ug9q8u5hac73ah6im0` | ibm_kingston | 2026-08-13 | `transition_boundary_hardware…md` Tier 3 — mult=4.0, n=12 (224-item) | query-included (leaky), code-pattern only |
| `d9ugf5s98n5s7392r6ag` | ibm_kingston | 2026-08-13 | `transition_boundary_hardware…md` Tier 4 — mult=4.0, n=12 (240-item) | query-included (leaky), code-pattern only |
| `d9uiq7535hes73fk02lg` | ibm_kingston | 2026-08-13 | `transition_boundary_hardware…md` Tier 5 — mult=4.0, n=12 (232-item) | query-included (leaky), code-pattern only |

## Corrected jobs (leak-free PCA-fitting convention)

Each row below is a real IBM Quantum job, re-submitted from scratch with the query excluded from
the PCA-fitting batch. Every corrected job reuses the same corpus, seed, candidate pairs, circuit
type(s), n_qubits, and backend as the original job it replaces — the PCA-fitting convention is the
only thing that changed. Full per-job detail, including the exact rotation angles used and a
pointer to the raw result file, is in `corrected_job_records/<job_id>.json`; the raw per-item counts
and computed fidelities are in `corrected_results/<file>.pkl`.

Before submission, every corrected circuit was checked against
`corrected_results/scratch_hw_corrected_circuits_readiness.pkl` for equivalence with the intended
leak-free encoding. After submission, identity was independently
re-verified end to end: the circuit actually pulled live from IBM's own job-submission record was
compared against a freshly-rebuilt leak-free circuit via full quantum-state fidelity (not gate-count
matching), and each stored result was independently recomputed from its own raw measurement counts.
This covered all 42 query/candidate pairs behind the four-depth (n_qubits=4/5/6/8) gate-noise
comparison and all 682 queries behind the 16-job corpus-scale comparison, with zero mismatches
(differences at floating-point noise level, 1e-11 to 1e-16) — see the paper's Limitations section
for the same claim in the manuscript's own words.

| Corrected Job ID | Original Job ID | Backend | Date | Label / n_qubits | Circuit type(s) |
|---|---|---|---|---|---|
| `dacdh9t1ierc738jf8hg` | `d9l0mrqbr2fc73e812hg` | ibm_fez | 2026-09-03 | Test1-CSWAP (n_qubits=4) | ancilla |
| `dacdhc51ierc738jf8k0` | `d9l17rbjf64c739isibg` | ibm_fez | 2026-09-03 | Test1-destructive (n_qubits=4) | destructive |
| `dacdhh3dd5gc73d5trp0` | `d9l7shabr2fc73e8ao40` | ibm_kingston | 2026-09-03 | Test3-ranking (n_qubits=4) | destructive |
| `dacdl9m42tqs73as1h0g` | `d9l820rhdfks73ckkco0` | ibm_kingston | 2026-09-03 | Test4-opt0 (n_qubits=4) | destructive |
| `dacdsrd1ierc738jfkj0` | `d9l842rjf64c739j5q4g` | ibm_kingston | 2026-09-03 | Test5-multiretrieval (n_qubits=4) | destructive |
| `dacgt551ierc738jj9l0` | `d9l9kb3jf64c739j7ni0` | ibm_kingston | 2026-09-03 | Test7-n4permethod (n_qubits=4) | destructive |
| `dacogvjdd5gc73d6brdg` | `d9l9uh0ii2cc73eh7bl0` | ibm_kingston | 2026-09-03 | Test8-batch1 (n_qubits=4) | destructive |
| `dacoh9642tqs73asfef0` | `d9l9vbabr2fc73e8d8o0` | ibm_kingston | 2026-09-03 | Test8-batch2 (n_qubits=4) | destructive |
| `dacohmd1ierc738jt7l0` | `d9la4sgii2cc73eh7j3g` | ibm_kingston | 2026-09-03 | Test9-fill7 (n_qubits=4) | destructive |
| `dacohqtnj4cs73actna0` | `d9lac03jf64c739j8i50` | ibm_kingston | 2026-09-03 | Test10-nq8 (n_qubits=8) | destructive |
| `dacohutnj4cs73actnf0` | `d9lal32br2fc73e8e0sg` | ibm_kingston | 2026-09-03 | Test11-nq6 (n_qubits=6) | destructive |
| `dacoi2tnj4cs73actnk0` | `d9lavtibr2fc73e8ed5g` | ibm_kingston | 2026-09-03 | Test12-batch1 (n_qubits=4) | destructive |
| `dacoierdd5gc73d6bta0` | `d9lb080ii2cc73eh8h7g` | ibm_kingston | 2026-09-03 | Test12-batch2 (n_qubits=4) | destructive |
| `dacoinu42tqs73asfg50` | `d9lblv8ii2cc73eh9a40` | ibm_kingston | 2026-09-03 | Test13-nq5 (n_qubits=5) | destructive |
| `dacoir3dd5gc73d6bts0` | `d9lcqtjjf64c739jbkrg` | ibm_fez | 2026-09-03 | Test14-cswapvsdestr (n_qubits=4) | ancilla, destructive |
| `dacpanjdd5gc73d6cpm0` | `d9ls5umh4e6s738ubqt0` | ibm_kingston | 2026-09-03 | Result2b-mult2.0-4query (n_qubits=4) | destructive |
| `dacpfcl1ierc738jubng` | `d9uedg343mgs73essa60` | ibm_kingston | 2026-09-03 | 480-mult2.0-8new (n_qubits=4) | destructive |
| `dacpfnjdd5gc73d6d0n0` | `d9tokp343mgs73es1aug` | ibm_kingston | 2026-09-03 | 480-mult4.0-n12 (n_qubits=4) | destructive |
| `dacpftdnj4cs73acurp0` | `d9uejad35hes73fjr31g` | ibm_kingston | 2026-09-03 | 480-mult6.0-n12 (n_qubits=4) | destructive |
| `dacpg451ierc738juco0` | `d9toqms98n5s7391tn30` | ibm_kingston | 2026-09-03 | Tier2-nq14-selfmatch (n_qubits=14) | destructive |
| `dacpi4lnj4cs73acuub0` | `d9ufha343mgs73estpa0` | ibm_kingston | 2026-09-03 | 192-mult1.2 (n_qubits=4) | destructive |
| `dacpkal1ierc738jukf0` | `d9ufjngu5hac73ah5phg` | ibm_kingston | 2026-09-03 | 192-mult2.0 (n_qubits=4) | destructive |
| `dacpnmm42tqs73ash38g` | `d9ufpbgu5hac73ah61p0` | ibm_kingston | 2026-09-03 | 192-mult4.0 (n_qubits=4) | destructive |
| `dacpo33dd5gc73d6dgb0` | `d9ufqc535hes73fjsoe0` | ibm_kingston | 2026-09-03 | 192-mult6.0 (n_qubits=4) | destructive |
| `dacpo8tnj4cs73acvcb0` | `d9ug2ik98n5s7392qol0` | ibm_kingston | 2026-09-03 | 216-mult1.2 (n_qubits=4) | destructive |
| `dacpoejdd5gc73d6dgpg` | `d9ug37498n5s7392qp9g` | ibm_kingston | 2026-09-03 | 216-mult2.0 (n_qubits=4) | destructive |
| `dacpokjdd5gc73d6dh2g` | `d9ug1i8u5hac73ah6a7g` | ibm_kingston | 2026-09-03 | 216-mult4.0 (n_qubits=4) | destructive |
| `dacpoqd1ierc738jut3g` | `d9ug5r498n5s7392qrpg` | ibm_kingston | 2026-09-03 | 216-mult6.0 (n_qubits=4) | destructive |
| `dacpp05nj4cs73acvd60` | `d9ug9q8u5hac73ah6im0` | ibm_kingston | 2026-09-03 | 224-mult4.0 (n_qubits=4) | destructive |
| `dacpp6642tqs73ash4r0` | `d9uiq7535hes73fk02lg` | ibm_kingston | 2026-09-03 | 232-mult4.0 (n_qubits=4) | destructive |
| `dacppel1ierc738jutp0` | `d9ugf5s98n5s7392r6ag` | ibm_kingston | 2026-09-03 | 240-mult4.0 (n_qubits=4) | destructive |

**Note on transpiled circuit depth.** Correcting the PCA-fitting convention changes the rotation
angles baked into each circuit, which can shift the transpiled depth by a few gates even though the
circuit's structure (qubit count, gate types, two-qubit gate count) is unchanged. For example, the
corrected Test1-CSWAP job transpiles to depth 142 (vs. 138 originally) and the corrected
Test1-destructive job to depth 53 (vs. 54 originally), both at unchanged two-qubit gate counts (66
and 31 respectively). Any prose or table citing transpiled depth should cite the depth of the
circuit that actually produced the reported fidelity — the corrected depth for corrected-job
numbers, the original depth for original-job numbers — rather than mixing the two.

For the leakage-trace investigation on the query-included jobs above (per-pair rotation-angle
shifts, basis stability) see `repo/audit_history/before_after_comparisons/COMPARISON.md`.

For the local (no-hardware) validation of the destructive circuit's shot-based fidelity estimate
against exact statevector fidelity, across all 144 pairs staged for the destructive-circuit jobs
above, see `repo/main_reproduction/results/appendix_destructive_circuit_validation/`.

## Calibration Snapshots

Calibration data associated with the original hardware campaigns is archived in `calibration_snapshots/`.

See `calibration_snapshots/README.md`.

## Reproducibility notes

1. **Circuit-building steps were mostly inline, not saved as scripts.** Every *submission* script
   survives on disk (see each job record's `surviving_artifacts.submit_script`). The
   *circuit-building* step (computing R/scale via `fit_pca_projection_scale`, pickling the circuit)
   was run inline for most jobs — only the resulting `.pkl` circuit files survive, not the inline
   code. The leakage trace in `repo/audit_history/before_after_comparisons/COMPARISON.md`
   therefore reconstructs the historical corpus rather than re-running a saved script.

2. **Reconstruction accounts for a corpus-pool change.** `_POOL_PER_CATEGORY` in
   `benchmark_realdata_embeddings_hard.py` was 60 when the first 18 jobs in the table above were
   built, and was raised to 65 later in the project for an unrelated large-corpus test. The
   leakage trace reconstructs the historical pool=60 embedding set from `_TEXTS_BY_CAT` (which is
   not truncated by `_POOL_PER_CATEGORY` at load time) rather than using the current 65-item pool,
   to avoid comparing against the wrong documents.

## Contents

```
hardware/
├── README.md                       (this file)
├── _build_job_records.py           (generator script — provenance/reproducibility, not itself data)
├── reports/
│   ├── hardware_validation_2026-07-30.md            (Tests 1-14, Priority 1-2 — 48-item corpus, ORIGINAL/leaky)
│   └── hardware_confirmation_480item_2026-08-12.md  (480-item corpus confirmations, n_qubits=14 spot-check, ORIGINAL/leaky)
├── job_records/
│   ├── index.json                  (job_id, backend, test, n_pubs — flat summary, all 34 original jobs)
│   └── <job_id>.json × 34          (one per ORIGINAL job: backend, timestamp, corpus, n_qubits,
│                                     circuit_type, pca_fitting_convention, surviving_artifacts)
├── corrected_job_records/          (leak-free rerun of 31 of the 34 original jobs — see
│                                     "Corrected jobs" above for which 3 have no counterpart)
│   ├── index.json                  (job_id, original_job_id, backend, test, n_pubs — flat summary, all 31)
│   └── <job_id>.json × 31          (one per CORRECTED job: original_job_id, backend, timestamp,
│                                     corpus, n_qubits, circuit_type, pca_fitting_convention
│                                     ["out-of-sample (leak-free, corrected)"], surviving_artifacts)
├── corrected_results/
│   ├── scratch_hw_corrected_<label>_results.pkl × 31   (raw per-item counts, hw/sim fidelity,
│   │                                 transpiled depth/gate count — one file per corrected job,
│   │                                 referenced by corrected_job_records/<job_id>.json)
│   └── scratch_hw_corrected_circuits_readiness.pkl     (pre-submission equivalence check for all
│                                     31 corrected circuits — R_free, scale_free, rotation angles,
│                                     per-candidate sanity_check, keyed by original job_id)
└── calibration_snapshots/
    ├── README.md                   (full detail — epochs, whole-chip aggregates, scope caveats)
    ├── index.json                  (scope note + account used — covers epochs A-E)
    └── epoch_{A,B,C,D,E}.json      (last_update_date, jobs under that epoch, gate/readout error
                                      stats — see calibration_snapshots/README.md; ORIGINAL jobs
                                      only, no calibration snapshots were pulled for the corrected
                                      round)
```
