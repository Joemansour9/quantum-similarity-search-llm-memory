"""
Generator for repo/hardware/job_records/*.json and calibration_snapshots/*.json --
the structured per-job records referenced by hardware/README.md.

Source data: this project's IBM Quantum job history (job IDs, backends, timestamps,
PUB composition), cross-checked against the markdown result reports describing each
job's methodology and results, and sourced directly from IBM Quantum account job
records (job ID, backend, and creation timestamp match exactly for all 34 jobs).
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "job_records")
CAL_DIR = os.path.join(os.path.dirname(__file__), "calibration_snapshots")

# PCA-fitting-convention values used consistently:
#   "query-included (leaky) -- verified via circuit-level angle-shift trace"
#   "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified"
#   "N/A (random projection matrix is data-independent by construction)"
#   "mixed" jobs get a pca_fitting_convention.pca and .random sub-split

JOBS = [
    {
        "job_id": "d9l0mrqbr2fc73e812hg", "backend": "ibm_fez",
        "created": "2026-07-30T00:17:19.410940+10:00",
        "test": "Test 1 -- CSWAP-PCA fidelity", "n_pubs": 1, "corpus": "48-item hard corpus, seed=0",
        "n_qubits": 4, "circuit_type": "ancilla/CSWAP",
        "pca_fitting_convention": "query-included (leaky) -- verified via circuit-level angle-shift trace (mean shift 1.517 rad, max 1.952 rad / 62.1% of pi, PCA basis itself changed)",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit.py", "circuit_pkl": "scratch_hw_circuit.pkl",
                                 "note": "circuit-building step (R/scale fit) was run inline, not saved as a standalone script; only the pickled circuit output survives"},
    },
    {
        "job_id": "d9l17rbjf64c739isibg", "backend": "ibm_fez",
        "created": "2026-07-30T00:53:33.976479+10:00",
        "test": "Test 1 -- destructive-PCA fidelity", "n_pubs": 1, "corpus": "48-item hard corpus, seed=0",
        "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- verified via circuit-level angle-shift trace (same underlying query/candidate pair and R/scale as d9l0mrqbr2fc73e812hg, different SWAP-test apparatus)",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_destructive.py", "circuit_pkl": "scratch_hw_circuit_destructive.pkl",
                                 "note": "circuit-building step run inline"},
    },
    {
        "job_id": "d9l17vrjf64c739isifg", "backend": "ibm_fez",
        "created": "2026-07-30T00:53:51.434285+10:00",
        "test": "Test 2 -- destructive-random fidelity (near-floor pair)", "n_pubs": 1,
        "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "N/A (random projection matrix is data-independent by construction -- R drawn from (seed, n_qubits, embedding_dim) only, never touches query or candidate data)",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_destructive_random.py", "circuit_pkl": "scratch_hw_circuit_destructive_random.pkl"},
    },
    {
        "job_id": "d9l7shabr2fc73e8ao40", "backend": "ibm_kingston", "created": "2026-07-30T08:27:17.592661+10:00",
        "test": "Test 3 -- ranking preservation (1 query x 4 candidates)", "n_pubs": 4,
        "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- verified via circuit-level angle-shift trace (item_0_c0 pair, part of the n=18 leakage trace)",
        "calibration_epoch": "A",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_ranking.py", "circuit_pkl": "scratch_hw_ranking_test.pkl"},
    },
    {
        "job_id": "d9l820rhdfks73ckkco0", "backend": "ibm_kingston",
        "created": "2026-07-30T08:38:59.889595+10:00",
        "test": "Test 4 -- gate-count midpoint (opt_level=0 vs 3)", "n_pubs": 1,
        "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis, opt_level=0",
        "pca_fitting_convention": "query-included (leaky) -- verified: reuses the exact same query/candidate/R/scale as d9l0mrqbr2fc73e812hg (Test 1), only the transpiler optimization_level differs",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_opt0.py", "note": "reuses scratch_hw_circuit_destructive.pkl, no separate circuit file"},
    },
    {
        "job_id": "d9l842rjf64c739j5q4g", "backend": "ibm_kingston",
        "created": "2026-07-30T08:43:23.503444+10:00",
        "test": "Test 5 -- multi-candidate retrieval benchmark (3 queries x 4 candidates)", "n_pubs": 12,
        "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace (identical fit_pca_projection_scale(full_batch,...) call), not independently angle-verified for this job's specific pairs (item_1_c0, item_2_c0 outside the traced set; item_0_c0 portion is covered by the Test 1/3 trace)",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_multi_retrieval.py", "circuit_pkl": "scratch_hw_multi_retrieval.pkl"},
    },
    {
        "job_id": "d9l8n22br2fc73e8bma0", "backend": "ibm_kingston", "created": "2026-07-30T09:23:52.900132+10:00",
        "test": "Test 6 -- high-fidelity random-projection pair", "n_pubs": 1,
        "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "N/A (random projection, data-independent)", "calibration_epoch": "A",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_random_highfid.py", "circuit_pkl": "scratch_hw_circuit_destructive_random_highfid.pkl"},
    },
    {
        "job_id": "d9l9kb3jf64c739j7ni0", "backend": "ibm_kingston", "created": "2026-07-30T10:26:20.582379+10:00",
        "test": "Test 7 -- n=4-per-method statistical comparison", "n_pubs": 6,
        "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": {
            "pca": "query-included (leaky) -- verified via circuit-level angle-shift trace, all 3 PCA circuits (item_6_c1, item_18_c3, item_10_c1)",
            "random": "N/A (random projection, data-independent), 3 circuits",
        },
        "calibration_epoch": "B",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_ngroup.py", "circuit_pkl": "scratch_hw_ngroup.pkl"},
    },
    {
        "job_id": "d9l9uh0ii2cc73eh7bl0", "backend": "ibm_kingston", "created": "2026-07-30T10:48:05.028260+10:00",
        "test": "Test 8 -- n~18-per-method multi-seed comparison, batch 1 of 2", "n_pubs": 14,
        "corpus": "48-item hard corpus, seeds 0-2", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": {
            "pca": "query-included (leaky) -- verified via circuit-level angle-shift trace, 2 of the batch's PCA circuits confirmed at 31 gates and included in the final n=18 set (item_30_c5 seed1, item_19_c3 seed1); 7 other PCA circuits in this batch collapsed to 15 gates and were excluded from the gate-matched comparison (not separately leakage-traced)",
            "random": "N/A (random projection, data-independent), 5 circuits",
        },
        "calibration_epoch": "C",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_ngroup2.py", "circuit_pkl": "scratch_hw_ngroup2.pkl", "results_pkl": "scratch_hw_ngroup2_rows.pkl"},
    },
    {
        "job_id": "d9l9vbabr2fc73e8d8o0", "backend": "ibm_kingston", "created": "2026-07-30T10:49:49.825152+10:00",
        "test": "Test 8 -- n~18-per-method multi-seed comparison, batch 2 of 2", "n_pubs": 14,
        "corpus": "48-item hard corpus, seeds 0-2", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": {
            "pca": "query-included (leaky) -- verified via circuit-level angle-shift trace, all 5 PCA circuits in this batch confirmed at 31 gates and included in the final n=18 set (item_20_c3, item_43_c7, item_10_c1, item_30_c5, item_39_c6, all seed=2)",
            "random": "N/A (random projection, data-independent), 9 circuits",
        },
        "calibration_epoch": "C",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_ngroup2.py", "circuit_pkl": "scratch_hw_ngroup2.pkl", "results_pkl": "scratch_hw_ngroup2_rows.pkl",
                                 "note": "same submit script as batch 1 -- it loops over both batches internally"},
    },
    {
        "job_id": "d9la4sgii2cc73eh7j3g", "backend": "ibm_kingston", "created": "2026-07-30T11:01:38.247389+10:00",
        "test": "Test 9 -- gate-matched fill (7 more confirmed-31-gate PCA pairs to reach n=18)", "n_pubs": 7,
        "corpus": "48-item hard corpus, seeds 0-2", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- verified via circuit-level angle-shift trace, all 7 pairs (mean shift 1.013 rad, largest single shift 2.360 rad / 75.1% of pi across this job's 7 circuits)",
        "calibration_epoch": "C",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_pca_fill7.py", "circuit_pkl": "scratch_hw_pca_fill7.pkl"},
    },
    {
        "job_id": "d9lac03jf64c739j8i50", "backend": "ibm_kingston", "created": "2026-07-30T11:16:48.687333+10:00",
        "test": "Test 10 -- n_qubits=8 replication", "n_pubs": 16,
        "corpus": "48-item hard corpus, seeds 0/1/4/6/7", "n_qubits": 8, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": {
            "pca": "query-included (leaky) -- verified via circuit-level angle-shift trace, all 8 circuits (mean of mean-shift 1.003 rad, largest single shift 2.066 rad / 65.8% of pi)",
            "random": "N/A (random projection, data-independent), 8 circuits",
        },
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_n8.py", "circuit_pkl": "scratch_hw_n8_final.pkl"},
    },
    {
        "job_id": "d9lal32br2fc73e8e0sg", "backend": "ibm_kingston", "created": "2026-07-30T11:36:12.284422+10:00",
        "test": "Test 11 -- n_qubits=6 intermediate depth", "n_pubs": 16,
        "corpus": "48-item hard corpus, seeds 0-4", "n_qubits": 6, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": {
            "pca": "query-included (leaky) -- verified via circuit-level angle-shift trace, all 8 circuits (mean of mean-shift 1.010 rad, largest single shift 2.413 rad / 76.8% of pi -- the largest single shift found in the first trace)",
            "random": "N/A (random projection, data-independent), 8 circuits",
        },
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_n6.py", "circuit_pkl": "scratch_hw_n6_final.pkl"},
    },
    {
        "job_id": "d9lavtibr2fc73e8ed5g", "backend": "ibm_kingston", "created": "2026-07-30T11:59:18.701362+10:00",
        "test": "Test 12 -- strengthened retrieval, batch 1 of 2 (8 new queries added to Recall@1=n)",
        "n_pubs": 16, "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified (queries not fidelity-filtered/deliberately outside the traced set, since Test 12 targets retrieval correctness rather than fidelity magnitude)",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_retrieval_extra.py", "note": "circuit-building step run inline, no standalone circuit pkl found"},
    },
    {
        "job_id": "d9lb080ii2cc73eh8h7g", "backend": "ibm_kingston", "created": "2026-07-30T12:00:00.621413+10:00",
        "test": "Test 12 -- strengthened retrieval, batch 2 of 2", "n_pubs": 16,
        "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_retrieval_extra.py", "note": "same submit script as batch 1"},
    },
    {
        "job_id": "d9lblv8ii2cc73eh9a40", "backend": "ibm_kingston", "created": "2026-07-30T12:46:21.850934+10:00",
        "test": "Priority 1 -- n_qubits=5, unresolved-window test", "n_pubs": 16,
        "corpus": "48-item hard corpus, seeds 0-1", "n_qubits": 5, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": {
            "pca": "query-included (leaky) -- verified via circuit-level angle-shift trace, all 8 circuits (mean of mean-shift 1.028 rad, largest single shift 2.362 rad / 75.2% of pi)",
            "random": "N/A (random projection, data-independent), 8 circuits",
        },
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_n5.py", "circuit_pkl": "scratch_hw_n5_final.pkl"},
    },
    {
        "job_id": "d9lcqtjjf64c739jbkrg", "backend": "ibm_fez", "created": "2026-07-30T14:05:10.245677+10:00",
        "test": "Priority 2 -- CSWAP-vs-destructive, n=4 paired (strengthens Test 1)", "n_pubs": 8,
        "corpus": "48-item hard corpus, seed=0", "n_qubits": 4, "circuit_type": "4 destructive + 4 CSWAP, same 4 query/candidate pairs",
        "pca_fitting_convention": "query-included (leaky) -- verified via circuit-level angle-shift trace: reuses the exact same 4 pairs as d9l7shabr2fc73e8ao40/d9l9kb3jf64c739j7ni0 (item_0_c0, item_6_c1, item_18_c3, item_10_c1, all seed=0), same R/scale computation shared between the destructive and CSWAP circuit constructions",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_cswap_strengthen.py", "circuit_pkl": "scratch_hw_cswap_strengthen.pkl"},
    },
    {
        "job_id": "d9ls5umh4e6s738ubqt0", "backend": "ibm_kingston", "created": "2026-07-31T07:32:42.430984+10:00",
        "test": "large_corpus_generalization_2026-07-30.md, Result 2b -- hardware confirmation of the mult=2.0 query-noise-dissociation breakdown",
        "n_pubs": 16, "corpus": "480-item corpus (60/category), seed=0", "n_qubits": 4, "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified (480-item corpus, outside the 48-item-corpus angle-shift traces; the leakage effect is corpus-size-dependent and was not detected at n_qubits=4 on the 480-item corpus specifically in the qubit-sweep check -- see co_author_feedback_verification_2026-08-14.md, though that check covered recall statistics, not this job's specific circuits)",
        "surviving_artifacts": {"submit_script": "scratch_hw_submit_breakdown.py", "circuit_pkl": "scratch_hw_breakdown_circuits.pkl"},
    },
    {
        "job_id": "d9sfqi7pemts73ctm3m0", "backend": "ibm_kingston",
        "created": "2026-08-10T08:20:56.810458+10:00",
        "test": "hardware_confirmation_480item_2026-08-12.md, Tier 1a -- mult=1.2, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "480-item corpus (60/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace (identical fit_pca_projection_scale(full_batch,...) call site inherited via AgentMemory.query()), not independently angle-verified for this job's specific pairs",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_mult1.2.py", "circuit_pkl": "scratch_hw_mult1.2_circuits.pkl",
            "submit_script": "scratch_hw_submit_mult1.2.py", "results_pkl": "scratch_hw_mult1.2_results.pkl",
        },
    },
    {
        "job_id": "d9tokp343mgs73es1aug", "backend": "ibm_kingston",
        "created": "2026-08-12T06:47:32.131127+10:00",
        "test": "hardware_confirmation_480item_2026-08-12.md, Tier 1b -- mult=4.0, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "480-item corpus (60/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified. Gate counts non-uniform this batch (15 or 23, vs the usual 31) -- clipping-driven RZZ simplification at this noise level, a documented pattern, not a defect.",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_mult4.0.py", "circuit_pkl": "scratch_hw_mult4.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_mult4.0.py", "results_pkl": "scratch_hw_mult4.0_results.pkl",
        },
    },
    {
        "job_id": "d9uedg343mgs73essa60", "backend": "ibm_kingston",
        "created": "2026-08-13T07:33:52.527551+10:00",
        "test": "hardware_confirmation_480item_2026-08-12.md, Tier 1c -- mult=2.0 extension, 8 new wrong-top-1 queries (q_idx 4-13, excluding 0-3)",
        "n_pubs": 32, "corpus": "480-item corpus (60/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_mult2.0_extend.py", "circuit_pkl": "scratch_hw_mult2.0_extend_circuits.pkl",
            "submit_script": "scratch_hw_submit_mult2.0_extend.py", "results_pkl": "scratch_hw_mult2.0_extend_results.pkl",
            "note": "extends d9ls5umh4e6s738ubqt0 (q_idx 0-3) to n=12; the original 4 queries were reused unchanged, not resubmitted",
        },
    },
    {
        "job_id": "d9uejad35hes73fjr31g", "backend": "ibm_kingston",
        "created": "2026-08-13T07:46:17.587846+10:00",
        "test": "hardware_confirmation_480item_2026-08-12.md, Tier 3 -- mult=6.0, n=12 wrong-top-1 reproduction (the paper's flagged-borderline noise level)",
        "n_pubs": 48, "corpus": "480-item corpus (60/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified. Gate counts uniform at 23 (clipping-driven simplification, same pattern as mult=4.0).",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_mult6.0.py", "circuit_pkl": "scratch_hw_mult6.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_mult6.0.py", "results_pkl": "scratch_hw_mult6.0_results.pkl",
        },
    },
    {
        "job_id": "d9toqms98n5s7391tn30", "backend": "ibm_kingston",
        "created": "2026-08-12T07:00:12.015652+10:00",
        "test": "hardware_confirmation_480item_2026-08-12.md, Tier 2 (bonus) -- n_qubits=14 gate-count-saturation spot-check, PCA self-match pairs only",
        "n_pubs": 10, "corpus": "480-item corpus (60/category), seed=0", "n_qubits": 14,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified (n_qubits=14 is outside the n_qubits 4/5/6/8 range covered by the angle-shift traces)",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_tier2_nq14.py", "circuit_pkl": "scratch_hw_tier2_nq14_pool.pkl",
            "submit_script": "scratch_hw_submit_tier2_nq14.py", "results_pkl": "scratch_hw_tier2_nq14_results.pkl",
            "note": "random projection produced zero qualifying (sim_fid>=0.5) candidates at this depth/corpus combination -- PCA-only job, no PCA-vs-random comparison possible here",
        },
    },
    {
        "job_id": "d9ufha343mgs73estpa0", "backend": "ibm_kingston",
        "created": "2026-08-13T08:50:16.297700+10:00",
        "test": "hardware_gradual_onset_2026-08-13.md -- 192-item corpus, mult=1.2, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "192-item corpus (24/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified. Gate counts mixed 15/23/31 (clipping-driven simplification).",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_192_mult1.2.py", "circuit_pkl": "scratch_hw_192_mult1.2_circuits.pkl",
            "submit_script": "scratch_hw_submit_192_mult1.2.py", "results_pkl": "scratch_hw_192_mult1.2_results.pkl",
        },
    },
    {
        "job_id": "d9ufjngu5hac73ah5phg", "backend": "ibm_kingston",
        "created": "2026-08-13T08:55:26.354082+10:00",
        "test": "hardware_gradual_onset_2026-08-13.md -- 192-item corpus, mult=2.0, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "192-item corpus (24/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_192_mult2.0.py", "circuit_pkl": "scratch_hw_192_mult2.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_192_mult2.0.py", "results_pkl": "scratch_hw_192_mult2.0_results.pkl",
        },
    },
    {
        "job_id": "d9ufpbgu5hac73ah61p0", "backend": "ibm_kingston",
        "created": "2026-08-13T09:07:26.419328+10:00",
        "test": "hardware_gradual_onset_2026-08-13.md -- 192-item corpus, mult=4.0, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "192-item corpus (24/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified. Wrong-top-1 rate ~3% at this level/size -- scan cap raised to 500, 347 queries scanned to reach n=12. Gate counts mixed 15/31.",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_192_mult4.0.py", "circuit_pkl": "scratch_hw_192_mult4.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_192_mult4.0.py", "results_pkl": "scratch_hw_192_mult4.0_results.pkl",
        },
    },
    {
        "job_id": "d9ufqc535hes73fjsoe0", "backend": "ibm_kingston",
        "created": "2026-08-13T09:09:36.191175+10:00",
        "test": "hardware_gradual_onset_2026-08-13.md -- 192-item corpus, mult=6.0, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "192-item corpus (24/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified. Gate counts uniform at 31.",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_192_mult6.0.py", "circuit_pkl": "scratch_hw_192_mult6.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_192_mult6.0.py", "results_pkl": "scratch_hw_192_mult6.0_results.pkl",
        },
    },
    {
        "job_id": "d9ug2ik98n5s7392qol0", "backend": "ibm_kingston",
        "created": "2026-08-13T09:27:06.226973+10:00",
        "test": "transition_boundary_hardware_2026-08-13.md, Tier 2 -- 216-item corpus, mult=1.2, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "216-item corpus (27/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_216_mult1.2.py", "circuit_pkl": "scratch_hw_216_mult1.2_circuits.pkl",
            "submit_script": "scratch_hw_submit_216_mult1.2.py", "results_pkl": "scratch_hw_216_mult1.2_results.pkl",
        },
    },
    {
        "job_id": "d9ug37498n5s7392qp9g", "backend": "ibm_kingston",
        "created": "2026-08-13T09:28:28.629847+10:00",
        "test": "transition_boundary_hardware_2026-08-13.md, Tier 2 -- 216-item corpus, mult=2.0, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "216-item corpus (27/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_216_mult2.0.py", "circuit_pkl": "scratch_hw_216_mult2.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_216_mult2.0.py", "results_pkl": "scratch_hw_216_mult2.0_results.pkl",
        },
    },
    {
        "job_id": "d9ug1i8u5hac73ah6a7g", "backend": "ibm_kingston",
        "created": "2026-08-13T09:24:57.592751+10:00",
        "test": "transition_boundary_hardware_2026-08-13.md, Tier 1 -- 216-item corpus, mult=4.0, n=12 wrong-top-1 reproduction (headline scan-count result for this corpus size: 197 queries scanned for wrong-rate 6.1%)",
        "n_pubs": 48, "corpus": "216-item corpus (27/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_216_mult4.0.py", "circuit_pkl": "scratch_hw_216_mult4.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_216_mult4.0.py", "results_pkl": "scratch_hw_216_mult4.0_results.pkl",
        },
    },
    {
        "job_id": "d9ug5r498n5s7392qrpg", "backend": "ibm_kingston",
        "created": "2026-08-13T09:34:04.701789+10:00",
        "test": "transition_boundary_hardware_2026-08-13.md, Tier 2 -- 216-item corpus, mult=6.0, n=12 wrong-top-1 reproduction",
        "n_pubs": 48, "corpus": "216-item corpus (27/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_216_mult6.0.py", "circuit_pkl": "scratch_hw_216_mult6.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_216_mult6.0.py", "results_pkl": "scratch_hw_216_mult6.0_results.pkl",
        },
    },
    {
        "job_id": "d9ug9q8u5hac73ah6im0", "backend": "ibm_kingston",
        "created": "2026-08-13T09:42:33.741842+10:00",
        "test": "transition_boundary_hardware_2026-08-13.md, Tier 3 -- 224-item corpus, mult=4.0, n=12 wrong-top-1 reproduction (wrong-rate 5.2%, confirms 216's plateau)",
        "n_pubs": 48, "corpus": "224-item corpus (28/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_224_mult4.0.py", "circuit_pkl": "scratch_hw_224_mult4.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_224_mult4.0.py", "results_pkl": "scratch_hw_224_mult4.0_results.pkl",
        },
    },
    {
        "job_id": "d9uiq7535hes73fk02lg", "backend": "ibm_kingston",
        "created": "2026-08-13T12:34:04.319481+10:00",
        "test": "transition_boundary_hardware_2026-08-13.md, Tier 5 -- 232-item corpus, mult=4.0, n=12 wrong-top-1 reproduction (wrong-rate 8.6%, brackets the 224-240 jump)",
        "n_pubs": 48, "corpus": "232-item corpus (29/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_232_mult4.0.py", "circuit_pkl": "scratch_hw_232_mult4.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_232_mult4.0.py", "results_pkl": "scratch_hw_232_mult4.0_results.pkl",
        },
    },
    {
        "job_id": "d9ugf5s98n5s7392r6ag", "backend": "ibm_kingston",
        "created": "2026-08-13T09:53:59.188989+10:00",
        "test": "transition_boundary_hardware_2026-08-13.md, Tier 4 -- 240-item corpus, mult=4.0, n=12 wrong-top-1 reproduction (wrong-rate 12.8%, first clear break from the 192-224 plateau)",
        "n_pubs": 48, "corpus": "240-item corpus (30/category), seed=0", "n_qubits": 4,
        "circuit_type": "destructive/Bell-basis",
        "pca_fitting_convention": "query-included (leaky) -- confirmed by code-pattern trace, not independently angle-verified",
        "surviving_artifacts": {
            "build_script": "scratch_hw_build_240_mult4.0.py", "circuit_pkl": "scratch_hw_240_mult4.0_circuits.pkl",
            "submit_script": "scratch_hw_submit_240_mult4.0.py", "results_pkl": "scratch_hw_240_mult4.0_results.pkl",
        },
    },
]

os.makedirs(OUT_DIR, exist_ok=True)
for j in JOBS:
    with open(os.path.join(OUT_DIR, f"{j['job_id']}.json"), "w") as f:
        json.dump(j, f, indent=2)

index = [{"job_id": j["job_id"], "backend": j["backend"], "test": j["test"], "n_pubs": j["n_pubs"]} for j in JOBS]
index.sort(key=lambda j: next(x["created"] for x in JOBS if x["job_id"] == j["job_id"]))
with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"Wrote {len(JOBS)} job records + index.json to {OUT_DIR}")

# --- calibration snapshots ---
CAL = {
    "epoch_A.json": {
        "epoch_label": "A", "last_update_date": "2026-07-30T08:10:48+10:00", "backend": "ibm_kingston",
        "jobs_under_this_epoch": ["d9l7shabr2fc73e8ao40 (Test 3)", "d9l8n22br2fc73e8bma0 (Test 6)"],
        "two_qubit_gate_error": {"mean": 0.03926, "median": 0.00185, "min": 0.00081, "max": 1.00000, "n_gates": 704},
        "readout_error": {"mean": 0.02546, "median": 0.00873, "n_qubits": 156},
        "note": "Pulled via backend.properties(datetime=...) using the default-ibm-cloud account, the account under which these jobs were run.",
    },
    "epoch_B.json": {
        "epoch_label": "B", "last_update_date": "2026-07-30T09:31:34+10:00", "backend": "ibm_kingston",
        "jobs_under_this_epoch": ["d9l9kb3jf64c739j7ni0 (Test 7)"],
        "two_qubit_gate_error": {"mean": 0.03926, "median": 0.00185, "min": 0.00081, "max": 1.00000, "n_gates": 704},
        "readout_error": {"mean": 0.02546, "median": 0.00873, "n_qubits": 156},
        "note": "Whole-chip aggregate stats identical to epoch A and C at this precision -- see hardware/README.md for interpretation.",
    },
    "epoch_C.json": {
        "epoch_label": "C", "last_update_date": "2026-07-30T10:36:54+10:00", "backend": "ibm_kingston",
        "jobs_under_this_epoch": ["d9l9uh0ii2cc73eh7bl0 (Test 8 batch1)", "d9l9vbabr2fc73e8d8o0 (Test 8 batch2)", "d9la4sgii2cc73eh7j3g (Test 9 fill7)"],
        "two_qubit_gate_error": {"mean": 0.03926, "median": 0.00185, "min": 0.00081, "max": 1.00000, "n_gates": 704},
        "readout_error": {"mean": 0.02546, "median": 0.00873, "n_qubits": 156},
        "note": "Whole-chip aggregate stats identical to epoch A and B at this precision -- see hardware/README.md for interpretation.",
    },
}
os.makedirs(CAL_DIR, exist_ok=True)
for fname, data in CAL.items():
    with open(os.path.join(CAL_DIR, fname), "w") as f:
        json.dump(data, f, indent=2)

cal_index = {
    "epochs_retrieved": ["A", "B", "C", "D", "E"],
    "scope": "Epochs A/B/C cover the n=18 PCA-vs-random drift check (Test 3/6/7/8/9 job cluster). Epochs D/E cover a separate four-job mult=4.0 corpus-size drift check -- see hardware/README.md. Not pulled for the remaining jobs in job_records/ -- see hardware/README.md for current scope.",
    "account_used": "default-ibm-cloud",
}
with open(os.path.join(CAL_DIR, "index.json"), "w") as f:
    json.dump(cal_index, f, indent=2)

print(f"Wrote 3 calibration snapshots + index.json to {CAL_DIR}")
