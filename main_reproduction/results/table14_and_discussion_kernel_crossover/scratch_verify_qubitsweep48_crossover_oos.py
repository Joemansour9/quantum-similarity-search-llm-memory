"""
scratch_verify_qubitsweep48_crossover_oos.py — out-of-sample rerun of the
cosine-kernel column behind the paper's claim that "the cosine kernel
reproduces raw-embedding classical recall exactly at every tested qubit
count" on the 48-item hard corpus (mult=1.2, n_qubits 4-14).

The existing cosine-kernel values come from
scratch_pca_vector_classical_baseline.py's qubit_sweep_48 block
(classical_pca_vec), which fits PCA leakily (full_batch =
vstack([query, candidates])). This rerun changes only the fitting step:
R fit on candidate_batch alone (query excluded), same corpus
construction (make_hard_real_corpus, today's pool=65 -- confirmed to
be the convention that actually reproduces Table qubit-hard's
published numbers), same 5 seeds x 10 queries = 50/
cell, same mult=1.2, same n_qubits sweep.

raw_classical is recomputed fresh too, purely as a sanity check -- it
does not involve PCA fitting and should match the published 1.000
constant at every qubit count exactly.

Additional note: the paper's
qualitative claim "fidelity kernel consistently trails" was checked
against BOTH the leaky qubit_sweep_48 quantum_pca (0.88-1.00) and the
already-corrected Table qubit-hard PCA column (0.660-1.00) -- the
fidelity kernel trails the cosine kernel's ceiling under either version,
so that qualitative direction does not depend on which fidelity data is
used, unlike the noise-sweep comparison recomputed alongside this one.
"""
import time
import numpy as np
import pickle
from q_encoder import fit_pca_projection_scale
from q_search import cosine_similarity, recall_at_k
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD

MULT = 1.2
N_SEEDS = 5
N_QUERIES = 10
QUBIT_COUNTS = (4, 6, 8, 10, 12, 14)
noise = MULT * CORPUS_PER_DIM_STD

# Published (leaky) reference, from scratch_pca_vector_classical_baseline_results.pkl['qubit_sweep_48']
PUBLISHED = {4: 1.0, 6: 1.0, 8: 1.0, 10: 1.0, 12: 1.0, 14: 1.0}  # classical_pca_vec, all ceiling
PUBLISHED_RAW = {nq: 1.0 for nq in QUBIT_COUNTS}


def cosine_kernel_recall1_oos(query_vec, candidate_batch, keys, relevant_idx, n_qubits):
    R, _, _ = fit_pca_projection_scale(candidate_batch, n_qubits)  # OUT-OF-SAMPLE
    query_proj = R @ query_vec
    candidates_proj = candidate_batch @ R.T
    cosines = np.array([cosine_similarity(query_proj, candidates_proj[i]) for i in range(len(keys))])
    rank = np.argsort(-cosines)
    return recall_at_k(rank, relevant_idx, 1)


print(f"=== Out-of-sample cosine-kernel rerun, 48-item hard corpus, mult={MULT}, n=50/cell ===")
results = {}
for n_qubits in QUBIT_COUNTS:
    t0 = time.perf_counter()
    cos_recalls, raw_recalls = [], []
    for seed in range(N_SEEDS):
        corpus, keys = make_hard_real_corpus(48, 8, seed=seed)
        d_dim = next(iter(corpus.values())).shape[0]
        candidate_batch = np.stack([corpus[k] for k in keys])
        rng = np.random.default_rng(seed + 1)
        for q_idx in range(N_QUERIES):
            relevant_key = keys[q_idx % len(keys)]
            relevant_idx = keys.index(relevant_key)
            query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)

            cos_recalls.append(cosine_kernel_recall1_oos(query_vec, candidate_batch, keys, relevant_idx, n_qubits))

            raw_cos = np.array([cosine_similarity(query_vec, candidate_batch[i]) for i in range(len(keys))])
            raw_rank = np.argsort(-raw_cos)
            raw_recalls.append(recall_at_k(raw_rank, relevant_idx, 1))

    elapsed = time.perf_counter() - t0
    cos_kernel = float(np.mean(cos_recalls))
    raw_classical = float(np.mean(raw_recalls))
    results[n_qubits] = {"cosine_kernel_oos": cos_kernel, "raw_classical": raw_classical}
    print(f"  n_qubits={n_qubits:2d}  cosine_kernel(OOS)={cos_kernel:.3f}  raw_classical={raw_classical:.3f}  ({elapsed:.1f}s)")

print("\n=== Sanity check: fresh raw_classical vs. published (should be 1.000 everywhere) ===")
all_match = True
for nq in QUBIT_COUNTS:
    fresh = results[nq]["raw_classical"]
    published = PUBLISHED_RAW[nq]
    ok = abs(fresh - published) < 1e-9
    all_match = all_match and ok
    print(f"  n_qubits={nq:2d}  fresh={fresh:.3f}  published={published:.3f}  match={ok}")
print(f"raw_classical matches published exactly at every qubit count: {all_match}")

print("\n=== Before / after: cosine kernel, leaky (published) vs. out-of-sample ===")
still_exact = True
for nq in QUBIT_COUNTS:
    leaky = PUBLISHED[nq]
    oos = results[nq]["cosine_kernel_oos"]
    exact = abs(oos - results[nq]["raw_classical"]) < 1e-9
    still_exact = still_exact and exact
    print(f"  n_qubits={nq:2d}  leaky(published)={leaky:.3f}  out-of-sample={oos:.3f}  "
          f"still exact match to raw_classical={exact}")
print(f"\nCosine kernel still reproduces raw-classical exactly at every qubit count, out-of-sample: {still_exact}")

out = {
    "description": (
        "Out-of-sample rerun of the cosine-kernel column behind the "
        "48-item hard corpus qubit-sweep comparison (mult=1.2, "
        "n_qubits 4-14). The published cosine-kernel values (all 1.000) "
        "came from scratch_pca_vector_classical_baseline.py's "
        "qubit_sweep_48 block, which fits PCA leakily. raw_classical is "
        "recomputed fresh here too, purely as a sanity check."
    ),
    "source_of_leaky_baseline": "scratch_pca_vector_classical_baseline_results.pkl['qubit_sweep_48']",
    "mult": MULT,
    "n_seeds": N_SEEDS,
    "n_queries": N_QUERIES,
    "raw_classical_cross_check_all_match": all_match,
    "cosine_kernel_still_exact_match": still_exact,
    "results": results,
}

with open("scratch_qubitsweep48_crossover_oos_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE -- saved scratch_qubitsweep48_crossover_oos_results.pkl")
