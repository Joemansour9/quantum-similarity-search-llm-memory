"""
scratch_three_baselines_qubit_sweep_hard_v2.py

v2: uses make_hard_real_corpus / CORPUS_PER_DIM_STD directly from the
CURRENT benchmark_realdata_embeddings_hard.py (pool=65), matching
scratch_verify_pca_leakage_remaining.py's exact methodology -- NOT a
pool=60 historical reconstruction. This matters: the paper's currently
published Table qubit-hard numbers turn out to already be the leak-free
(out-of-sample) numbers from that script, which itself used pool=65 (the
code's current default at the time it ran), not the older pool=60 corpus
that produced the ORIGINAL leaky Section 5.1 prose. v1 of this script
(scratch_three_baselines_qubit_sweep_hard.py) incorrectly reconstructed
pool=60 and did not reproduce the published table -- archived at
repo/audit_history/investigation/other_checks/ (with its own non-matching
output) as a record of the wrong assumption, superseded by this version.

First validates by recomputing leak-free PCA/random and checking it
reproduces the published table exactly, THEN computes the three new
baselines (svd, whiten, firstn) under the identical setup.
"""
import time
import pickle
import numpy as np
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k

MULT = 1.2
N_SEEDS = 5
N_QUERIES = 10
QUBIT_COUNTS = [4, 6, 8, 10, 12, 14]
noise = MULT * CORPUS_PER_DIM_STD

# Published Table qubit-hard (paper) -- for the validation check
PUBLISHED_PCA = {4: 0.660, 6: 0.920, 8: 0.940, 10: 1.000, 12: 1.000, 14: 1.000}
PUBLISHED_RANDOM = {4: 0.180, 6: 0.120, 8: 0.220, 10: 0.380, 12: 0.380, 14: 0.420}


def fit_svd_projection_scale(batch: np.ndarray, n_qubits: int):
    d = batch.shape[1]
    _, _, Vt = np.linalg.svd(batch, full_matrices=False)  # no centering
    R = Vt[:n_qubits]
    projected = batch @ R.T
    return R, projected.min(), projected.max()


def fit_whitened_pca_projection_scale(batch: np.ndarray, n_qubits: int):
    R_pca, _, _ = fit_pca_projection_scale(batch, n_qubits)
    proj = batch @ R_pca.T
    stds = proj.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    R = R_pca / stds[:, None]
    projected = batch @ R.T
    return R, projected.min(), projected.max()


def fit_firstn_projection_scale(batch: np.ndarray, n_qubits: int):
    d = batch.shape[1]
    R = np.eye(n_qubits, d)
    projected = batch @ R.T
    return R, projected.min(), projected.max()


METHODS = {
    "random": lambda batch, nq, seed: fit_projection_scale(batch, nq, seed=seed),
    "pca": lambda batch, nq, seed: fit_pca_projection_scale(batch, nq),
    "svd": lambda batch, nq, seed: fit_svd_projection_scale(batch, nq),
    "whiten": lambda batch, nq, seed: fit_whitened_pca_projection_scale(batch, nq),
    "firstn": lambda batch, nq, seed: fit_firstn_projection_scale(batch, nq),
}

print(f"corpus_per_dim_std={CORPUS_PER_DIM_STD:.5f} (pool={65} -- current code default)")
print(f"=== Out-of-sample qubit-count sweep, 48-item hard corpus ===")
print(f"(mult={MULT}, {N_SEEDS} seeds x {N_QUERIES} queries = {N_SEEDS*N_QUERIES} query-runs/cell)\n")

results = {}
for n_qubits in QUBIT_COUNTS:
    edges = KnowledgeGraph.chain(n_qubits).edges()
    for method_name, fit_fn in METHODS.items():
        t0 = time.perf_counter()
        recalls = []
        for seed in range(N_SEEDS):
            corpus, keys = make_hard_real_corpus(48, 8, seed=seed)
            d_dim = next(iter(corpus.values())).shape[0]
            candidate_batch = np.stack([corpus[k] for k in keys]).astype(np.float64)
            rng = np.random.default_rng(seed + 1)
            for q_idx in range(N_QUERIES):
                relevant_key = keys[q_idx % len(keys)]
                relevant_idx = keys.index(relevant_key)
                query_vec = corpus[relevant_key].astype(np.float64) + rng.normal(scale=noise, size=d_dim)

                R, lo, hi = fit_fn(candidate_batch, n_qubits, q_idx)  # OUT-OF-SAMPLE: query excluded
                scale = (lo, hi)

                qc_query = build_feature_map(query_vec, n_qubits, scale, edges, R, entangler="rzz")
                fids = np.zeros(len(keys))
                for i in range(len(keys)):
                    fid, _ = statevector_fidelity_from_circuits(
                        qc_query, candidate_batch[i], n_qubits, scale, edges, R, entangler="rzz"
                    )
                    fids[i] = fid
                rank = np.argsort(-fids)
                recalls.append(recall_at_k(rank, relevant_idx, 1))
        elapsed = time.perf_counter() - t0
        mean_recall = float(np.mean(recalls))
        results[(n_qubits, method_name)] = mean_recall
        print(f"  n_qubits={n_qubits:2d}  {method_name:7s}  Recall@1={mean_recall:.3f}  ({elapsed:.1f}s)")

print("\n=== Validation: does recomputed PCA/random match the published table? ===")
print(f"{'n_qubits':>8}  {'PCA (reproduced)':>16}  {'PCA (paper)':>11}  {'match':>6}  "
      f"{'random (reproduced)':>20}  {'random (paper)':>14}  {'match':>6}")
all_match = True
for nq in QUBIT_COUNTS:
    p_reproduced = results[(nq, "pca")]
    p_paper = PUBLISHED_PCA[nq]
    r_reproduced = results[(nq, "random")]
    r_paper = PUBLISHED_RANDOM[nq]
    p_ok = abs(p_reproduced - p_paper) < 1e-9
    r_ok = abs(r_reproduced - r_paper) < 1e-9
    all_match = all_match and p_ok and r_ok
    print(f"{nq:>8}  {p_reproduced:>16.3f}  {p_paper:>11.3f}  {str(p_ok):>6}  "
          f"{r_reproduced:>20.3f}  {r_paper:>14.3f}  {str(r_ok):>6}")
print(f"\nALL MATCH: {all_match}")

with open("scratch_three_baselines_qubit_sweep_hard_v2_results.pkl", "wb") as f:
    pickle.dump({
        "results": results,
        "corpus_per_dim_std": CORPUS_PER_DIM_STD,
        "mult": MULT,
        "n_seeds": N_SEEDS,
        "n_queries": N_QUERIES,
        "qubit_counts": QUBIT_COUNTS,
        "all_match_published": all_match,
    }, f)

print("\nDONE -- saved scratch_three_baselines_qubit_sweep_hard_v2_results.pkl")
