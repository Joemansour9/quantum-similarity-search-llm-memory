"""
scratch_verify_pca_vector_crossover_oos.py — out-of-sample (leak-free)
rerun of the cosine-kernel column in Table pca-vector-crossover (480-item
corpus, noise multiplier 1.2, n_qubits 4-14).

The existing cosine-kernel column comes directly from
scratch_pca_vector_classical_baseline.py's classical_pca_recall1(), which
fits PCA on full_batch = vstack([query, candidate_batch]) -- leaky,
confirmed by reading that script directly. This rerun changes only the
fitting step: R is fit on candidate_batch alone (query excluded), same
corpus, same seed, same noise, same n_qubits sweep, same n=20/cell
(rng=default_rng(1), q_idx 0-19) as the original.

classical_raw (raw-embedding cosine, no projection involved) is also
recomputed fresh here as a sanity check -- it should reproduce 1.000 at
every qubit count exactly, since it never touches PCA fitting.
"""
import pickle, time
import numpy as np
from q_encoder import fit_pca_projection_scale
from q_search import cosine_similarity, recall_at_k

with open("scratch_large_corpus_480.pkl", "rb") as f:
    d480 = pickle.load(f)
corpus480, keys480 = d480["corpus"], d480["keys"]
d_dim480 = next(iter(corpus480.values())).shape[0]
candidate_batch480 = np.stack([corpus480[k] for k in keys480])

CORPUS_STD = 0.04882
MULT = 1.2
noise = MULT * CORPUS_STD
N_QUERIES = 20

# Published (leaky) values, for the before/after record
PUBLISHED_COSINE_KERNEL = {4: 0.200, 6: 0.450, 8: 0.900, 10: 1.000, 12: 1.000, 14: 1.000}
PUBLISHED_RAW_CLASSICAL = {nq: 1.000 for nq in (4, 6, 8, 10, 12, 14)}


def cosine_kernel_recall1_oos(query_vec, candidate_batch, keys, relevant_idx, n_qubits):
    # OUT-OF-SAMPLE: fit on candidate_batch only, query excluded
    R, _, _ = fit_pca_projection_scale(candidate_batch, n_qubits)
    query_proj = R @ query_vec
    candidates_proj = candidate_batch @ R.T
    cosines = np.array([cosine_similarity(query_proj, candidates_proj[i]) for i in range(len(keys))])
    rank = np.argsort(-cosines)
    return recall_at_k(rank, relevant_idx, 1)


print("=== Out-of-sample cosine-kernel rerun, 480-item corpus, mult=1.2, n=20/cell ===")
results = {}
for n_qubits in (4, 6, 8, 10, 12, 14):
    t0 = time.perf_counter()
    rng = np.random.default_rng(1)
    cosine_recalls = []
    raw_recalls = []
    for q_idx in range(N_QUERIES):
        relevant_key = keys480[q_idx % len(keys480)]
        relevant_idx = keys480.index(relevant_key)
        query_vec = corpus480[relevant_key] + rng.normal(scale=noise, size=d_dim480)

        cosine_recalls.append(
            cosine_kernel_recall1_oos(query_vec, candidate_batch480, keys480, relevant_idx, n_qubits)
        )

        raw_cos = np.array([cosine_similarity(query_vec, candidate_batch480[i]) for i in range(len(keys480))])
        raw_rank = np.argsort(-raw_cos)
        raw_recalls.append(recall_at_k(raw_rank, relevant_idx, 1))

    elapsed = time.perf_counter() - t0
    cosine_kernel = float(np.mean(cosine_recalls))
    raw_classical = float(np.mean(raw_recalls))
    dev = abs(cosine_kernel - raw_classical)
    results[n_qubits] = {"cosine_kernel_oos": cosine_kernel, "raw_classical": raw_classical, "deviation": dev}
    print(f"  n_qubits={n_qubits:2d}  cosine_kernel(OOS)={cosine_kernel:.3f}  "
          f"raw_classical={raw_classical:.3f}  |dev|={dev:.3f}  ({elapsed:.1f}s)")

print("\n=== Sanity check: fresh raw_classical vs. published (should be 1.000 everywhere) ===")
all_match = True
for nq in (4, 6, 8, 10, 12, 14):
    fresh = results[nq]["raw_classical"]
    published = PUBLISHED_RAW_CLASSICAL[nq]
    ok = abs(fresh - published) < 1e-9
    all_match = all_match and ok
    print(f"  n_qubits={nq:2d}  fresh={fresh:.3f}  published={published:.3f}  match={ok}")
print(f"raw_classical matches published exactly at every qubit count: {all_match}")

print("\n=== Before / after: cosine kernel, leaky (published) vs. out-of-sample (this run) ===")
for nq in (4, 6, 8, 10, 12, 14):
    leaky = PUBLISHED_COSINE_KERNEL[nq]
    oos = results[nq]["cosine_kernel_oos"]
    print(f"  n_qubits={nq:2d}  leaky(published)={leaky:.3f}  out-of-sample={oos:.3f}  delta={oos-leaky:+.3f}")

out = {
    "description": (
        "Out-of-sample (leak-free) rerun of Table pca-vector-crossover's "
        "cosine-kernel column (480-item corpus, mult=1.2, n_qubits 4-14, "
        "n=20/cell). The published cosine-kernel column came from "
        "scratch_pca_vector_classical_baseline.py's classical_pca_recall1(), "
        "which fits PCA on a query-included batch (leaky) -- confirmed by "
        "reading that script directly before this rerun. raw_classical is "
        "recomputed fresh here too, purely as a sanity check (it does not "
        "involve PCA fitting and should -- and does -- match the published "
        "1.000 constant at every qubit count)."
    ),
    "source_of_leaky_baseline": "scratch_pca_vector_classical_baseline.py / scratch_pca_vector_classical_baseline_results.pkl['qubit_sweep_480']",
    "corpus": "scratch_large_corpus_480.pkl (480-item hard corpus)",
    "mult": MULT,
    "n_queries": N_QUERIES,
    "raw_classical_cross_check_all_match": all_match,
    "results": results,
}

with open("scratch_pca_vector_crossover_oos_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE -- saved scratch_pca_vector_crossover_oos_results.pkl")
