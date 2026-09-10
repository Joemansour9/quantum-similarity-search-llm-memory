"""
scratch_verify_noisesweep480_crossover_oos.py — out-of-sample rerun of
the "0.225 vs 0.350" fidelity-kernel-vs-cosine-kernel noise-sweep
comparison (480-item corpus, n_qubits=4, full noise range).

IMPORTANT FINDING, checked before computing anything else: the paper's
current 0.225 (fidelity kernel) and 0.350 (cosine kernel) figures BOTH
trace exactly to the LEAKY noise_sweep_480 block of
scratch_pca_vector_classical_baseline_results.pkl -- mean|quantum_pca -
classical_raw| = 0.225 and mean|classical_pca_vec - classical_raw| =
0.350 there, reproduced below bit-for-bit. The fidelity-kernel side was
NOT already corrected, contrary to the paper's implicit assumption --
confirmed explicitly rather than assumed. Both kernels are
therefore recomputed out-of-sample here, not just the cosine kernel, so
the comparison is apples-to-apples rather than trading one leaky number
for a mix of corrected-and-leaky.

Methodology mirrors scratch_pca_vector_classical_baseline.py's
noise_sweep_480 block exactly: n_qubits=4, chain topology, 480-item
corpus, n=20/cell (rng=default_rng(1), q_idx 0-19), same 10 multipliers.
Only the fitting step changes: R fit on candidate_batch alone (query
excluded) for both kernels.
"""
import pickle, time
import numpy as np
from q_encoder import fit_pca_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, cosine_similarity, recall_at_k

with open("scratch_large_corpus_480.pkl", "rb") as f:
    d480 = pickle.load(f)
corpus480, keys480 = d480["corpus"], d480["keys"]
d_dim480 = next(iter(corpus480.values())).shape[0]
candidate_batch480 = np.stack([corpus480[k] for k in keys480])

CORPUS_STD = 0.04882
N_QUBITS = 4
edges = KnowledgeGraph.chain(N_QUBITS).edges()
NOISE_MULTIPLIERS = (0.2, 1.2, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
N_QUERIES = 20

# Published (leaky) reference, from scratch_pca_vector_classical_baseline_results.pkl['noise_sweep_480']
PUBLISHED = {
    0.2: {"quantum_pca": 1.00, "classical_raw": 1.00, "classical_pca_vec": 0.95},
    1.2: {"quantum_pca": 0.25, "classical_raw": 1.00, "classical_pca_vec": 0.20},
    2.0: {"quantum_pca": 0.15, "classical_raw": 1.00, "classical_pca_vec": 0.10},
    4.0: {"quantum_pca": 0.45, "classical_raw": 0.95, "classical_pca_vec": 0.00},
    6.0: {"quantum_pca": 0.40, "classical_raw": 0.55, "classical_pca_vec": 0.00},
    8.0: {"quantum_pca": 0.15, "classical_raw": 0.15, "classical_pca_vec": 0.00},
    10.0: {"quantum_pca": 0.05, "classical_raw": 0.05, "classical_pca_vec": 0.00},
    12.0: {"quantum_pca": 0.05, "classical_raw": 0.05, "classical_pca_vec": 0.00},
    16.0: {"quantum_pca": 0.00, "classical_raw": 0.00, "classical_pca_vec": 0.00},
    20.0: {"quantum_pca": 0.00, "classical_raw": 0.00, "classical_pca_vec": 0.00},
}

print("=== Step 0: confirm which side(s) of the 0.225/0.350 claim are leaky ===")
pub_dev_fid = [abs(PUBLISHED[m]["quantum_pca"] - PUBLISHED[m]["classical_raw"]) for m in NOISE_MULTIPLIERS]
pub_dev_cos = [abs(PUBLISHED[m]["classical_pca_vec"] - PUBLISHED[m]["classical_raw"]) for m in NOISE_MULTIPLIERS]
print(f"  mean|leaky fidelity kernel - raw| = {np.mean(pub_dev_fid):.3f}  (paper claims 0.225)")
print(f"  mean|leaky cosine kernel - raw|   = {np.mean(pub_dev_cos):.3f}  (paper claims 0.350)")
print("  => BOTH sides confirmed leaky. Recomputing both out-of-sample below.\n")


def fidelity_kernel_recall1_oos(query_vec, candidate_batch, keys, relevant_idx, n_qubits):
    R, lo, hi = fit_pca_projection_scale(candidate_batch, n_qubits)  # OUT-OF-SAMPLE
    qc = build_feature_map(query_vec, n_qubits, (lo, hi), edges, R, entangler="rzz")
    fids = np.zeros(len(keys))
    for i in range(len(keys)):
        fid, _ = statevector_fidelity_from_circuits(qc, candidate_batch[i], n_qubits, (lo, hi), edges, R, entangler="rzz")
        fids[i] = fid
    rank = np.argsort(-fids)
    return recall_at_k(rank, relevant_idx, 1)


def cosine_kernel_recall1_oos(query_vec, candidate_batch, keys, relevant_idx, n_qubits):
    R, _, _ = fit_pca_projection_scale(candidate_batch, n_qubits)  # OUT-OF-SAMPLE
    query_proj = R @ query_vec
    candidates_proj = candidate_batch @ R.T
    cosines = np.array([cosine_similarity(query_proj, candidates_proj[i]) for i in range(len(keys))])
    rank = np.argsort(-cosines)
    return recall_at_k(rank, relevant_idx, 1)


print("=== Out-of-sample rerun: 480-item corpus, n_qubits=4, n=20/cell ===")
results = {}
for mult in NOISE_MULTIPLIERS:
    noise = mult * CORPUS_STD
    t0 = time.perf_counter()
    rng = np.random.default_rng(1)
    fid_recalls, cos_recalls, raw_recalls = [], [], []
    for q_idx in range(N_QUERIES):
        relevant_key = keys480[q_idx % len(keys480)]
        relevant_idx = keys480.index(relevant_key)
        query_vec = corpus480[relevant_key] + rng.normal(scale=noise, size=d_dim480)

        fid_recalls.append(fidelity_kernel_recall1_oos(query_vec, candidate_batch480, keys480, relevant_idx, N_QUBITS))
        cos_recalls.append(cosine_kernel_recall1_oos(query_vec, candidate_batch480, keys480, relevant_idx, N_QUBITS))

        raw_cos = np.array([cosine_similarity(query_vec, candidate_batch480[i]) for i in range(len(keys480))])
        raw_rank = np.argsort(-raw_cos)
        raw_recalls.append(recall_at_k(raw_rank, relevant_idx, 1))

    elapsed = time.perf_counter() - t0
    fid_kernel = float(np.mean(fid_recalls))
    cos_kernel = float(np.mean(cos_recalls))
    raw_classical = float(np.mean(raw_recalls))
    results[mult] = {"fidelity_kernel_oos": fid_kernel, "cosine_kernel_oos": cos_kernel, "raw_classical": raw_classical}
    print(f"  mult={mult:>5.1f}  fidelity_kernel(OOS)={fid_kernel:.3f}  cosine_kernel(OOS)={cos_kernel:.3f}  "
          f"raw_classical={raw_classical:.3f}  ({elapsed:.1f}s)")

print("\n=== Sanity check: fresh raw_classical vs. published (should match exactly) ===")
all_match = True
for m in NOISE_MULTIPLIERS:
    fresh = results[m]["raw_classical"]
    published = PUBLISHED[m]["classical_raw"]
    ok = abs(fresh - published) < 1e-9
    all_match = all_match and ok
    print(f"  mult={m:>5.1f}  fresh={fresh:.3f}  published={published:.3f}  match={ok}")
print(f"raw_classical matches published exactly at every multiplier: {all_match}")

dev_fid_oos = [abs(results[m]["fidelity_kernel_oos"] - results[m]["raw_classical"]) for m in NOISE_MULTIPLIERS]
dev_cos_oos = [abs(results[m]["cosine_kernel_oos"] - results[m]["raw_classical"]) for m in NOISE_MULTIPLIERS]
mean_dev_fid_oos = float(np.mean(dev_fid_oos))
mean_dev_cos_oos = float(np.mean(dev_cos_oos))

print("\n=== Before / after: mean deviation across full noise sweep ===")
print(f"  Fidelity kernel:  leaky(published)=0.225   out-of-sample={mean_dev_fid_oos:.3f}")
print(f"  Cosine kernel:    leaky(published)=0.350   out-of-sample={mean_dev_cos_oos:.3f}")
closer = "fidelity kernel" if mean_dev_fid_oos < mean_dev_cos_oos else "cosine kernel"
print(f"  Closer tracker of raw-classical, out-of-sample: {closer}")

out = {
    "description": (
        "Out-of-sample rerun of the 480-item, n_qubits=4 noise-sweep "
        "fidelity-kernel-vs-cosine-kernel comparison (paper's '0.225 vs "
        "0.350' claim). Both the published fidelity-kernel (0.225) and "
        "cosine-kernel (0.350) mean deviations were confirmed to trace "
        "to the LEAKY noise_sweep_480 block of "
        "scratch_pca_vector_classical_baseline_results.pkl before this "
        "rerun -- the fidelity-kernel side was NOT already corrected, "
        "contrary to an initial assumption, so both kernels are "
        "recomputed out-of-sample here for an apples-to-apples "
        "comparison. raw_classical is recomputed fresh too as a sanity "
        "check against the published values."
    ),
    "source_of_leaky_baseline": "scratch_pca_vector_classical_baseline_results.pkl['noise_sweep_480']",
    "n_qubits": N_QUBITS,
    "n_queries": N_QUERIES,
    "corpus_std": CORPUS_STD,
    "raw_classical_cross_check_all_match": all_match,
    "mean_deviation": {
        "fidelity_kernel": {"leaky_published": 0.225, "out_of_sample": mean_dev_fid_oos},
        "cosine_kernel": {"leaky_published": 0.350, "out_of_sample": mean_dev_cos_oos},
    },
    "results": results,
}

with open("scratch_noisesweep480_crossover_oos_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE -- saved scratch_noisesweep480_crossover_oos_results.pkl")
