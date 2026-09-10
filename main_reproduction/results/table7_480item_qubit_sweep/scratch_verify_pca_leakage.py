"""
scratch_verify_pca_leakage.py — Task 2 (co-author feedback): rerun the
480-item qubit sweep (Result 1's flagship setting, mirroring
scratch_large_corpus_sweep.py exactly: mult=1.2, n_qubits sweep, n=20/cell,
rng=default_rng(1)) with the PCA/random projection fit EXCLUDING the query
vector -- i.e. fit once on candidate_batch alone, then project the
held-out query into that fixed space via project_to_n_features(query, ...,
scale, R). This is the "no leakage" condition. Compared directly against
the already-archived LEAKY numbers in scratch_large_corpus_qubit_sweep.pkl
(computed via the project's actual AgentMemory pipeline, which always
includes the query in the fit batch -- confirmed by code trace).
"""
import pickle, time
import numpy as np
from q_encoder import fit_pca_projection_scale, fit_projection_scale, project_to_n_features, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k

with open("scratch_large_corpus_480.pkl", "rb") as f:
    d = pickle.load(f)
corpus, keys = d["corpus"], d["keys"]
d_dim = next(iter(corpus.values())).shape[0]
candidate_batch = np.stack([corpus[k] for k in keys])
CORPUS_STD = 0.04882
MULT = 1.2
noise = MULT * CORPUS_STD
N_QUEUE = 20

with open("scratch_large_corpus_qubit_sweep.pkl", "rb") as f:
    leaky_archived = pickle.load(f)

print("=== Leak-free qubit sweep (480-item corpus, mult=1.2, n=20/cell) ===")
print("(projection fit on candidate_batch ONLY -- query excluded from the fit)")
leak_free = {}
for n_qubits in (4, 6, 8, 10, 12, 14):
    edges = KnowledgeGraph.chain(n_qubits).edges()
    for proj in ("pca", "random"):
        t0 = time.perf_counter()
        rng = np.random.default_rng(1)
        recalls = []
        for q_idx in range(N_QUEUE):
            relevant_key = keys[q_idx % len(keys)]
            relevant_idx = keys.index(relevant_key)
            query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)

            # LEAK-FREE FIT: candidate_batch only, query never seen
            if proj == "pca":
                R, lo, hi = fit_pca_projection_scale(candidate_batch, n_qubits)
            else:
                R, lo, hi = fit_projection_scale(candidate_batch, n_qubits, seed=q_idx)
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
        leak_free[(n_qubits, proj)] = mean_recall
        leaky_val = leaky_archived[(n_qubits, proj)]
        print(f"  n_qubits={n_qubits:2d}  {proj:6s}  leak-free={mean_recall:.3f}  "
              f"leaky(archived)={leaky_val:.3f}  diff={mean_recall-leaky_val:+.3f}  ({elapsed:.1f}s)")

print("\n=== Summary: PCA-random gap, leaky vs leak-free ===")
print(f"{'n_qubits':>8}  {'leaky gap':>10}  {'leak-free gap':>14}  {'gap change':>11}")
for n_qubits in (4, 6, 8, 10, 12, 14):
    leaky_gap = leaky_archived[(n_qubits, "pca")] - leaky_archived[(n_qubits, "random")]
    lf_gap = leak_free[(n_qubits, "pca")] - leak_free[(n_qubits, "random")]
    print(f"{n_qubits:>8}  {leaky_gap:>10.3f}  {lf_gap:>14.3f}  {lf_gap-leaky_gap:>+11.3f}")

with open("scratch_pca_leakage_results.pkl", "wb") as f:
    pickle.dump({"leak_free": leak_free, "leaky_archived": leaky_archived}, f)

print("\nDONE")
