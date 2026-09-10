"""
scratch_check_underpowered_mult6_n500.py — Check 3: is the mult=6.0/480-item
"genuinely borderline" flag (large_corpus_generalization_2026-07-30.md, n=100,
gap=0.09, overlapping CIs) an artifact of insufficient n, using the ORIGINAL
(leaky) methodology exactly as published -- not the leak-free correction,
which is a separate axis already resolved elsewhere. n=500/cell.
"""
import time
import numpy as np
import pickle
from q_encoder import fit_pca_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k, cosine_similarity

with open("scratch_large_corpus_480.pkl", "rb") as f:
    d = pickle.load(f)
corpus, keys = d["corpus"], d["keys"]
d_dim = next(iter(corpus.values())).shape[0]
candidate_batch = np.stack([corpus[k] for k in keys])
CORPUS_STD = 0.04882
n_qubits = 4
edges = KnowledgeGraph.chain(n_qubits).edges()
mult = 6.0
noise = mult * CORPUS_STD
N_QUEUE = 500

def wald_ci(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)

t0 = time.perf_counter()
rng = np.random.default_rng(1)
pca_recalls, classical_recalls = [], []
for q_idx in range(N_QUEUE):
    relevant_key = keys[q_idx % len(keys)]
    relevant_idx = keys.index(relevant_key)
    query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
    full_batch = np.vstack([query_vec[None, :], candidate_batch])  # leaky, matches original exactly
    R, lo, hi = fit_pca_projection_scale(full_batch, n_qubits)
    scale = (lo, hi)
    qc_query = build_feature_map(query_vec, n_qubits, scale, edges, R, entangler="rzz")
    fids = np.zeros(len(keys))
    for i in range(len(keys)):
        fid, _ = statevector_fidelity_from_circuits(qc_query, candidate_batch[i], n_qubits, scale, edges, R, entangler="rzz")
        fids[i] = fid
    rank = np.argsort(-fids)
    pca_recalls.append(recall_at_k(rank, relevant_idx, 1))

    cos = np.array([cosine_similarity(query_vec, candidate_batch[i]) for i in range(len(keys))])
    rank_c = np.argsort(-cos)
    classical_recalls.append(recall_at_k(rank_c, relevant_idx, 1))
    if (q_idx + 1) % 100 == 0:
        print(f"  ... {q_idx+1}/{N_QUEUE} done ({time.perf_counter()-t0:.0f}s elapsed)")

p_pca = float(np.mean(pca_recalls))
p_cls = float(np.mean(classical_recalls))
ci_pca = wald_ci(p_pca, N_QUEUE)
ci_cls = wald_ci(p_cls, N_QUEUE)
gap = p_cls - p_pca
lo_p, hi_p = p_pca - ci_pca, p_pca + ci_pca
lo_c, hi_c = p_cls - ci_cls, p_cls + ci_cls
overlap = not (hi_p < lo_c or hi_c < lo_p)

print(f"\n=== mult=6.0, 480-item corpus, n_qubits=4, LEAKY (original methodology), n=500 ===")
print(f"  Published (n=100): PCA=0.560+/-0.097  Classical=0.650+/-0.093  gap=0.09  OVERLAPPING")
print(f"  This run  (n=500): PCA={p_pca:.3f}+/-{ci_pca:.3f}  Classical={p_cls:.3f}+/-{ci_cls:.3f}  gap={gap:.3f}  "
      f"{'OVERLAPPING' if overlap else 'NON-OVERLAPPING'}")

with open("scratch_underpowered_mult6_n500_results.pkl", "wb") as f:
    pickle.dump({"pca": (p_pca, ci_pca), "classical": (p_cls, ci_cls), "overlap": overlap}, f)

print("\nDONE")
