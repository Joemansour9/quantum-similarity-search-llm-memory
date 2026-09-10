"""
scratch_check_rzz_ablation.py — Check 2: does the RZZ entangling layer
contribute to Recall@1 (not just fidelity), vs a plain product-state
(Ry-only, edges=[]) encoding? Same hard corpus, same PCA fitting convention
(query included, matching the project's actual/leaky pipeline -- ablation
is about entangler contribution, independent of the leakage question),
same noise sweep, across n_qubits=4 and n_qubits=8.
"""
import pickle
import numpy as np
from agent_memory import AgentMemory
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD
from q_encoder import fit_pca_projection_scale, build_feature_map
from q_search import statevector_fidelity_from_circuits, recall_at_k

N_QUBITS_LIST = (4, 8)
NOISE_MULTIPLIERS = (0.2, 1.2, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0)
N_SEEDS = 5
N_QUERIES = 10

print("=== RZZ entanglement ablation: Recall@1, rzz vs no-entangler (product state) ===")
print("(48-item hard corpus, PCA projection, n_qubits=4 and 8, n=50/cell)")
results = {}
for n_qubits in N_QUBITS_LIST:
    for mult in NOISE_MULTIPLIERS:
        noise = mult * CORPUS_PER_DIM_STD
        for variant, edges_override in (("rzz", None), ("none", [])):
            recalls = []
            for seed in range(N_SEEDS):
                corpus, keys = make_hard_real_corpus(48, 8, seed=seed)
                d_dim = next(iter(corpus.values())).shape[0]
                candidate_batch = np.stack([corpus[k] for k in keys])
                rng = np.random.default_rng(seed + 1)
                for q_idx in range(N_QUERIES):
                    relevant_key = keys[q_idx % len(keys)]
                    relevant_idx = keys.index(relevant_key)
                    query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
                    full_batch = np.vstack([query_vec[None, :], candidate_batch])
                    R, lo, hi = fit_pca_projection_scale(full_batch, n_qubits)
                    scale = (lo, hi)
                    edges = edges_override if edges_override is not None else [(i, i + 1) for i in range(n_qubits - 1)]
                    qc_query = build_feature_map(query_vec, n_qubits, scale, edges, R, entangler="rzz")
                    fids = np.zeros(len(keys))
                    for i in range(len(keys)):
                        fid, _ = statevector_fidelity_from_circuits(
                            qc_query, candidate_batch[i], n_qubits, scale, edges, R, entangler="rzz"
                        )
                        fids[i] = fid
                    rank = np.argsort(-fids)
                    recalls.append(recall_at_k(rank, relevant_idx, 1))
            results[(n_qubits, mult, variant)] = float(np.mean(recalls))

    print(f"\n--- n_qubits={n_qubits} ---")
    print(f"{'mult':>6}  {'rzz':>8}  {'no-entangler':>13}  {'delta':>8}")
    for mult in NOISE_MULTIPLIERS:
        r_rzz = results[(n_qubits, mult, "rzz")]
        r_none = results[(n_qubits, mult, "none")]
        print(f"{mult:>6.1f}  {r_rzz:>8.3f}  {r_none:>13.3f}  {r_none-r_rzz:>+8.3f}")

with open("scratch_rzz_ablation_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved results to scratch_rzz_ablation_results.pkl")

print("\nDONE")
