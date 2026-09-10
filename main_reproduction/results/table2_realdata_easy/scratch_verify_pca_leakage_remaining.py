"""
scratch_verify_pca_leakage_remaining.py — extends scratch_verify_pca_leakage.py's
leak-free-vs-leaky qubit-sweep check to the three sources not yet covered:
  1. Synthetic corpus (benchmark_qubit_sweep.py / Table 1)
  2. Real-easy corpus (benchmark_qubit_sweep_realdata.py / Table 2)
  3. Real-hard 48-item corpus (benchmark_qubit_sweep_realdata_hard.py / Section 5.1 prose)

For each, recomputes BOTH the leaky (query included in PCA/random fit batch --
matching the project's actual AgentMemory-based pipeline exactly) and leak-free
(fit on candidate_batch only, query excluded) recall@1, using each source's own
n_seeds/n_queries/query-noise convention, so leaky vs leak-free are perfectly
apples-to-apples within this one run (rather than trusting old archived numbers,
which for the 48-item hard corpus specifically are known to predate a pool-size
change and would not match exactly for reasons unrelated to leakage).
"""
import time
import numpy as np
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k

N_QUBITS_LIST = (4, 6, 8, 10, 12, 14)


def sweep_leaky_vs_leakfree(corpus_fn, query_noise, n_seeds, n_queries, label):
    print(f"\n=== {label} ===")
    results = {}
    for n_qubits in N_QUBITS_LIST:
        edges = KnowledgeGraph.chain(n_qubits).edges()
        for proj in ("pca", "random"):
            for mode in ("leaky", "leak_free"):
                t0 = time.perf_counter()
                recalls = []
                for seed in range(n_seeds):
                    corpus, keys = corpus_fn(seed)
                    d_dim = next(iter(corpus.values())).shape[0]
                    candidate_batch = np.stack([corpus[k] for k in keys])
                    rng = np.random.default_rng(seed + 1)
                    for q_idx in range(n_queries):
                        relevant_key = keys[q_idx % len(keys)]
                        relevant_idx = keys.index(relevant_key)
                        query_vec = corpus[relevant_key] + rng.normal(scale=query_noise, size=d_dim)

                        if mode == "leaky":
                            fit_batch = np.vstack([query_vec[None, :], candidate_batch])
                        else:
                            fit_batch = candidate_batch

                        if proj == "pca":
                            R, lo, hi = fit_pca_projection_scale(fit_batch, n_qubits)
                        else:
                            R, lo, hi = fit_projection_scale(fit_batch, n_qubits, seed=q_idx)
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
                results[(n_qubits, proj, mode)] = mean_recall
                print(f"  n_qubits={n_qubits:2d}  {proj:6s}  {mode:9s}  recall@1={mean_recall:.3f}  ({elapsed:.1f}s)")

    print(f"\n--- {label}: leaky vs leak-free deltas ---")
    print(f"{'n_qubits':>8}  {'PCA leaky':>10}  {'PCA leak-free':>14}  {'PCA delta':>10}  "
          f"{'gap leaky':>10}  {'gap leak-free':>14}  {'gap delta':>10}")
    for n_qubits in N_QUBITS_LIST:
        pca_leaky = results[(n_qubits, "pca", "leaky")]
        pca_lf = results[(n_qubits, "pca", "leak_free")]
        rand_leaky = results[(n_qubits, "random", "leaky")]
        rand_lf = results[(n_qubits, "random", "leak_free")]
        gap_leaky = pca_leaky - rand_leaky
        gap_lf = pca_lf - rand_lf
        print(f"{n_qubits:>8}  {pca_leaky:>10.3f}  {pca_lf:>14.3f}  {pca_lf-pca_leaky:>+10.3f}  "
              f"{gap_leaky:>10.3f}  {gap_lf:>14.3f}  {gap_lf-gap_leaky:>+10.3f}")
    return results


all_results = {}

# --- 1. Synthetic corpus ---
from benchmark_projection import make_synthetic_corpus
all_results["synthetic"] = sweep_leaky_vs_leakfree(
    corpus_fn=lambda seed: make_synthetic_corpus(24, 32, 8, seed=seed, cluster_spread=2.0),
    query_noise=1.2, n_seeds=5, n_queries=10,
    label="Synthetic corpus (Table 1): 24 items, d=32, 8 clusters, query_noise=1.2 (absolute)",
)

# --- 2. Real-easy corpus ---
from benchmark_realdata_embeddings import make_real_corpus, CORPUS_PER_DIM_STD as EASY_STD
all_results["real_easy"] = sweep_leaky_vs_leakfree(
    corpus_fn=lambda seed: make_real_corpus(24, 8, seed=seed),
    query_noise=1.2 * EASY_STD, n_seeds=5, n_queries=10,
    label=f"Real-easy corpus (Table 2): 24 items, 8 topics, MiniLM, mult=1.2 (std={EASY_STD:.5f})",
)

# --- 3. Real-hard 48-item corpus ---
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD as HARD_STD
all_results["real_hard_48"] = sweep_leaky_vs_leakfree(
    corpus_fn=lambda seed: make_hard_real_corpus(48, 8, seed=seed),
    query_noise=1.2 * HARD_STD, n_seeds=5, n_queries=10,
    label=f"Real-hard 48-item corpus (Sec 5.1 prose): 48 items, 8 topics, MiniLM, mult=1.2 (std={HARD_STD:.5f})",
)

import pickle
with open("scratch_pca_leakage_remaining_results.pkl", "wb") as f:
    pickle.dump(all_results, f)

print("\nALL DONE")
