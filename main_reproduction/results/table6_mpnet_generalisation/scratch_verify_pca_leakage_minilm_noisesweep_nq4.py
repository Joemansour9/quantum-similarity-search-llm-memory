"""
scratch_verify_pca_leakage_minilm_noisesweep_nq4.py — MiniLM 48-item hard
corpus noise sweep at n_qubits=4, the OTHER qubit count from the paper's
main tab:noise-hard (which is at n_qubits=8), so the mpnet-vs-MiniLM
embedding-generalization comparison can be checked at both qubit counts
rather than picking one arbitrarily.

Identical to scratch_verify_pca_leakage_minilm_noisesweep_nq8.py in every
respect (make_hard_real_corpus pool=65, n_seeds=5 x n_queries=10 = 50/cell,
same 10 multipliers, same leaky/leak-free-both-computed convention) except
n_qubits=4 instead of 8.
"""
import time
import numpy as np
import pickle
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k, cosine_similarity
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD

n_qubits = 4
edges = KnowledgeGraph.chain(n_qubits).edges()
N_SEEDS = 5
N_QUERIES = 10
NOISE_MULTIPLIERS = (0.2, 1.2, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)

print(f"=== MiniLM 48-item hard corpus: full noise sweep, n_qubits=4, n_seeds=5 x n_queries=10 ===")
print(f"(corpus_std={CORPUS_PER_DIM_STD:.5f})")
results = {}
for mult in NOISE_MULTIPLIERS:
    noise = mult * CORPUS_PER_DIM_STD
    row = {}
    for proj in ("pca", "random"):
        for mode in ("leaky", "leak_free"):
            t0 = time.perf_counter()
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
            row[(proj, mode)] = float(np.mean(recalls))
            print(f"  mult={mult:>5.1f}  {proj:6s}  {mode:9s}  recall@1={row[(proj, mode)]:.3f}  ({elapsed:.1f}s)")

    classical_recalls = []
    for seed in range(N_SEEDS):
        corpus, keys = make_hard_real_corpus(48, 8, seed=seed)
        d_dim = next(iter(corpus.values())).shape[0]
        candidate_batch = np.stack([corpus[k] for k in keys])
        rng = np.random.default_rng(seed + 1)
        for q_idx in range(N_QUERIES):
            relevant_key = keys[q_idx % len(keys)]
            relevant_idx = keys.index(relevant_key)
            query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
            cos = np.array([cosine_similarity(query_vec, candidate_batch[i]) for i in range(len(keys))])
            rank = np.argsort(-cos)
            classical_recalls.append(recall_at_k(rank, relevant_idx, 1))
    row["classical"] = float(np.mean(classical_recalls))
    results[mult] = row
    print(f"  mult={mult:>5.1f}  classical={row['classical']:.3f}")

print("\n--- MiniLM n_qubits=4 noise sweep: leaky vs leak-free, all 10 multipliers ---")
print(f"{'mult':>6}  {'pca leaky':>10}  {'pca leak-free':>14}  {'pca delta':>10}  "
      f"{'random leaky':>13}  {'random leak-free':>17}  {'classical':>10}")
for mult in NOISE_MULTIPLIERS:
    row = results[mult]
    print(f"{mult:>6.1f}  {row[('pca','leaky')]:>10.3f}  {row[('pca','leak_free')]:>14.3f}  "
          f"{row[('pca','leak_free')]-row[('pca','leaky')]:>+10.3f}  "
          f"{row[('random','leaky')]:>13.3f}  {row[('random','leak_free')]:>17.3f}  {row['classical']:>10.3f}")

dev_pca_leaky, dev_pca_lf, dev_rand = [], [], []
for mult in NOISE_MULTIPLIERS:
    row = results[mult]
    cls = row["classical"]
    dev_pca_leaky.append(abs(row[("pca","leaky")] - cls))
    dev_pca_lf.append(abs(row[("pca","leak_free")] - cls))
    dev_rand.append(abs(row[("random","leak_free")] - cls))

mean_pca_leaky = np.mean(dev_pca_leaky)
mean_pca_lf = np.mean(dev_pca_lf)
mean_rand = np.mean(dev_rand)
print(f"\nMean |PCA-classical| deviation, LEAKY (fresh recompute):    {mean_pca_leaky:.4f}")
print(f"Mean |PCA-classical| deviation, LEAK-FREE (corrected):      {mean_pca_lf:.4f}")
print(f"Mean |random-classical| deviation:                           {mean_rand:.4f}")
print(f"\nRatio (random/PCA), LEAKY:      {mean_rand/mean_pca_leaky:.2f}x")
print(f"Ratio (random/PCA), LEAK-FREE:  {mean_rand/mean_pca_lf:.2f}x")

with open("scratch_pca_leakage_minilm_noisesweep_nq4_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nDONE")
