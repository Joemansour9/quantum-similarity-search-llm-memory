"""
scratch_verify_pca_leakage_480_noisesweep.py — full leak-free correction for
the 480-item corpus's query-noise sweep at n_qubits=4 (large_corpus_generalization_2026-07-30.md
Result 2's initial n=20/cell pass, all 10 multipliers -- distinct from the
480-item QUBIT sweep already checked, which varied n_qubits at fixed mult=1.2
and found zero leakage effect there). Matches scratch_large_corpus_sweep.py's
exact noise-sweep methodology: single fixed corpus (seed=0), rng=default_rng(1)
reset per multiplier, n=20 queries/cell. Recomputes BOTH leaky and leak-free
fresh (480-item corpus is NOT affected by the pool-size change that confounds
the 48-item corpus checks, but recomputing fresh keeps this apples-to-apples
with the rest of this investigation).
"""
import time
import numpy as np
import pickle
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
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
N_QUEUE = 20
NOISE_MULTIPLIERS = (0.2, 1.2, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)

print(f"=== 480-item corpus: full noise sweep, n_qubits=4, n=20/cell ===")
results = {}
for mult in NOISE_MULTIPLIERS:
    noise = mult * CORPUS_STD
    row = {}
    for proj in ("pca", "random"):
        for mode in ("leaky", "leak_free"):
            t0 = time.perf_counter()
            rng = np.random.default_rng(1)
            recalls = []
            for q_idx in range(N_QUEUE):
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

    rng = np.random.default_rng(1)
    classical_recalls = []
    for q_idx in range(N_QUEUE):
        relevant_key = keys[q_idx % len(keys)]
        relevant_idx = keys.index(relevant_key)
        query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
        cos = np.array([cosine_similarity(query_vec, candidate_batch[i]) for i in range(len(keys))])
        rank = np.argsort(-cos)
        classical_recalls.append(recall_at_k(rank, relevant_idx, 1))
    row["classical"] = float(np.mean(classical_recalls))
    results[mult] = row
    print(f"  mult={mult:>5.1f}  classical={row['classical']:.3f}")

print("\n--- 480-item corpus noise sweep: leaky vs leak-free, all 10 multipliers ---")
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
print(f"\nMean |PCA-classical| deviation, LEAKY:      {mean_pca_leaky:.4f}")
print(f"Mean |PCA-classical| deviation, LEAK-FREE:  {mean_pca_lf:.4f}")
print(f"Mean |random-classical| deviation:           {mean_rand:.4f}")
print(f"\nRatio (random/PCA), LEAKY:      {mean_rand/mean_pca_leaky:.2f}x" if mean_pca_leaky > 0 else "n/a")
print(f"Ratio (random/PCA), LEAK-FREE:  {mean_rand/mean_pca_lf:.2f}x" if mean_pca_lf > 0 else "n/a")

with open("scratch_pca_leakage_480_noisesweep_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nDONE")
