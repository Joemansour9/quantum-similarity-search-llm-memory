"""
scratch_verify_pca_leakage_mpnet_noisesweep.py — full leak-free correction
across the mpnet 48-item hard corpus's noise sweep (all 10 multipliers,
n_qubits=4, matching embedding_model_generalization_2026-07-30.md Result 2's
methodology and multiplier list exactly). Adds classical (raw-embedding
cosine) recall, which doesn't depend on PCA fitting at all and is therefore
identical whether "leaky" or "leak-free" -- computed once as the reference
point for the mean|PCA-classical| / mean|random-classical| deviation metric.
"""
import time
import numpy as np
import pickle
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k, cosine_similarity

with open("scratch_mpnet_corpus.pkl", "rb") as f:
    d = pickle.load(f)
corpus, keys = d["corpus"], d["keys"]
CORPUS_STD = d["corpus_std"]
d_dim = d["d"]
candidate_batch = np.stack([corpus[k] for k in keys]).astype(np.float64)
N_QUEUE = 50
n_qubits = 4
edges = KnowledgeGraph.chain(n_qubits).edges()

NOISE_MULTIPLIERS = (0.2, 1.2, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)

print(f"=== mpnet 48-item corpus: full noise sweep, n_qubits=4, n=50/cell ===")
print(f"(corpus_std={CORPUS_STD:.5f}; matches embedding_model_generalization_2026-07-30.md's 10-level sweep)")
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
                query_vec = corpus[relevant_key].astype(np.float64) + rng.normal(scale=noise, size=d_dim)

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

    # classical: raw-embedding cosine, independent of any projection fitting
    rng = np.random.default_rng(1)
    classical_recalls = []
    for q_idx in range(N_QUEUE):
        relevant_key = keys[q_idx % len(keys)]
        relevant_idx = keys.index(relevant_key)
        query_vec = corpus[relevant_key].astype(np.float64) + rng.normal(scale=noise, size=d_dim)
        cos = np.array([cosine_similarity(query_vec, candidate_batch[i]) for i in range(len(keys))])
        rank = np.argsort(-cos)
        classical_recalls.append(recall_at_k(rank, relevant_idx, 1))
    row["classical"] = float(np.mean(classical_recalls))

    results[mult] = row
    print(f"  mult={mult:>5.1f}  pca_leaky={row[('pca','leaky')]:.3f}  pca_leakfree={row[('pca','leak_free')]:.3f}  "
          f"random_leaky={row[('random','leaky')]:.3f}  random_leakfree={row[('random','leak_free')]:.3f}  "
          f"classical={row['classical']:.3f}")

print("\n--- mpnet noise sweep: leaky vs leak-free, all 10 multipliers ---")
print(f"{'mult':>6}  {'pca leaky':>10}  {'pca leak-free':>14}  {'random':>8}  {'classical':>10}  "
      f"{'|pca_lk-cls|':>12}  {'|pca_lf-cls|':>12}  {'|rand-cls|':>10}")
dev_pca_leaky = []
dev_pca_lf = []
dev_rand = []
for mult in NOISE_MULTIPLIERS:
    row = results[mult]
    pca_leaky = row[("pca", "leaky")]
    pca_lf = row[("pca", "leak_free")]
    rand = row[("random", "leak_free")]  # random unaffected by leakage; leak_free==leaky up to noise
    cls = row["classical"]
    d1 = abs(pca_leaky - cls)
    d2 = abs(pca_lf - cls)
    d3 = abs(rand - cls)
    dev_pca_leaky.append(d1)
    dev_pca_lf.append(d2)
    dev_rand.append(d3)
    print(f"{mult:>6.1f}  {pca_leaky:>10.3f}  {pca_lf:>14.3f}  {rand:>8.3f}  {cls:>10.3f}  "
          f"{d1:>12.3f}  {d2:>12.3f}  {d3:>10.3f}")

mean_dev_pca_leaky = np.mean(dev_pca_leaky)
mean_dev_pca_lf = np.mean(dev_pca_lf)
mean_dev_rand = np.mean(dev_rand)
print(f"\nMean |PCA-classical| deviation, LEAKY:      {mean_dev_pca_leaky:.4f}")
print(f"Mean |PCA-classical| deviation, LEAK-FREE:   {mean_dev_pca_lf:.4f}")
print(f"Mean |random-classical| deviation:            {mean_dev_rand:.4f}")
print(f"\nRatio (random/PCA), LEAKY  (original claim basis): {mean_dev_rand/mean_dev_pca_leaky:.2f}x")
print(f"Ratio (random/PCA), LEAK-FREE (corrected):         {mean_dev_rand/mean_dev_pca_lf:.2f}x")

with open("scratch_pca_leakage_mpnet_noisesweep_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nDONE")
