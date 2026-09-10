"""
scratch_verify_pca_leakage_transition_sizes.py — leak-free correction for the
transition-boundary study's six intermediate corpus sizes (96,144,192,240,
320,400 items), at the two multipliers that study focused on (1.2, 2.0).
Matches scratch_transition_sweep.py's exact original methodology: n_qubits=4,
single fixed corpus per size (seed=0), rng=default_rng(1), n=20/cell.
Recomputes leaky fresh too (cross-checked against the archived pickle) for a
clean apples-to-apples comparison within one run.
"""
import time
import numpy as np
import pickle
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k, cosine_similarity

CORPUS_STD = 0.04882
n_qubits = 4
edges = KnowledgeGraph.chain(n_qubits).edges()
N_QUEUE = 20
SIZES = (96, 144, 192, 240, 320, 400)
MULTS = (1.2, 2.0)

with open("scratch_transition_sweep_results.pkl", "rb") as f:
    archived = pickle.load(f)

print("=== Leak-free correction: 6 transition-study corpus sizes x mult={1.2, 2.0} ===")
results = {}
for n_items in SIZES:
    with open(f"scratch_transition_corpus_{n_items}.pkl", "rb") as f:
        d = pickle.load(f)
    corpus, keys = d["corpus"], d["keys"]
    d_dim = next(iter(corpus.values())).shape[0]
    candidate_batch = np.stack([corpus[k] for k in keys])

    for mult in MULTS:
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
        results[(n_items, mult)] = row
        arch = archived[n_items][mult]
        print(f"  n={n_items:4d} mult={mult:>4.1f}  pca_leaky={row[('pca','leaky')]:.3f} "
              f"(archived={arch['pca']:.3f})  pca_leakfree={row[('pca','leak_free')]:.3f}  "
              f"random_leaky={row[('random','leaky')]:.3f}  random_leakfree={row[('random','leak_free')]:.3f}  "
              f"classical={row['classical']:.3f}")

print("\n--- SUMMARY: leaky vs leak-free gap (classical - PCA), all 6 sizes x 2 mults ---")
print(f"{'n_items':>8}  {'mult':>5}  {'pca_leaky':>10}  {'pca_leakfree':>13}  {'pca delta':>10}  "
      f"{'gap_leaky':>10}  {'gap_leakfree':>13}  {'gap delta':>10}")
for n_items in SIZES:
    for mult in MULTS:
        row = results[(n_items, mult)]
        pca_leaky = row[("pca", "leaky")]
        pca_lf = row[("pca", "leak_free")]
        cls = row["classical"]
        gap_leaky = cls - pca_leaky
        gap_lf = cls - pca_lf
        print(f"{n_items:>8}  {mult:>5.1f}  {pca_leaky:>10.3f}  {pca_lf:>13.3f}  {pca_lf-pca_leaky:>+10.3f}  "
              f"{gap_leaky:>10.3f}  {gap_lf:>13.3f}  {gap_lf-gap_leaky:>+10.3f}")

with open("scratch_pca_leakage_transition_sizes_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nDONE")
