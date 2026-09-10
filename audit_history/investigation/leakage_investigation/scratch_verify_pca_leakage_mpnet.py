"""
scratch_verify_pca_leakage_mpnet.py — Task 2 leak-free-vs-leaky check for the
mpnet-embedded 48-item hard corpus (Section 6/embedding-model-generalization
test, embedding_model_generalization_2026-07-30.md). Same method as
scratch_verify_pca_leakage_remaining.py, adapted for this corpus's structure:
ONE fixed corpus (seed=0 draw, re-embedded with all-mpnet-base-v2, not rebuilt
per-seed), n=50 queries in a single pass (matching the report's own
"query count increased to n=50 per cell... matching the original's 5x10=50
total draws" convention), n_qubits=4 only (matches Result 1's table AND
Result 2's noise sweep, both run at n_qubits=4 in the original report).
"""
import time
import numpy as np
import pickle
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k

with open("scratch_mpnet_corpus.pkl", "rb") as f:
    d = pickle.load(f)
corpus, keys = d["corpus"], d["keys"]
CORPUS_STD = d["corpus_std"]
d_dim = d["d"]
candidate_batch = np.stack([corpus[k] for k in keys]).astype(np.float64)
N_QUEUE = 50
MULT = 1.2
noise = MULT * CORPUS_STD
n_qubits = 4
edges = KnowledgeGraph.chain(n_qubits).edges()

print(f"=== mpnet 48-item hard corpus (Sec 6): n_qubits=4, mult=1.2, n=50, corpus_std={CORPUS_STD:.5f} ===")
results = {}
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
        mean_recall = float(np.mean(recalls))
        results[(proj, mode)] = mean_recall
        print(f"  {proj:6s}  {mode:9s}  recall@1={mean_recall:.3f}  ({elapsed:.1f}s)")

pca_leaky = results[("pca", "leaky")]
pca_lf = results[("pca", "leak_free")]
rand_leaky = results[("random", "leaky")]
rand_lf = results[("random", "leak_free")]
gap_leaky = pca_leaky - rand_leaky
gap_lf = pca_lf - rand_lf

print("\n--- mpnet 48-item corpus: leaky vs leak-free ---")
print(f"{'PCA leaky':>10}  {'PCA leak-free':>14}  {'PCA delta':>10}  "
      f"{'gap leaky':>10}  {'gap leak-free':>14}  {'gap delta':>10}")
print(f"{pca_leaky:>10.3f}  {pca_lf:>14.3f}  {pca_lf-pca_leaky:>+10.3f}  "
      f"{gap_leaky:>10.3f}  {gap_lf:>14.3f}  {gap_lf-gap_leaky:>+10.3f}")

with open("scratch_pca_leakage_mpnet_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nDONE")
