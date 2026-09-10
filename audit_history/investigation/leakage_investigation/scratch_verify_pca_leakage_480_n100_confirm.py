"""
scratch_verify_pca_leakage_480_n100_confirm.py — publication-grade leak-free
rerun at n=100/cell for the two most consequential points flagged by the
n=20 pass: 480-item corpus, n_qubits=4, mult=4.0 and mult=6.0. Matches the
original n=100 confirmation methodology (large_corpus_generalization_2026-07-30.md's
Result 2: single fixed corpus seed=0, rng=default_rng(1), Wald 95% CI =
1.96*sqrt(p(1-p)/n)) that produced the currently-published PCA=0.420+/-0.097
(mult=4.0) and PCA=0.560+/-0.097 (mult=6.0) figures.
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
N_QUEUE = 100
MULTS = (4.0, 6.0)

def wald_ci(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)

print(f"=== 480-item corpus: n=100/cell confirmation, n_qubits=4, mult=4.0 & 6.0 ===")
results = {}
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
            p = float(np.mean(recalls))
            ci = wald_ci(p, N_QUEUE)
            row[(proj, mode)] = (p, ci)
            print(f"  mult={mult:>4.1f}  {proj:6s}  {mode:9s}  recall@1={p:.3f} +/- {ci:.3f}  ({elapsed:.1f}s)")

    rng = np.random.default_rng(1)
    classical_recalls = []
    for q_idx in range(N_QUEUE):
        relevant_key = keys[q_idx % len(keys)]
        relevant_idx = keys.index(relevant_key)
        query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
        cos = np.array([cosine_similarity(query_vec, candidate_batch[i]) for i in range(len(keys))])
        rank = np.argsort(-cos)
        classical_recalls.append(recall_at_k(rank, relevant_idx, 1))
    p_cls = float(np.mean(classical_recalls))
    ci_cls = wald_ci(p_cls, N_QUEUE)
    row["classical"] = (p_cls, ci_cls)
    print(f"  mult={mult:>4.1f}  classical           recall@1={p_cls:.3f} +/- {ci_cls:.3f}")
    results[mult] = row

print("\n=== SUMMARY: n=100/cell, leak-free vs leaky vs published ===")
published = {4.0: (0.420, 0.097, 0.980, 0.027), 6.0: (0.560, 0.097, 0.650, 0.093)}
for mult in MULTS:
    row = results[mult]
    pca_leaky, pca_leaky_ci = row[("pca", "leaky")]
    pca_lf, pca_lf_ci = row[("pca", "leak_free")]
    rand_leaky, rand_leaky_ci = row[("random", "leaky")]
    rand_lf, rand_lf_ci = row[("random", "leak_free")]
    cls, cls_ci = row["classical"]
    pub_pca, pub_pca_ci, pub_cls, pub_cls_ci = published[mult]

    print(f"\n--- mult={mult} ---")
    print(f"  Published (leaky):        PCA={pub_pca:.3f}+/-{pub_pca_ci:.3f}  Classical={pub_cls:.3f}+/-{pub_cls_ci:.3f}")
    print(f"  This rerun, LEAKY:         PCA={pca_leaky:.3f}+/-{pca_leaky_ci:.3f}  Classical={cls:.3f}+/-{cls_ci:.3f}")
    print(f"  This rerun, LEAK-FREE:     PCA={pca_lf:.3f}+/-{pca_lf_ci:.3f}  Classical={cls:.3f}+/-{cls_ci:.3f}")
    print(f"  Random (leaky/leak-free):  {rand_leaky:.3f}+/-{rand_leaky_ci:.3f} / {rand_lf:.3f}+/-{rand_lf_ci:.3f}")

    lo_p, hi_p = pca_lf - pca_lf_ci, pca_lf + pca_lf_ci
    lo_c, hi_c = cls - cls_ci, cls + cls_ci
    overlap = not (hi_p < lo_c or hi_c < lo_p)
    gap = cls - pca_lf
    print(f"  LEAK-FREE gap (Classical-PCA) = {gap:.3f}   95% CIs: {'OVERLAPPING' if overlap else 'NON-OVERLAPPING'}")

with open("scratch_pca_leakage_480_n100_confirm_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nDONE")
