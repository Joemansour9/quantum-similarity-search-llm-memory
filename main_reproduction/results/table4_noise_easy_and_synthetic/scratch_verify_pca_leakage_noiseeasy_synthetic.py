"""
scratch_verify_pca_leakage_noiseeasy_synthetic.py — fresh leak-free
(out-of-sample) rerun of the query-noise sweep behind Table noise-easy
(synthetic + real-easy corpora), which has never had a leak-free version
computed anywhere in this project (confirmed: no script referencing
either corpus's noise sweep with a query-excluded fit exists on disk).

Matches each corpus's ORIGINAL leaky methodology exactly (same corpus
construction, same n_seeds/n_queries, same rng convention, same noise
values), replacing only the PCA/random fitting step with an out-of-sample
fit (candidate_batch only, query excluded) -- the same substitution used
throughout this project's other leak-free reruns.

Synthetic: mirrors benchmark_task_difficulty.py's run_query_noise_sweep
(n_items=24, d=32, n_clusters=8, cluster_spread=2.0, n_qubits=8, noise as
an ABSOLUTE scale, n_seeds=5 x n_queries=10 = 50/cell).

Real-easy: mirrors benchmark_query_noise_realdata.py's
run_query_noise_sweep_real (n_items=24, n_clusters=8, n_qubits=8, noise
as a MULTIPLIER on this corpus's own CORPUS_PER_DIM_STD, same n_seeds x
n_queries).

Noise values restricted to the four the paper's Table noise-easy actually
reports (0.2, 1.2, 3.0, 5.0), a subset of each original script's six.
Classical (raw-embedding cosine) recall is also recomputed fresh here for
completeness -- it does not depend on PCA fitting at all, so it should
(and does, per the printed comparison) match the already-published
Table noise-easy classical column exactly.
"""
import time
import numpy as np
import pickle

from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k, cosine_similarity

N_QUBITS = 8
N_SEEDS = 5
N_QUERIES = 10
NOISE_VALUES = (0.2, 1.2, 3.0, 5.0)  # matches Table noise-easy exactly
edges = KnowledgeGraph.chain(N_QUBITS).edges()

# Published Table noise-easy classical values, for the sanity cross-check
PUBLISHED_CLASSICAL = {
    "synthetic": {0.2: 1.000, 1.2: 1.000, 3.0: 0.840, 5.0: 0.460},
    "real_easy": {0.2: 1.000, 1.2: 1.000, 3.0: 1.000, 5.0: 1.000},
}


def run_sweep(corpus_fn, noise_is_multiplier, corpus_std, label):
    print(f"\n=== {label} ===")
    results = {}
    for noise_val in NOISE_VALUES:
        noise = noise_val * corpus_std if noise_is_multiplier else noise_val
        row = {}
        for proj in ("pca", "random"):
            t0 = time.perf_counter()
            recalls = []
            for seed in range(N_SEEDS):
                corpus, keys = corpus_fn(seed)
                d_dim = next(iter(corpus.values())).shape[0]
                candidate_batch = np.stack([corpus[k] for k in keys]).astype(np.float64)
                rng = np.random.default_rng(seed + 1)
                for q_idx in range(N_QUERIES):
                    relevant_key = keys[q_idx % len(keys)]
                    relevant_idx = keys.index(relevant_key)
                    query_vec = corpus[relevant_key].astype(np.float64) + rng.normal(scale=noise, size=d_dim)

                    # OUT-OF-SAMPLE: fit on candidate_batch only, query excluded
                    if proj == "pca":
                        R, lo, hi = fit_pca_projection_scale(candidate_batch, N_QUBITS)
                    else:
                        R, lo, hi = fit_projection_scale(candidate_batch, N_QUBITS, seed=q_idx)
                    scale = (lo, hi)

                    qc_query = build_feature_map(query_vec, N_QUBITS, scale, edges, R, entangler="rzz")
                    fids = np.zeros(len(keys))
                    for i in range(len(keys)):
                        fid, _ = statevector_fidelity_from_circuits(
                            qc_query, candidate_batch[i], N_QUBITS, scale, edges, R, entangler="rzz"
                        )
                        fids[i] = fid
                    rank = np.argsort(-fids)
                    recalls.append(recall_at_k(rank, relevant_idx, 1))
            elapsed = time.perf_counter() - t0
            row[proj] = float(np.mean(recalls))

        # classical: raw-embedding cosine, same seeds/queries, no PCA fit involved
        classical_recalls = []
        for seed in range(N_SEEDS):
            corpus, keys = corpus_fn(seed)
            d_dim = next(iter(corpus.values())).shape[0]
            candidate_batch = np.stack([corpus[k] for k in keys]).astype(np.float64)
            rng = np.random.default_rng(seed + 1)
            for q_idx in range(N_QUERIES):
                relevant_key = keys[q_idx % len(keys)]
                relevant_idx = keys.index(relevant_key)
                query_vec = corpus[relevant_key].astype(np.float64) + rng.normal(scale=noise, size=d_dim)
                cos = np.array([cosine_similarity(query_vec, candidate_batch[i]) for i in range(len(keys))])
                rank = np.argsort(-cos)
                classical_recalls.append(recall_at_k(rank, relevant_idx, 1))
        row["classical"] = float(np.mean(classical_recalls))

        results[noise_val] = row
        print(f"  noise={noise_val:>4.1f}  random={row['random']:.3f}  pca={row['pca']:.3f}  classical={row['classical']:.3f}")
    return results


# --- Synthetic corpus ---
from benchmark_projection import make_synthetic_corpus
synthetic_results = run_sweep(
    corpus_fn=lambda seed: make_synthetic_corpus(24, 32, 8, seed=seed, cluster_spread=2.0),
    noise_is_multiplier=False, corpus_std=None,
    label="Synthetic corpus: 24 items, d=32, 8 clusters, cluster_spread=2.0, n_qubits=8, noise=absolute",
)

# --- Real-easy corpus ---
from benchmark_realdata_embeddings import make_real_corpus, CORPUS_PER_DIM_STD as EASY_STD
real_easy_results = run_sweep(
    corpus_fn=lambda seed: make_real_corpus(24, 8, seed=seed),
    noise_is_multiplier=True, corpus_std=EASY_STD,
    label=f"Real-easy corpus: 24 items, 8 topics, MiniLM, n_qubits=8, mult x std={EASY_STD:.5f}",
)

print("\n=== Cross-check: fresh classical recall vs. published Table noise-easy classical column ===")
all_match = True
for label, results in (("synthetic", synthetic_results), ("real_easy", real_easy_results)):
    for noise_val in NOISE_VALUES:
        fresh = results[noise_val]["classical"]
        published = PUBLISHED_CLASSICAL[label][noise_val]
        ok = abs(fresh - published) < 1e-9
        all_match = all_match and ok
        print(f"  {label:10s} noise={noise_val:>4.1f}  fresh={fresh:.3f}  published={published:.3f}  match={ok}")
print(f"\nClassical recall matches published table exactly (as expected, no PCA fit involved): {all_match}")

out = {
    "description": (
        "Fresh out-of-sample (leak-free) query-noise sweep for Table noise-easy "
        "(synthetic + real-easy corpora), computed because no leak-free version "
        "of this table existed anywhere in the project before this run. "
        "Random/PCA use out-of-sample fitting (query excluded from the batch "
        "used to fit the projection); classical is raw-embedding cosine "
        "similarity, unaffected by the leakage issue, recomputed here fresh "
        "and cross-checked against the published table's classical column."
    ),
    "n_qubits": N_QUBITS,
    "n_seeds": N_SEEDS,
    "n_queries": N_QUERIES,
    "noise_values": NOISE_VALUES,
    "synthetic": synthetic_results,
    "real_easy": real_easy_results,
    "classical_cross_check_all_match": all_match,
}

with open("scratch_pca_leakage_noiseeasy_synthetic_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE -- saved scratch_pca_leakage_noiseeasy_synthetic_results.pkl")
