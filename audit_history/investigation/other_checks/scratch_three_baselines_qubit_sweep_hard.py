"""
scratch_three_baselines_qubit_sweep_hard.py

Three additional classical projection baselines, requested against the
existing PCA/random qubit-count sweep on the 48-item hard corpus
(Table qubit-hard in the paper / benchmark_qubit_sweep_realdata_hard.py):

  - svd       : truncated SVD, no centering (plain SVD keeps the origin
                fixed; PCA centers on the batch mean first)
  - whiten    : the same PCA projection, each of the n_qubits output
                dimensions rescaled to unit variance before angle encoding
  - firstn    : the first n_qubits raw embedding dimensions directly, no
                learned projection at all

All three, plus a same-run PCA/random cross-check, are fit OUT-OF-SAMPLE:
the query vector is excluded from whatever batch statistic the method
needs (matches this project's established leak-free convention, e.g.
scratch_verify_hw_circuit_leakage.py / scratch_verify_pca_leakage*.py).

Corpus reconstruction: benchmark_qubit_sweep_realdata_hard.py's PUBLISHED
Table qubit-hard numbers were produced when _POOL_PER_CATEGORY was 60, not
today's 65 (documented in repo/main_reproduction/README.md's caveat and
already handled once before in scratch_verify_hw_circuit_leakage.py). This
script reconstructs that same pool=60 corpus from _TEXTS_BY_CAT (not
truncated at load time) so the corpus draws are identical to what actually
produced the published numbers, not merely "the same code run today".

Same seeds/noise/query convention as benchmark_qubit_sweep_realdata_hard.py:
  - 5 seeds (0..4), each seed redraws its own 48-item corpus via
    make_corpus(seed) (pool=60 version)
  - 10 queries per seed, noise rng = np.random.default_rng(seed + 1),
    drawn sequentially, multiplier 1.2 x CORPUS_PER_DIM_STD (matches the
    default query_noise_multiplier in run_qubit_sweep_hard)
  - n=50 query-runs per cell, Recall@1, statevector fidelity (exact,
    noise-free reference -- same as the published table)
"""
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from benchmark_realdata_embeddings_hard import CATEGORIES, _TEXTS_BY_CAT
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_graph import KnowledgeGraph
from q_search import statevector_fidelity_from_circuits, recall_at_k

ORIGINAL_POOL_SIZE = 60
MULT = 1.2
N_SEEDS = 5
N_QUERIES = 10
QUBIT_COUNTS = [4, 6, 8, 10, 12, 14]

# ---------------------------------------------------------------------
# Reconstruct the historical pool=60 embedded corpus (all-MiniLM-L6-v2)
# ---------------------------------------------------------------------
print("Reconstructing the historical pool=60 embedded corpus (all-MiniLM-L6-v2)...")
t0 = time.perf_counter()
model = SentenceTransformer("all-MiniLM-L6-v2")
all_texts = []
cat_ranges = {}
idx = 0
for cat in CATEGORIES:
    texts60 = _TEXTS_BY_CAT[cat][:ORIGINAL_POOL_SIZE]
    all_texts.extend(texts60)
    cat_ranges[cat] = (idx, idx + len(texts60))
    idx += len(texts60)
all_emb = model.encode(all_texts, batch_size=32, show_progress_bar=False)
emb_by_cat = {cat: list(all_emb[s:e]) for cat, (s, e) in cat_ranges.items()}
CORPUS_PER_DIM_STD = float(all_emb.std(axis=0).mean())
print(f"  done in {time.perf_counter()-t0:.1f}s. {len(all_texts)} docs, "
      f"corpus per-dim std={CORPUS_PER_DIM_STD:.5f}")


def make_corpus(seed: int, n_items: int = 48, n_clusters: int = 8):
    per_cluster = n_items // n_clusters
    rng = np.random.default_rng(seed)
    items, keys = {}, []
    for ci, cat in enumerate(CATEGORIES):
        pool = emb_by_cat[cat]
        chosen = rng.choice(len(pool), size=per_cluster, replace=False)
        for j, i in enumerate(chosen):
            key = f"item_{ci * per_cluster + j}_c{ci}"
            items[key] = np.asarray(pool[i], dtype=float)
            keys.append(key)
    return items, keys


# ---------------------------------------------------------------------
# New projection-fitting functions -- same (batch, n_qubits) -> (R, lo, hi)
# contract as fit_pca_projection_scale / fit_projection_scale, so they
# drop straight into build_feature_map / statevector_fidelity_from_circuits
# unchanged.
# ---------------------------------------------------------------------

def fit_svd_projection_scale(batch: np.ndarray, n_qubits: int):
    """Truncated SVD, NO centering -- origin fixed, unlike PCA."""
    d = batch.shape[1]
    if batch.shape[0] < 2:
        raise ValueError("Need at least 2 batch vectors.")
    if n_qubits > min(batch.shape[0], d):
        raise ValueError(f"n_qubits={n_qubits} exceeds min(batch_size, d)")
    _, _, Vt = np.linalg.svd(batch, full_matrices=False)  # no mean subtraction
    R = Vt[:n_qubits]
    projected = batch @ R.T
    return R, projected.min(), projected.max()


def fit_whitened_pca_projection_scale(batch: np.ndarray, n_qubits: int):
    """Same PCA directions (mean-centered SVD), each dimension rescaled
    to unit variance (measured on the same uncentered projection
    fit_pca_projection_scale itself uses) before the shared min/max fit."""
    R_pca, _, _ = fit_pca_projection_scale(batch, n_qubits)
    proj = batch @ R_pca.T  # (batch_size, n_qubits), uncentered -- matches
                            # fit_pca_projection_scale's own convention
    stds = proj.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    R = R_pca / stds[:, None]
    projected = batch @ R.T
    return R, projected.min(), projected.max()


def fit_firstn_projection_scale(batch: np.ndarray, n_qubits: int):
    """No learned projection: R selects the first n_qubits raw dimensions.
    The (lo, hi) scale still has to come from *somewhere* to turn raw
    values into [0, pi] angles under this pipeline's encoding convention
    -- fit from the out-of-sample candidate batch, same as every other
    method here, rather than an arbitrary fixed constant (flagged in the
    accompanying report)."""
    d = batch.shape[1]
    R = np.eye(n_qubits, d)
    projected = batch @ R.T  # == batch[:, :n_qubits]
    return R, projected.min(), projected.max()


METHODS = {
    "random": lambda batch, nq, seed: fit_projection_scale(batch, nq, seed=seed),
    "pca": lambda batch, nq, seed: fit_pca_projection_scale(batch, nq),
    "svd": lambda batch, nq, seed: fit_svd_projection_scale(batch, nq),
    "whiten": lambda batch, nq, seed: fit_whitened_pca_projection_scale(batch, nq),
    "firstn": lambda batch, nq, seed: fit_firstn_projection_scale(batch, nq),
}

print(f"\n=== Out-of-sample qubit-count sweep, 48-item hard corpus (pool=60 reconstruction) ===")
print(f"(mult={MULT}, {N_SEEDS} seeds x {N_QUERIES} queries = {N_SEEDS*N_QUERIES} query-runs/cell, statevector fidelity)\n")

results = {}  # (n_qubits, method) -> mean recall@1
noise = MULT * CORPUS_PER_DIM_STD

for n_qubits in QUBIT_COUNTS:
    edges = KnowledgeGraph.chain(n_qubits).edges()
    for method_name, fit_fn in METHODS.items():
        t0 = time.perf_counter()
        recalls = []
        for seed in range(N_SEEDS):
            corpus, keys = make_corpus(seed)
            d_dim = next(iter(corpus.values())).shape[0]
            candidate_batch = np.stack([corpus[k] for k in keys]).astype(np.float64)
            rng = np.random.default_rng(seed + 1)
            for q_idx in range(N_QUERIES):
                relevant_key = keys[q_idx % len(keys)]
                relevant_idx = keys.index(relevant_key)
                query_vec = corpus[relevant_key].astype(np.float64) + rng.normal(scale=noise, size=d_dim)

                # OUT-OF-SAMPLE fit: candidate_batch only, query excluded
                R, lo, hi = fit_fn(candidate_batch, n_qubits, q_idx)
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
        results[(n_qubits, method_name)] = mean_recall
        print(f"  n_qubits={n_qubits:2d}  {method_name:7s}  Recall@1={mean_recall:.3f}  "
              f"({elapsed:.1f}s for {len(recalls)} query-runs)")

with open("scratch_three_baselines_qubit_sweep_hard_results.pkl", "wb") as f:
    pickle.dump({
        "results": results,
        "corpus_per_dim_std": CORPUS_PER_DIM_STD,
        "mult": MULT,
        "n_seeds": N_SEEDS,
        "n_queries": N_QUERIES,
        "qubit_counts": QUBIT_COUNTS,
        "pool_size": ORIGINAL_POOL_SIZE,
    }, f)

print("\nDONE -- saved scratch_three_baselines_qubit_sweep_hard_results.pkl")
