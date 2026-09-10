"""
scratch_verify_scarcity_oos.py — out-of-sample (leak-free) rerun of the
corpus-scarcity finding's PCA-pair fractions, at both the 480-item corpus
(456/480 published) and the 48-item corpus (no PCA fraction was ever
published there -- only random's 8/48 -- and no script computing even
that 8/48 number was found anywhere in the project; it appears only as a
hardcoded reference comment in scratch_large_corpus_sweep.py). Random's
fractions are NOT recomputed here per the task (random projection isn't
affected by the query-inclusion leak), except as a cheap validation: the
48-item random fraction is recomputed fresh too, purely to check whether
this script's corpus/seed/query reconstruction reproduces the previously
reported 8/48 -- since no source script survives to confirm the
convention directly.

Methodology mirrors scratch_large_corpus_sweep.py's scarcity scan exactly
(n_qubits=4, chain topology, mult=1.2, exact statevector fidelity,
threshold=0.5, self-match pairs: query q_idx compared against candidate
q_idx, i.e. its own noisy self), applied at both corpus sizes. The
48-item corpus uses seed=0, matching "the 48-item corpus and seed used
throughout this hardware validation" per the paper's own Limitations
text.
"""
import pickle, time
import numpy as np
from sentence_transformers import SentenceTransformer
from q_encoder import fit_pca_projection_scale, fit_projection_scale, build_feature_map
from q_search import statevector_fidelity_from_circuits
from benchmark_realdata_embeddings_hard import CATEGORIES, _TEXTS_BY_CAT

# Reconstruct the historical pool=60 corpus (the pool size active when the
# 48-item hardware-validation corpus was built), NOT today's pool=65
# default -- same reconstruction already used in
# scratch_verify_hw_circuit_leakage.py and the fairer-baselines task.
ORIGINAL_POOL_SIZE = 60
print("Reconstructing historical pool=60 corpus (all-MiniLM-L6-v2)...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
_all_texts, _cat_ranges, _idx = [], {}, 0
for _cat in CATEGORIES:
    _texts60 = _TEXTS_BY_CAT[_cat][:ORIGINAL_POOL_SIZE]
    _all_texts.extend(_texts60)
    _cat_ranges[_cat] = (_idx, _idx + len(_texts60))
    _idx += len(_texts60)
_all_emb = _model.encode(_all_texts, batch_size=32, show_progress_bar=False)
_emb_by_cat = {c: list(_all_emb[s:e]) for c, (s, e) in _cat_ranges.items()}
HARD_STD = float(_all_emb.std(axis=0).mean())
print(f"  done. corpus per-dim std={HARD_STD:.5f} (expect ~0.04882)")


def make_hard_real_corpus(n_items: int, n_clusters: int, seed: int = 0):
    per_cluster = n_items // n_clusters
    rng = np.random.default_rng(seed)
    items, keys = {}, []
    for ci, cat in enumerate(CATEGORIES):
        pool = _emb_by_cat[cat]
        chosen = rng.choice(len(pool), size=per_cluster, replace=False)
        for j, idx in enumerate(chosen):
            key = f"item_{ci * per_cluster + j}_c{ci}"
            items[key] = np.asarray(pool[idx], dtype=float)
            keys.append(key)
    return items, keys

N_QUBITS = 4
EDGES = [(i, i + 1) for i in range(N_QUBITS - 1)]
MULT = 1.2
THRESHOLD = 0.5

PUBLISHED_480 = {"pca": (456, 480), "random": (184, 480)}
PUBLISHED_48 = {"random": (8, 48)}  # no PCA fraction was ever published for 48 items


def scarcity_scan(corpus, keys, corpus_std, oos: bool, compute_random: bool = False):
    d_dim = next(iter(corpus.values())).shape[0]
    candidate_batch = np.stack([corpus[k] for k in keys])
    noise = MULT * corpus_std
    rng = np.random.default_rng(1)
    pca_fids, rand_fids = [], []
    for q_idx in range(len(keys)):
        relevant_key = keys[q_idx % len(keys)]
        query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
        candidate_vec = candidate_batch[q_idx % len(keys)]

        if oos:
            fit_batch = candidate_batch  # query excluded
        else:
            fit_batch = np.vstack([query_vec[None, :], candidate_batch])

        R, lo, hi = fit_pca_projection_scale(fit_batch, N_QUBITS)
        qc = build_feature_map(query_vec, N_QUBITS, (lo, hi), EDGES, R, entangler="rzz")
        fid, _ = statevector_fidelity_from_circuits(qc, candidate_vec, N_QUBITS, (lo, hi), EDGES, R, entangler="rzz")
        pca_fids.append(fid)

        if compute_random:
            Rr, lor, hir = fit_projection_scale(fit_batch, N_QUBITS, seed=q_idx)
            qcr = build_feature_map(query_vec, N_QUBITS, (lor, hir), EDGES, Rr, entangler="rzz")
            fidr, _ = statevector_fidelity_from_circuits(qcr, candidate_vec, N_QUBITS, (lor, hir), EDGES, Rr, entangler="rzz")
            rand_fids.append(fidr)

    pca_fids = np.array(pca_fids)
    pca_above = int((pca_fids >= THRESHOLD).sum())
    result = {"pca_above": pca_above, "pca_total": len(keys), "pca_fids": pca_fids}
    if compute_random:
        rand_fids = np.array(rand_fids)
        rand_above = int((rand_fids >= THRESHOLD).sum())
        result["random_above"] = rand_above
        result["random_total"] = len(keys)
        result["rand_fids"] = rand_fids
    return result


# ============================================================
# 480-item corpus: out-of-sample PCA fraction
# ============================================================
print("=== 480-item corpus: out-of-sample PCA-pair scarcity scan ===")
with open("scratch_large_corpus_480.pkl", "rb") as f:
    d480 = pickle.load(f)
corpus480, keys480 = d480["corpus"], d480["keys"]
CORPUS_STD_480 = 0.04882
t0 = time.perf_counter()
res_480 = scarcity_scan(corpus480, keys480, CORPUS_STD_480, oos=True, compute_random=False)
print(f"  PCA (out-of-sample): {res_480['pca_above']}/{res_480['pca_total']} "
      f"({100*res_480['pca_above']/res_480['pca_total']:.1f}%) clear sim_fid>=0.5  "
      f"({time.perf_counter()-t0:.0f}s)")
print(f"  Published (leaky): {PUBLISHED_480['pca'][0]}/{PUBLISHED_480['pca'][1]} "
      f"({100*PUBLISHED_480['pca'][0]/PUBLISHED_480['pca'][1]:.1f}%)")

# ============================================================
# 48-item corpus, seed=0: out-of-sample PCA fraction + fresh random cross-check
# ============================================================
print("\n=== 48-item corpus (seed=0): out-of-sample PCA-pair scarcity scan + random cross-check ===")
corpus48, keys48 = make_hard_real_corpus(48, 8, seed=0)
t0 = time.perf_counter()
res_48 = scarcity_scan(corpus48, keys48, HARD_STD, oos=True, compute_random=True)
print(f"  PCA (out-of-sample): {res_48['pca_above']}/{res_48['pca_total']} "
      f"({100*res_48['pca_above']/res_48['pca_total']:.1f}%) clear sim_fid>=0.5  "
      f"({time.perf_counter()-t0:.0f}s)")
print(f"  Random (fresh, leaky==out-of-sample since random doesn't leak): "
      f"{res_48['random_above']}/{res_48['random_total']} "
      f"({100*res_48['random_above']/res_48['random_total']:.1f}%)")
print(f"  Published random reference: {PUBLISHED_48['random'][0]}/{PUBLISHED_48['random'][1]} "
      f"({100*PUBLISHED_48['random'][0]/PUBLISHED_48['random'][1]:.1f}%)")
match_48_random = (res_48['random_above'], res_48['random_total']) == PUBLISHED_48['random']
print(f"  Reconstruction matches published 8/48 random reference: {match_48_random}")
if not match_48_random:
    print("  NOTE: no source script for the original 8/48 number was found anywhere in the "
          "project (only a hardcoded reference comment) -- this mismatch cannot be resolved "
          "against an original script, only flagged.")

out = {
    "description": (
        "Out-of-sample (leak-free) rerun of the corpus-scarcity finding's "
        "PCA-pair fractions at 480 and 48 items. Random-projection fractions "
        "were not required to be rechecked (unaffected by the leak), except "
        "a fresh 48-item random computation included here solely to test "
        "whether this script's corpus/seed/query reconstruction reproduces "
        "the previously published 8/48 reference -- no script computing "
        "that original number survives in the project to verify against "
        "directly."
    ),
    "threshold": THRESHOLD,
    "n_qubits": N_QUBITS,
    "mult": MULT,
    "corpus_480": {
        "pca_above": res_480["pca_above"], "pca_total": res_480["pca_total"],
        "pca_fids": res_480["pca_fids"],
    },
    "corpus_48": {
        "pca_above": res_48["pca_above"], "pca_total": res_48["pca_total"],
        "pca_fids": res_48["pca_fids"],
        "random_above": res_48["random_above"], "random_total": res_48["random_total"],
        "rand_fids": res_48["rand_fids"],
        "matches_published_random_8_of_48": match_48_random,
    },
    "published_reference": {"480_pca": PUBLISHED_480["pca"], "480_random": PUBLISHED_480["random"], "48_random": PUBLISHED_48["random"]},
}

with open("scratch_scarcity_oos_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE -- saved scratch_scarcity_oos_results.pkl")
