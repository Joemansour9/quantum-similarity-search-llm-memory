"""
Leakage trace for the specific 18 PCA pairs composing the n=18-vs-18,
gate-matched, definitive PCA-vs-random comparison (mean gap 0.1030 vs
0.1515, t=5.372, p<0.0001, d=1.791), all n_qubits=4, all confirmed at
31 gates. Composition read directly from scratch_hw_final_gate_matched.pkl
/ scratch_hw_ngroup2_rows.pkl, not assumed.
"""
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from benchmark_realdata_embeddings_hard import CATEGORIES, _TEXTS_BY_CAT
from q_encoder import fit_pca_projection_scale, project_to_n_features

ORIGINAL_POOL_SIZE = 60
MULT = 1.2
CORPUS_STD = 0.04882
N_QUBITS = 4

print("Reconstructing historical pool=60 embedded corpus (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
all_texts, cat_ranges, idx = [], {}, 0
for cat in CATEGORIES:
    t60 = _TEXTS_BY_CAT[cat][:ORIGINAL_POOL_SIZE]
    all_texts.extend(t60)
    cat_ranges[cat] = (idx, idx + len(t60))
    idx += len(t60)
all_emb = model.encode(all_texts, batch_size=32, show_progress_bar=False)
emb_by_cat = {cat: list(all_emb[s:e]) for cat, (s, e) in cat_ranges.items()}
print("  done.")


def make_corpus(seed, n_items=48, n_clusters=8):
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


def query_vec_for(seed, q_idx, corpus, keys, d_dim):
    rng = np.random.default_rng(seed + 1)
    noise = MULT * CORPUS_STD
    for i in range(q_idx + 1):
        rk = keys[i % len(keys)]
        qv = corpus[rk] + rng.normal(scale=noise, size=d_dim)
    return qv


# 18 PCA pairs, read exactly from scratch_hw_final_gate_matched.pkl / scratch_hw_ngroup2_rows.pkl
PCA_PAIRS = [
    (0, 0, "item_0_c0", "Test3"), (0, 6, "item_6_c1", "Test7"),
    (0, 18, "item_18_c3", "Test7"), (0, 10, "item_10_c1", "Test7"),
    (1, 30, "item_30_c5", "Test8b1"), (1, 19, "item_19_c3", "Test8b1"),
    (2, 20, "item_20_c3", "Test8b2"), (2, 43, "item_43_c7", "Test8b2"),
    (2, 10, "item_10_c1", "Test8b2"), (2, 30, "item_30_c5", "Test8b2"),
    (2, 39, "item_39_c6", "Test8b2"), (2, 21, "item_21_c3", "Test9fill7"),
    (0, 37, "item_37_c6", "Test9fill7"), (2, 47, "item_47_c7", "Test9fill7"),
    (2, 16, "item_16_c2", "Test9fill7"), (0, 2, "item_2_c0", "Test9fill7"),
    (2, 4, "item_4_c0", "Test9fill7"), (1, 45, "item_45_c7", "Test9fill7"),
]
assert len(PCA_PAIRS) == 18

results = []
corpora = {}
for seed, q_idx, key, source in PCA_PAIRS:
    if seed not in corpora:
        corpora[seed] = make_corpus(seed)
    corpus, keys = corpora[seed]
    d_dim = next(iter(corpus.values())).shape[0]
    candidate_batch = np.stack([corpus[k] for k in keys])
    query_vec = query_vec_for(seed, q_idx, corpus, keys, d_dim)

    full_batch = np.vstack([query_vec[None, :], candidate_batch])
    R_leaky, lo_l, hi_l = fit_pca_projection_scale(full_batch, N_QUBITS)
    theta_leaky = project_to_n_features(query_vec, N_QUBITS, (lo_l, hi_l), R_leaky)

    R_free, lo_f, hi_f = fit_pca_projection_scale(candidate_batch, N_QUBITS)
    theta_free = project_to_n_features(query_vec, N_QUBITS, (lo_f, hi_f), R_free)

    diff = np.abs(theta_leaky - theta_free)
    cos_sim = np.abs(R_leaky @ R_free.T)
    basis_stable = bool((cos_sim.max(axis=1) > 0.95).all())

    mean_shift, max_shift = float(diff.mean()), float(diff.max())
    results.append({"source": source, "seed": seed, "q_idx": q_idx, "key": key,
                     "mean_shift_rad": mean_shift, "max_shift_rad": max_shift,
                     "basis_stable": basis_stable})
    flag = "" if basis_stable else "  [basis changed]"
    print(f"{source:12s} seed={seed} q_idx={q_idx:2d} {key:12s}  "
          f"mean|shift|={mean_shift:.4f} rad  max|shift|={max_shift:.4f} rad ({100*max_shift/np.pi:.1f}% of pi){flag}")

n_unstable = sum(1 for r in results if not r["basis_stable"])
mean_shifts = [r["mean_shift_rad"] for r in results]
max_shifts = [r["max_shift_rad"] for r in results]
print(f"\nSUMMARY: n=18 PCA pairs, all leaked (query in fit batch): 18/18")
print(f"  mean(mean_shift)={np.mean(mean_shifts):.4f} rad   mean(max_shift)={np.mean(max_shifts):.4f} rad")
print(f"  largest single shift={max(max_shifts):.4f} rad ({100*max(max_shifts)/np.pi:.1f}% of pi)")
print(f"  basis-unstable pairs: {n_unstable}/18")

with open("scratch_n18_leakage_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("DONE")
