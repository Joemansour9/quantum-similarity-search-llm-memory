"""
scratch_verify_hw_circuit_leakage.py — trace PCA-fit query leakage in the
ACTUAL circuit-building step for the five named 2026-07-30 hardware jobs
(not the offline qubit-sweep recall statistics that scratch_verify_pca_leakage*.py
already checked).

Reconstructs the historically-accurate 60/category embedded pool (the
value active when these jobs were built, before _POOL_PER_CATEGORY was
raised to 65 later in the project for the large-corpus test) from
_TEXTS_BY_CAT, which is NOT truncated at load time -- only the embedding
step slices it -- so re-slicing to [:60] here reproduces the original
pool exactly, using the same model (all-MiniLM-L6-v2).

For each PCA pair actually submitted to hardware, computes:
  - LEAKY R/scale (as actually used): fit on [query] + candidate_batch
  - LEAK-FREE R/scale: fit on candidate_batch alone, query excluded
  - the resulting query rotation angles (theta) under each condition
  - the per-qubit angle shift in radians and as a fraction of the [0,pi] range
"""
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from benchmark_realdata_embeddings_hard import CATEGORIES, _TEXTS_BY_CAT
from q_encoder import fit_pca_projection_scale, project_to_n_features

ORIGINAL_POOL_SIZE = 60
MULT = 1.2
CORPUS_STD = 0.04882  # matches the value used throughout the project at pool=60

print("Reconstructing the historical pool=60 embedded corpus (all-MiniLM-L6-v2)...")
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
print(f"  done. {len(all_texts)} docs embedded, {len(CATEGORIES)} categories x {ORIGINAL_POOL_SIZE}")


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


def query_vec_for(seed: int, q_idx: int, corpus, keys, d_dim):
    # matches the project's established sequential-rng convention:
    # one rng seeded (seed+1), drawn sequentially for q_idx=0..N
    rng = np.random.default_rng(seed + 1)
    noise = MULT * CORPUS_STD
    for i in range(q_idx + 1):
        rk = keys[i % len(keys)]
        qv = corpus[rk] + rng.normal(scale=noise, size=d_dim)
    return qv


# --- The five named jobs, exactly as built ---
JOBS = {
    "Test1 (d9l0mrqbr2fc73e812hg / d9l17rbjf64c739isibg)": {
        "n_qubits": 4,
        "pairs": [(0, 0, "item_0_c0")],
    },
    "Test10 (d9lac03jf64c739j8i50, n_qubits=8)": {
        "n_qubits": 8,
        "pairs": [
            (0, 0, "item_0_c0"), (7, 10, "item_10_c1"), (4, 7, "item_7_c1"),
            (7, 20, "item_20_c3"), (6, 28, "item_28_c4"), (4, 20, "item_20_c3"),
            (1, 19, "item_19_c3"), (7, 26, "item_26_c4"),
        ],
    },
    "Test11 (d9lal32br2fc73e8e0sg, n_qubits=6)": {
        "n_qubits": 6,
        "pairs": [
            (3, 39, "item_39_c6"), (0, 0, "item_0_c0"), (4, 7, "item_7_c1"),
            (2, 20, "item_20_c3"), (3, 10, "item_10_c1"), (1, 19, "item_19_c3"),
            (4, 20, "item_20_c3"), (0, 6, "item_6_c1"),
        ],
    },
    "Priority1/n5 (d9lblv8ii2cc73eh9a40, n_qubits=5)": {
        "n_qubits": 5,
        "pairs": [
            (0, 0, "item_0_c0"), (1, 19, "item_19_c3"), (0, 6, "item_6_c1"),
            (1, 30, "item_30_c5"), (0, 18, "item_18_c3"), (0, 37, "item_37_c6"),
            (0, 2, "item_2_c0"), (0, 10, "item_10_c1"),
        ],
    },
}

all_shifts = []
print("\n" + "=" * 100)
for job_name, spec in JOBS.items():
    n_qubits = spec["n_qubits"]
    print(f"\n### {job_name}  (n_qubits={n_qubits}) ###")
    corpora = {}
    for seed, q_idx, key in spec["pairs"]:
        if seed not in corpora:
            corpora[seed] = make_corpus(seed)
        corpus, keys = corpora[seed]
        d_dim = next(iter(corpus.values())).shape[0]
        candidate_batch = np.stack([corpus[k] for k in keys])
        query_vec = query_vec_for(seed, q_idx, corpus, keys, d_dim)

        # LEAKY (as actually submitted to hardware)
        full_batch = np.vstack([query_vec[None, :], candidate_batch])
        R_leaky, lo_l, hi_l = fit_pca_projection_scale(full_batch, n_qubits)
        theta_leaky = project_to_n_features(query_vec, n_qubits, (lo_l, hi_l), R_leaky)

        # LEAK-FREE (candidate_batch only, query excluded from fit)
        R_free, lo_f, hi_f = fit_pca_projection_scale(candidate_batch, n_qubits)
        theta_free = project_to_n_features(query_vec, n_qubits, (lo_f, hi_f), R_free)

        diff = np.abs(theta_leaky - theta_free)
        # angles from different R bases aren't necessarily comparable component-by-component
        # if PCA picked different top-k directions leaky vs leak-free -- check that first
        cos_sim = np.abs(R_leaky @ R_free.T)  # (n_qubits, n_qubits) cross terms
        best_match = cos_sim.max(axis=1)  # how well each leaky component is matched by some leak-free component
        basis_stable = bool((best_match > 0.95).all())

        mean_shift = float(diff.mean())
        max_shift = float(diff.max())
        frac_pi = max_shift / np.pi
        all_shifts.append({"job": job_name, "seed": seed, "q_idx": q_idx, "key": key,
                            "mean_shift_rad": mean_shift, "max_shift_rad": max_shift,
                            "frac_of_pi": frac_pi, "basis_stable": basis_stable})
        stability_note = "" if basis_stable else "  [WARNING: PCA basis itself changed, not just angles]"
        print(f"  seed={seed} q_idx={q_idx:2d} {key:12s}  "
              f"mean|shift|={mean_shift:.4f} rad  max|shift|={max_shift:.4f} rad ({100*frac_pi:.1f}% of pi){stability_note}")

with open("scratch_hw_circuit_leakage_results.pkl", "wb") as f:
    pickle.dump(all_shifts, f)

print("\n" + "=" * 100)
print("SUMMARY")
by_job = {}
for r in all_shifts:
    by_job.setdefault(r["job"], []).append(r)
for job, rows in by_job.items():
    mean_shifts = [r["mean_shift_rad"] for r in rows]
    max_shifts = [r["max_shift_rad"] for r in rows]
    n_unstable = sum(1 for r in rows if not r["basis_stable"])
    print(f"{job}")
    print(f"  n pairs={len(rows)}  mean(mean_shift)={np.mean(mean_shifts):.4f} rad  "
          f"mean(max_shift)={np.mean(max_shifts):.4f} rad  "
          f"largest single shift={max(max_shifts):.4f} rad ({100*max(max_shifts)/np.pi:.1f}% of pi)  "
          f"basis-unstable pairs={n_unstable}/{len(rows)}")
print("\nDONE")
