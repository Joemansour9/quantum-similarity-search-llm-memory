import pickle
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD

SIZES = (96, 144, 192, 240, 320, 400)

print(f"CORPUS_PER_DIM_STD = {CORPUS_PER_DIM_STD:.5f}")

for n in SIZES:
    corpus, keys = make_hard_real_corpus(n_items=n, n_clusters=8, seed=0)
    assert len(keys) == n, (n, len(keys))
    per_cat = n // 8
    cat_counts = {}
    for k in keys:
        c = k.rsplit("_c", 1)[1]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    assert all(v == per_cat for v in cat_counts.values()), cat_counts
    fname = f"scratch_transition_corpus_{n}.pkl"
    with open(fname, "wb") as f:
        pickle.dump({"corpus": corpus, "keys": keys, "n_items": n}, f)
    print(f"  n={n:4d}  per_cat={per_cat:3d}  -> {fname}  (verified {per_cat}/category x 8)")

print("DONE building corpora")
