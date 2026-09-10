"""
scratch_verify_variance_enrichment.py -- independent re-derivation of Background
Section 2.3's variance/enrichment percentages (21.7%/58.8% MiniLM, 21.0%/57.5% mpnet)
against the actual saved/regenerated 48-item hard-corpus embeddings, per
main_reproduction/README.md's note that these numbers are "computed directly from
data already present in this folder / pipeline/ ... rather than from a separate
saved result file."

MiniLM: regenerated via make_hard_real_corpus(48, 8, seed=0) from
benchmark_realdata_embeddings_hard.py -- this is the seed=0 corpus, the first of the
N_SEEDS corpora used in table3_realdata_hard_and_table13_baselines's qubit sweep.
20 Newsgroups fetch + all-MiniLM-L6-v2 encoding are both deterministic given the same
inputs, so this should reproduce the exact corpus.

mpnet: loaded directly from the pre-existing saved fixture scratch_mpnet_corpus.pkl
(48 items, d=768, corpus_std=0.03344293311238289), the same corpus
table6_mpnet_generalisation's scripts load.
"""
import pickle
import numpy as np
from sklearn.decomposition import PCA

print("=== MiniLM: regenerating 48-item hard corpus (seed=0) ===")
from benchmark_realdata_embeddings_hard import make_hard_real_corpus
corpus_m, keys_m = make_hard_real_corpus(48, 8, seed=0)
X_minilm = np.stack([corpus_m[k] for k in keys_m]).astype(np.float64)
print("MiniLM embedding matrix shape:", X_minilm.shape)

print("\n=== mpnet: loading saved scratch_mpnet_corpus.pkl ===")
with open("scratch_mpnet_corpus.pkl", "rb") as f:
    d = pickle.load(f)
corpus_p, keys_p = d["corpus"], d["keys"]
X_mpnet = np.stack([corpus_p[k] for k in keys_p]).astype(np.float64)
print("mpnet embedding matrix shape:", X_mpnet.shape)

def variance_report(X, name, dims):
    pca = PCA(n_components=min(X.shape))
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    print(f"\n--- {name} (d={X.shape[1]}) ---")
    for k in dims:
        pct = float(evr[:k].sum() * 100)
        null = 100.0 * k / X.shape[1]
        print(f"  top {k:2d} PCs: {pct:.1f}% of variance  (isotropic null={null:.2f}%, enrichment={pct/null:.1f}x)")
    return evr

evr_m = variance_report(X_minilm, "MiniLM", [4, 16])
evr_p = variance_report(X_mpnet, "mpnet", [4, 16])

out = {
    "description": (
        "Independent re-derivation of Background Sec 2.3's variance/enrichment "
        "percentages against the actual 48-item hard-corpus embeddings. MiniLM "
        "corpus regenerated via make_hard_real_corpus(48,8,seed=0) (deterministic "
        "20 Newsgroups fetch + all-MiniLM-L6-v2 encoding); mpnet corpus loaded "
        "from the pre-existing scratch_mpnet_corpus.pkl fixture."
    ),
    "minilm_evr_top4_pct": float(evr_m[:4].sum() * 100),
    "minilm_evr_top16_pct": float(evr_m[:16].sum() * 100),
    "mpnet_evr_top4_pct": float(evr_p[:4].sum() * 100),
    "mpnet_evr_top16_pct": float(evr_p[:16].sum() * 100),
    "minilm_dim": X_minilm.shape[1],
    "mpnet_dim": X_mpnet.shape[1],
}
with open("scratch_verify_variance_enrichment_results.pkl", "wb") as f:
    pickle.dump(out, f)
print("\nDONE -- saved scratch_verify_variance_enrichment_results.pkl")
