"""
benchmark_realdata_embeddings_hard.py — harder real-corpus variant

benchmark_realdata_embeddings.py used 8 maximally-distinct 20 Newsgroups
topics (space, baseball, graphics, guns, medicine, autos, religion,
forsale). On that easy corpus, PCA-quantum recall never dropped below 0.98
across the entire query-noise sweep (mult up to 5.0) -- so the "does PCA
also show quantum-specific noise sensitivity, like it did on synthetic
data" question came back unresolved rather than answered: we never found
where PCA's recall actually starts falling.

This variant swaps in 8 topics chosen to be genuinely confusable with each
other instead of maximally distinct:
  - 5 comp.* categories (graphics / ms-windows / ibm hardware / mac
    hardware / windows.x) -- all "using or configuring a computer",
    heavy vocabulary overlap.
  - rec.autos + rec.motorcycles -- both vehicle maintenance/discussion,
    overlapping vocabulary.
  - sci.electronics -- overlaps with the hardware categories.
This is a harder classical discrimination task by construction (more
semantically similar topics), not just a bigger corpus.

Also increases documents per topic from 3 (n_items=24) to 6 (n_items=48),
to increase the retrieval task's difficulty -- a larger, harder retrieval set,
not just harder topics.
"""

from __future__ import annotations

import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

CATEGORIES = [
    "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware", "comp.windows.x", "rec.autos", "rec.motorcycles",
    "sci.electronics",
]

_POOL_PER_CATEGORY = 65  # raised from 60 to support a ~480-520 item large-corpus test
# (per_cluster=60-65 x 8 categories); verified min category has 400 posts after the
# 200-2000 char filter (train subset), so 65/category is well within available headroom.
_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_pool() -> tuple[dict[str, list[str]], float, dict[str, list[np.ndarray]]]:
    data = fetch_20newsgroups(
        subset="train", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for text, t in zip(data.data, data.target):
        name = data.target_names[t]
        text = text.strip()
        if 200 < len(text) < 2000:
            by_cat[name].append(text)

    model = SentenceTransformer(_MODEL_NAME)
    all_texts = []
    for cat in CATEGORIES:
        all_texts.extend(by_cat[cat][:_POOL_PER_CATEGORY])
    all_emb = model.encode(all_texts, batch_size=32, show_progress_bar=False)

    emb_by_cat: dict[str, list[np.ndarray]] = {}
    idx = 0
    for cat in CATEGORIES:
        n = min(_POOL_PER_CATEGORY, len(by_cat[cat]))
        emb_by_cat[cat] = list(all_emb[idx: idx + n])
        idx += n

    corpus_std = float(all_emb.std(axis=0).mean())
    return by_cat, corpus_std, emb_by_cat


print("Loading HARD real corpus (20 Newsgroups: comp.*/rec.autos+motorcycles/sci.electronics) "
      "and encoding with all-MiniLM-L6-v2 ...")
_T0 = time.perf_counter()
_TEXTS_BY_CAT, CORPUS_PER_DIM_STD, _EMB_BY_CAT = _load_pool()
print(f"  done in {time.perf_counter() - _T0:.1f}s. "
      f"corpus per-dim std = {CORPUS_PER_DIM_STD:.5f} (embedding dim=384, unit-normalized)\n")


def make_hard_real_corpus(
    n_items: int = 48, n_clusters: int = 8, seed: int = 0, **_ignored
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Same shape/contract as benchmark_realdata_embeddings.make_real_corpus()."""
    if n_clusters != len(CATEGORIES):
        raise ValueError(f"n_clusters must be {len(CATEGORIES)} to match loaded categories")
    per_cluster = n_items // n_clusters
    rng = np.random.default_rng(seed)

    items: dict[str, np.ndarray] = {}
    keys: list[str] = []
    for ci, cat in enumerate(CATEGORIES):
        pool = _EMB_BY_CAT[cat]
        chosen = rng.choice(len(pool), size=per_cluster, replace=False)
        for j, idx in enumerate(chosen):
            key = f"item_{ci * per_cluster + j}_c{ci}"
            items[key] = np.asarray(pool[idx], dtype=float)
            keys.append(key)
    return items, keys
