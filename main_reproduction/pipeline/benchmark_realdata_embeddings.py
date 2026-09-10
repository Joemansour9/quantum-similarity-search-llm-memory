"""
benchmark_realdata_embeddings.py — Phase 2/3: real dense-embedding generalization check

benchmark_tfidf_realdata.py's docstring noted huggingface.co was unreachable
at that point, so it fell back to sparse TF-IDF vectors as a partial
substitute for real dense embeddings. huggingface.co is reachable here
(confirmed empirically: model download succeeds) — so this runs the real
thing: sentence-transformers' all-MiniLM-L6-v2
(384-dim, unit-normalized dense embeddings) over real, naturally-occurring
text (20 Newsgroups posts, 8 topics), instead of the synthetic Gaussian-blob
corpus used by benchmark_qubit_sweep.py / benchmark_task_difficulty.py.

Corpus construction mirrors make_synthetic_corpus()'s shape exactly
(n_items=24, n_clusters=8, i.e. 3 real documents per topic) so the sweep
logic below is otherwise IDENTICAL to the synthetic scripts — same
AgentMemory config, same query construction (Gaussian noise added directly
to the stored embedding), same metrics. The only variable changed is where
the base corpus vectors come from. This isolates "does real corpus
structure change the finding" from any other confound.

One necessary adaptation: query_noise scale. The synthetic corpus draws
within-cluster noise from N(0, 1.0) per dimension, so its query_noise sweep
values (0.2 .. 5.0) are calibrated relative to a per-dimension std of 1.0.
Real MiniLM embeddings are unit-normalized 384-dim vectors with a much
smaller per-dimension std (empirically ~0.036 for this corpus). Using the
literal synthetic noise values here would either do nothing (small values)
or completely destroy the signal (large values) — not a fair test. Instead,
noise multipliers are the same relative sweep (0.2 .. 5.0), scaled by this
corpus's actual per-dimension std, so "multiplier=1.2" means the same
relative perturbation strength in both experiments.

Deliberately NOT reproduced here: the cluster_spread (corpus-complexity)
sweep from benchmark_task_difficulty.py. Real embedding space doesn't
expose a "cluster separation" knob the way synthetic Gaussian centers do
(you can't dial topic similarity continuously), and the replication targets
here are specifically the PCA-over-random and query-noise-sensitivity
findings.
"""

from __future__ import annotations

import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

from agent_memory import AgentMemory

CATEGORIES = [
    "sci.space", "rec.sport.baseball", "comp.graphics", "talk.politics.guns",
    "sci.med", "rec.autos", "soc.religion.christian", "misc.forsale",
]

_POOL_PER_CATEGORY = 40  # enough for 5 non-degenerate seeds x 3 items/category
_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_pool() -> tuple[dict[str, list[str]], np.ndarray, dict[str, list[np.ndarray]]]:
    """
    Fetch real 20 Newsgroups posts and embed a pool per category with
    all-MiniLM-L6-v2. Returns (texts_by_category, corpus_per_dim_std,
    embeddings_by_category).
    """
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


# Loaded once at import time -- embedding the pool is the expensive part
# (network + model forward passes), sampling from it per-seed is cheap.
print("Loading real corpus (20 Newsgroups) and encoding with all-MiniLM-L6-v2 ...")
_T0 = time.perf_counter()
_TEXTS_BY_CAT, CORPUS_PER_DIM_STD, _EMB_BY_CAT = _load_pool()
print(f"  done in {time.perf_counter() - _T0:.1f}s. "
      f"corpus per-dim std = {CORPUS_PER_DIM_STD:.5f} (embedding dim=384, unit-normalized)\n")


def make_real_corpus(
    n_items: int = 24, n_clusters: int = 8, seed: int = 0, **_ignored
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Drop-in real-data replacement for benchmark_projection.make_synthetic_corpus().
    Same signature shape (extra synthetic-only kwargs like d/cluster_spread
    are accepted and ignored so it can be swapped into the same call sites).

    Draws n_items // n_clusters real, distinct documents per topic category
    (embedded with all-MiniLM-L6-v2), using `seed` to select which documents
    from each category's pool -- so different seeds genuinely sample
    different real text, mirroring how different seeds redraw the synthetic
    Gaussian corpus.
    """
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
