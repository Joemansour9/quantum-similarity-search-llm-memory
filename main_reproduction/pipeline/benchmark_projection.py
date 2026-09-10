"""
benchmark_projection.py — Phase 2/3: Random projection vs PCA

The comparison deliberately deferred until there was an actual retrieval
task to benchmark against (see q_encoder.py / agent_memory.py notes).
Now that agent_memory.py has evaluate_retrieval() with Recall@K, MRR,
Top-1, and Spearman, this runs a real head-to-head instead of eyeballing
raw fidelity numbers.

Ground truth: a synthetic dataset of cluster "documents" where each
query is a noisy copy of one specific stored item — so the correct
top-1 match is known for every query, and Recall@K / MRR are measuring
something real, not guessed.
"""

from __future__ import annotations

import numpy as np
from agent_memory import AgentMemory


def make_synthetic_corpus(
    n_items: int, d: int, n_clusters: int, seed: int = 0, cluster_spread: float = 8.0
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    n_items embeddings drawn from n_clusters Gaussian blobs in R^d.
    Returns {key: embedding} and the list of keys in cluster order, so
    cluster membership (= "topic") is known ground truth for evaluating
    retrieval quality.

    cluster_spread controls how well-separated the clusters are — too
    large and every method gets Recall@1=1.0 trivially (a ceiling
    effect, same failure mode as Paper 3's I6-I9 before that was caught
    and corrected); too small and the task becomes unsolvable noise.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=cluster_spread, size=(n_clusters, d))
    items = {}
    keys = []
    for i in range(n_items):
        cluster = i % n_clusters
        emb = centers[cluster] + rng.normal(scale=1.0, size=d)
        key = f"item_{i}_c{cluster}"
        items[key] = emb
        keys.append(key)
    return items, keys


def run_benchmark(
    n_items: int = 24,
    d: int = 32,
    n_clusters: int = 8,
    n_qubits: int = 6,
    n_queries: int = 10,
    cluster_spread: float = 2.0,
    query_noise: float = 1.2,
    seed: int = 0,
) -> None:
    corpus, keys = make_synthetic_corpus(n_items, d, n_clusters, seed=seed, cluster_spread=cluster_spread)
    rng = np.random.default_rng(seed + 1)

    results = {"random": [], "pca": []}

    for projection in ("random", "pca"):
        mem = AgentMemory(
            n_qubits=n_qubits, topology="chain", method="statevector", projection=projection
        )
        for key, emb in corpus.items():
            mem.store(key, emb)

        for q_idx in range(n_queries):
            # Query = noisy copy of a specific stored item -> that item
            # IS the known-correct top-1 answer.
            relevant_key = keys[q_idx % len(keys)]
            query_vec = corpus[relevant_key] + rng.normal(scale=query_noise, size=d)
            metrics = mem.evaluate_retrieval(
                query_vec, relevant_key=relevant_key, k_values=(1, 3), seed=q_idx
            )
            results[projection].append(metrics)

    print(f"--- Benchmark: {n_items} items, {n_clusters} clusters (spread={cluster_spread}), "
          f"{n_queries} queries (noise={query_noise}), d={d} -> n_qubits={n_qubits} ---\n")

    for projection in ("random", "pca"):
        rows = results[projection]
        recall1_q = np.mean([r["recall@1_quantum"] for r in rows])
        recall1_c = np.mean([r["recall@1_classical"] for r in rows])
        recall3_q = np.mean([r["recall@3_quantum"] for r in rows])
        mrr_q = np.mean([r["mrr_quantum"] for r in rows])
        top1_agree = np.mean([r["top1_match"] for r in rows])

        print(f"[{projection.upper()} projection]")
        print(f"  Recall@1 (quantum):    {recall1_q:.3f}")
        print(f"  Recall@1 (classical):  {recall1_c:.3f}")
        print(f"  Recall@3 (quantum):    {recall3_q:.3f}")
        print(f"  MRR (quantum):         {mrr_q:.3f}")
        print(f"  Top-1 quantum/classical agreement: {top1_agree:.3f}")
        print()

    print("--- Framing ---")
    print("Both projections are being fed through the SAME encoder, entangler,")
    print("and retrieval pipeline — this isolates the effect of the projection")
    print("choice specifically, not a confound with anything else changing.")
    print(f"n_queries={n_queries} is still small; treat differences here as a")
    print("first signal, not a publishable result, per the sample-size caveats")
    print("already documented for Paper 3's small-N correlations.")


if __name__ == "__main__":
    run_benchmark()
