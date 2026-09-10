"""
benchmark_query_noise_realdata_hard.py — query-noise sweep on the HARDER
real corpus (see benchmark_realdata_embeddings_hard.py), extended to
higher noise multipliers than benchmark_query_noise_realdata.py used.

Motivation: on the easy real corpus (8 maximally-distinct topics, mult up
to 5.0), classical recall never left the ceiling and PCA-quantum recall
never dropped below 0.98 -- so whether PCA shows any quantum-specific
noise sensitivity on real data was UNRESOLVED, not "no". A quick probe on
this harder corpus (confusable comp.*/rec.autos+motorcycles/sci.electronics
topics, 48 items) found classical and PCA-quantum recall both collapsing
somewhere around mult=4-16 (1.0 -> 0.0). The multiplier list below brackets
that transition directly, instead of stopping at a value chosen for the
easy corpus that turned out to be far below where anything interesting
happens on this one.
"""

from __future__ import annotations

import numpy as np
from agent_memory import AgentMemory
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD

NOISE_MULTIPLIERS = (0.2, 1.2, 2, 4, 6, 8, 10, 12, 16, 20)


def _mean_recall_hard(
    n_qubits: int,
    projection: str,
    query_noise_multiplier: float,
    n_items: int = 48,
    n_clusters: int = 8,
    n_seeds: int = 5,
    n_queries: int = 10,
) -> tuple[float, float]:
    query_noise = query_noise_multiplier * CORPUS_PER_DIM_STD
    recalls_q, recalls_c = [], []
    for seed in range(n_seeds):
        corpus, keys = make_hard_real_corpus(n_items, n_clusters, seed=seed)
        d = next(iter(corpus.values())).shape[0]
        rng = np.random.default_rng(seed + 1)
        mem = AgentMemory(n_qubits=n_qubits, topology="chain", method="statevector", projection=projection)
        for k, v in corpus.items():
            mem.store(k, v)
        for q_idx in range(n_queries):
            relevant_key = keys[q_idx % len(keys)]
            query_vec = corpus[relevant_key] + rng.normal(scale=query_noise, size=d)
            m = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
            recalls_q.append(m["recall@1_quantum"])
            recalls_c.append(m["recall@1_classical"])
    return float(np.mean(recalls_q)), float(np.mean(recalls_c))


def run_query_noise_sweep_hard(n_qubits: int = 8) -> None:
    print(f"--- Query noise sweep on HARD real embeddings (n_qubits={n_qubits}) ---")
    print(f"(corpus per-dim std={CORPUS_PER_DIM_STD:.5f}; multiplier x that std = actual noise scale)")
    print(f"{'mult':>7}  {'random_q':>9}  {'pca_q':>7}  {'classical':>10}")
    for mult in NOISE_MULTIPLIERS:
        r_random, r_classical = _mean_recall_hard(n_qubits, "random", mult)
        r_pca, r_classical_pca = _mean_recall_hard(n_qubits, "pca", mult)
        print(f"{mult:>7.1f}  {r_random:>9.3f}  {r_pca:>7.3f}  {r_classical:>10.3f}")


if __name__ == "__main__":
    run_query_noise_sweep_hard()
