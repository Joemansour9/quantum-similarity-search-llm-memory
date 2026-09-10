"""
benchmark_query_noise_realdata.py — rerun of benchmark_task_difficulty.py's
query_noise sweep (the query-side task-difficulty axis) against real MiniLM
sentence embeddings instead of the synthetic Gaussian corpus. n_qubits fixed
at 8, matching benchmark_task_difficulty.py's choice (the setting where the
synthetic qubit sweep found the largest random-vs-PCA gap).

The cluster_spread (corpus-complexity) sweep from benchmark_task_difficulty.py
is NOT reproduced here -- real embedding space has no continuous "cluster
separation" knob the way synthetic Gaussian centers do. Only the query-noise
axis is replicated, which is the one used here.

query_noise here is a MULTIPLIER on this corpus's actual per-dimension std
(CORPUS_PER_DIM_STD from benchmark_realdata_embeddings.py), using the same
multiplier values benchmark_task_difficulty.py used as raw noise scale
against its unit-std synthetic corpus -- so multiplier=1.2 means the same
relative perturbation strength in both experiments, even though the actual
Gaussian scale differs (real embeddings have much smaller per-dim std).
"""

from __future__ import annotations

import numpy as np
from agent_memory import AgentMemory
from benchmark_realdata_embeddings import make_real_corpus, CORPUS_PER_DIM_STD


def _mean_recall_real(
    n_qubits: int,
    projection: str,
    query_noise_multiplier: float,
    n_items: int = 24,
    n_clusters: int = 8,
    n_seeds: int = 5,
    n_queries: int = 10,
) -> tuple[float, float]:
    """Returns (mean_recall_at_1_quantum, mean_recall_at_1_classical)."""
    query_noise = query_noise_multiplier * CORPUS_PER_DIM_STD
    recalls_q, recalls_c = [], []
    for seed in range(n_seeds):
        corpus, keys = make_real_corpus(n_items, n_clusters, seed=seed)
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


def run_query_noise_sweep_real(n_qubits: int = 8) -> None:
    print(f"--- Query noise sweep on REAL embeddings (n_qubits={n_qubits}) ---")
    print(f"(corpus per-dim std={CORPUS_PER_DIM_STD:.5f}; multiplier x that std = actual noise scale)")
    print(f"{'mult':>7}  {'random_q':>9}  {'pca_q':>7}  {'classical':>10}")
    for mult in (0.2, 0.6, 1.2, 2.0, 3.0, 5.0):
        r_random, r_classical = _mean_recall_real(n_qubits, "random", mult)
        r_pca, _ = _mean_recall_real(n_qubits, "pca", mult)
        print(f"{mult:>7.1f}  {r_random:>9.3f}  {r_pca:>7.3f}  {r_classical:>10.3f}")


if __name__ == "__main__":
    run_query_noise_sweep_real()
