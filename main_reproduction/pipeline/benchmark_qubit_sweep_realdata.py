"""
benchmark_qubit_sweep_realdata.py — rerun of benchmark_qubit_sweep.py's
n_qubits x projection sweep against real MiniLM sentence embeddings
(see benchmark_realdata_embeddings.py) instead of the synthetic Gaussian
corpus. Sweep logic, AgentMemory config, and query construction are
otherwise identical to benchmark_qubit_sweep.py -- only the corpus source
changed, to isolate that one variable.
"""

from __future__ import annotations

import time
import numpy as np
from agent_memory import AgentMemory
from benchmark_realdata_embeddings import make_real_corpus, CORPUS_PER_DIM_STD


def run_qubit_sweep_real(
    n_qubits_list: list[int],
    n_items: int = 24,
    n_clusters: int = 8,
    query_noise_multiplier: float = 1.2,
    n_seeds: int = 5,
    n_queries: int = 10,
) -> dict[tuple[int, str], float]:
    query_noise = query_noise_multiplier * CORPUS_PER_DIM_STD
    results = {}
    for n_qubits in n_qubits_list:
        for projection in ("random", "pca"):
            recalls = []
            t0 = time.perf_counter()
            for seed in range(n_seeds):
                corpus, keys = make_real_corpus(n_items, n_clusters, seed=seed)
                d = next(iter(corpus.values())).shape[0]
                rng = np.random.default_rng(seed + 1)
                mem = AgentMemory(
                    n_qubits=n_qubits, topology="chain", method="statevector", projection=projection
                )
                for k, v in corpus.items():
                    mem.store(k, v)
                for q_idx in range(n_queries):
                    relevant_key = keys[q_idx % len(keys)]
                    query_vec = corpus[relevant_key] + rng.normal(scale=query_noise, size=d)
                    m = mem.evaluate_retrieval(
                        query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx
                    )
                    recalls.append(m["recall@1_quantum"])
            elapsed = time.perf_counter() - t0
            mean_recall = float(np.mean(recalls))
            results[(n_qubits, projection)] = mean_recall
            print(f"  n_qubits={n_qubits:2d}  {projection:6s}  Recall@1={mean_recall:.3f}  "
                  f"({elapsed:.1f}s for {n_seeds * n_queries} query-runs)")
    return results


if __name__ == "__main__":
    print("--- Qubit-count sweep on REAL embeddings: random vs PCA projection ---")
    print("(d=384 all-MiniLM-L6-v2 embeddings, 20 Newsgroups, 24-item corpus, "
          "8 topics, statevector fidelity)\n")

    results = run_qubit_sweep_real([4, 6, 8, 10, 12, 14])

    print("\n--- Summary (real embeddings) ---")
    print(f"{'n_qubits':>8}  {'random':>8}  {'pca':>8}  {'gap':>6}")
    n_qubits_list = sorted(set(k[0] for k in results))
    for nq in n_qubits_list:
        r = results[(nq, "random")]
        p = results[(nq, "pca")]
        print(f"{nq:>8}  {r:>8.3f}  {p:>8.3f}  {p - r:>6.3f}")
