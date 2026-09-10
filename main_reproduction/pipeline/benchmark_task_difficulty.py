"""
benchmark_task_difficulty.py — Phase 2/3: corpus complexity + query noise sweep

Completes the four-knob sweep (n_qubits, projection, corpus complexity,
query noise) — n_qubits and projection were covered in
benchmark_qubit_sweep.py; this covers the remaining two, holding
n_qubits fixed at 8 (where benchmark_qubit_sweep.py found the largest
random-vs-PCA gap, making it the most sensitive setting for detecting
further effects).

Two axes:
  - cluster_spread: how well-separated the corpus clusters are (task
    difficulty from the DATA side — tighter clusters = harder to
    distinguish "topics").
  - query_noise: how much the query vector deviates from its true match
    (task difficulty from the QUERY side — noisier queries = harder to
    identify the right target even with well-separated data).

These are conceptually different failure modes and can move Recall@1
independently, so they're swept separately rather than conflated into
one "difficulty" knob.
"""

from __future__ import annotations

import numpy as np
from agent_memory import AgentMemory
from benchmark_projection import make_synthetic_corpus


def _mean_recall(
    n_qubits: int,
    projection: str,
    cluster_spread: float,
    query_noise: float,
    n_items: int = 24,
    d: int = 32,
    n_clusters: int = 8,
    n_seeds: int = 5,
    n_queries: int = 10,
) -> tuple[float, float]:
    """Returns (mean_recall_at_1_quantum, mean_recall_at_1_classical)."""
    recalls_q, recalls_c = [], []
    for seed in range(n_seeds):
        corpus, keys = make_synthetic_corpus(
            n_items, d, n_clusters, seed=seed, cluster_spread=cluster_spread
        )
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


def run_corpus_complexity_sweep(n_qubits: int = 8, query_noise: float = 1.2) -> None:
    """Vary cluster_spread: smaller = clusters closer together = harder task."""
    print(f"--- Corpus complexity sweep (n_qubits={n_qubits}, query_noise={query_noise} fixed) ---")
    print(f"{'spread':>7}  {'random_q':>9}  {'pca_q':>7}  {'classical':>10}")
    for spread in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        r_random, r_classical = _mean_recall(n_qubits, "random", spread, query_noise)
        r_pca, _ = _mean_recall(n_qubits, "pca", spread, query_noise)
        print(f"{spread:>7.1f}  {r_random:>9.3f}  {r_pca:>7.3f}  {r_classical:>10.3f}")


def run_query_noise_sweep(n_qubits: int = 8, cluster_spread: float = 2.0) -> None:
    """Vary query_noise: larger = query deviates further from its true match."""
    print(f"\n--- Query noise sweep (n_qubits={n_qubits}, cluster_spread={cluster_spread} fixed) ---")
    print(f"{'noise':>7}  {'random_q':>9}  {'pca_q':>7}  {'classical':>10}")
    for noise in (0.2, 0.6, 1.2, 2.0, 3.0, 5.0):
        r_random, r_classical = _mean_recall(n_qubits, "random", cluster_spread, noise)
        r_pca, _ = _mean_recall(n_qubits, "pca", cluster_spread, noise)
        print(f"{noise:>7.1f}  {r_random:>9.3f}  {r_pca:>7.3f}  {r_classical:>10.3f}")


if __name__ == "__main__":
    run_corpus_complexity_sweep()
    run_query_noise_sweep()

    print("\n--- Interpretation notes (fill in after inspecting the tables above) ---")
    print("- If classical also degrades at the same difficulty level as quantum,")
    print("  the task itself is becoming genuinely harder (not a quantum-specific")
    print("  weakness) — expected and uninteresting.")
    print("- If quantum degrades notably FASTER than classical at a given knob,")
    print("  that isolates a real quantum-encoding weakness on that specific axis")
    print("  (compression sensitivity to cluster tightness, or to query noise).")
    print("- If random projection degrades faster than PCA specifically as task")
    print("  difficulty increases, that would mean PCA's advantage isn't fixed —")
    print("  it may matter MORE exactly when the task gets harder, which is the")
    print("  regime that matters most for a real deployment.")
