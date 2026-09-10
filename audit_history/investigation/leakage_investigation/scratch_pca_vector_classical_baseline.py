"""
scratch_pca_vector_classical_baseline.py — third curve for the existing
PCA-projection test cases: classical cosine similarity computed directly on
the n-dimensional PCA-projected vector (R @ x), NOT the raw embedding and
NOT going through any quantum encoding/SWAP test.

For every query this uses the EXACT SAME R that the quantum pathway used
(fit_pca_projection_scale is a deterministic function of full_batch =
vstack([query_vec, candidate_batch]) and n_qubits only -- no seed
dependence -- so calling it again here with the same full_batch reproduces
AgentMemory's internal R exactly, see q_encoder.fit_pca_projection_scale
and agent_memory.AgentMemory.query()).

Pure classical computation throughout (cosine_similarity + statevector
SIMULATION for the existing quantum curve, which was already
simulator-only in every prior report) -- no QPU time used.
"""
import pickle, time
import numpy as np
from agent_memory import AgentMemory
from q_encoder import fit_pca_projection_scale
from q_search import cosine_similarity, recall_at_k
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD

CORPUS_STD = CORPUS_PER_DIM_STD  # 0.04882, shared across all hard-corpus sizes


def classical_pca_recall1(query_vec, candidate_batch, keys, relevant_idx, n_qubits):
    full_batch = np.vstack([query_vec[None, :], candidate_batch])
    R, _, _ = fit_pca_projection_scale(full_batch, n_qubits)
    query_proj = R @ query_vec
    candidates_proj = candidate_batch @ R.T
    cosines = np.array([cosine_similarity(query_proj, candidates_proj[i]) for i in range(len(keys))])
    rank = np.argsort(-cosines)
    return recall_at_k(rank, relevant_idx, 1)


results = {}

# ============================================================
# 1. Qubit sweep, 48-item corpus (mirrors benchmark_qubit_sweep_realdata_hard.py)
# ============================================================
print("=== [1] Qubit sweep, 48-item corpus (mult=1.2, n_seeds=5, n_queries=10 => n=50/cell) ===")
mult = 1.2
row1 = {}
for n_qubits in (4, 6, 8, 10, 12, 14):
    t0 = time.perf_counter()
    rq, rc_raw, rc_pca = [], [], []
    for seed in range(5):
        corpus, keys = make_hard_real_corpus(48, 8, seed=seed)
        d_dim = next(iter(corpus.values())).shape[0]
        candidate_batch = np.stack([corpus[k] for k in keys])
        rng = np.random.default_rng(seed + 1)
        mem = AgentMemory(n_qubits=n_qubits, topology="chain", method="statevector", projection="pca")
        for k, v in corpus.items():
            mem.store(k, v)
        noise = mult * CORPUS_STD
        for q_idx in range(10):
            relevant_key = keys[q_idx % len(keys)]
            relevant_idx = keys.index(relevant_key)
            query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
            m = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
            rq.append(m["recall@1_quantum"])
            rc_raw.append(m["recall@1_classical"])
            rc_pca.append(classical_pca_recall1(query_vec, candidate_batch, keys, relevant_idx, n_qubits))
    elapsed = time.perf_counter() - t0
    row1[n_qubits] = {
        "quantum_pca": float(np.mean(rq)),
        "classical_raw": float(np.mean(rc_raw)),
        "classical_pca_vec": float(np.mean(rc_pca)),
    }
    print(f"  n_qubits={n_qubits:2d}  quantum(PCA)={row1[n_qubits]['quantum_pca']:.3f}  "
          f"classical(raw)={row1[n_qubits]['classical_raw']:.3f}  "
          f"classical(PCA-vec)={row1[n_qubits]['classical_pca_vec']:.3f}  ({elapsed:.1f}s)")
results["qubit_sweep_48"] = row1

# ============================================================
# 2. Qubit sweep, 480-item corpus (mirrors scratch_large_corpus_sweep.py)
# ============================================================
print("\n=== [2] Qubit sweep, 480-item corpus (mult=1.2, n=20/cell) ===")
with open("scratch_large_corpus_480.pkl", "rb") as f:
    d480 = pickle.load(f)
corpus480, keys480 = d480["corpus"], d480["keys"]
d_dim480 = next(iter(corpus480.values())).shape[0]
candidate_batch480 = np.stack([corpus480[k] for k in keys480])

row2 = {}
for n_qubits in (4, 6, 8, 10, 12, 14):
    t0 = time.perf_counter()
    mem = AgentMemory(n_qubits=n_qubits, topology="chain", method="statevector", projection="pca")
    for k, v in corpus480.items():
        mem.store(k, v)
    rng = np.random.default_rng(1)
    noise = mult * CORPUS_STD
    rq, rc_raw, rc_pca = [], [], []
    for q_idx in range(20):
        relevant_key = keys480[q_idx % len(keys480)]
        relevant_idx = keys480.index(relevant_key)
        query_vec = corpus480[relevant_key] + rng.normal(scale=noise, size=d_dim480)
        m = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
        rq.append(m["recall@1_quantum"])
        rc_raw.append(m["recall@1_classical"])
        rc_pca.append(classical_pca_recall1(query_vec, candidate_batch480, keys480, relevant_idx, n_qubits))
    elapsed = time.perf_counter() - t0
    row2[n_qubits] = {
        "quantum_pca": float(np.mean(rq)),
        "classical_raw": float(np.mean(rc_raw)),
        "classical_pca_vec": float(np.mean(rc_pca)),
    }
    print(f"  n_qubits={n_qubits:2d}  quantum(PCA)={row2[n_qubits]['quantum_pca']:.3f}  "
          f"classical(raw)={row2[n_qubits]['classical_raw']:.3f}  "
          f"classical(PCA-vec)={row2[n_qubits]['classical_pca_vec']:.3f}  ({elapsed:.1f}s)")
results["qubit_sweep_480"] = row2

# ============================================================
# 3. Query-noise sweep, 48-item corpus (mirrors benchmark_query_noise_realdata_hard.py)
# ============================================================
print("\n=== [3] Query-noise sweep, 48-item corpus (n_qubits=4, n_seeds=5, n_queries=10 => n=50/cell) ===")
NOISE_MULTIPLIERS = (0.2, 1.2, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
n_qubits = 4
row3 = {}
for m in NOISE_MULTIPLIERS:
    t0 = time.perf_counter()
    noise = m * CORPUS_STD
    rq, rc_raw, rc_pca = [], [], []
    for seed in range(5):
        corpus, keys = make_hard_real_corpus(48, 8, seed=seed)
        d_dim = next(iter(corpus.values())).shape[0]
        candidate_batch = np.stack([corpus[k] for k in keys])
        rng = np.random.default_rng(seed + 1)
        mem = AgentMemory(n_qubits=n_qubits, topology="chain", method="statevector", projection="pca")
        for k, v in corpus.items():
            mem.store(k, v)
        for q_idx in range(10):
            relevant_key = keys[q_idx % len(keys)]
            relevant_idx = keys.index(relevant_key)
            query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
            met = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
            rq.append(met["recall@1_quantum"])
            rc_raw.append(met["recall@1_classical"])
            rc_pca.append(classical_pca_recall1(query_vec, candidate_batch, keys, relevant_idx, n_qubits))
    elapsed = time.perf_counter() - t0
    row3[m] = {
        "quantum_pca": float(np.mean(rq)),
        "classical_raw": float(np.mean(rc_raw)),
        "classical_pca_vec": float(np.mean(rc_pca)),
    }
    print(f"  mult={m:>5.1f}  quantum(PCA)={row3[m]['quantum_pca']:.3f}  "
          f"classical(raw)={row3[m]['classical_raw']:.3f}  "
          f"classical(PCA-vec)={row3[m]['classical_pca_vec']:.3f}  ({elapsed:.1f}s)")
results["noise_sweep_48"] = row3

# ============================================================
# 4. Query-noise sweep, 480-item corpus (mirrors scratch_large_corpus_sweep.py)
# ============================================================
print("\n=== [4] Query-noise sweep, 480-item corpus (n_qubits=4, n=20/cell) ===")
row4 = {}
for m in NOISE_MULTIPLIERS:
    t0 = time.perf_counter()
    noise = m * CORPUS_STD
    mem = AgentMemory(n_qubits=n_qubits, topology="chain", method="statevector", projection="pca")
    for k, v in corpus480.items():
        mem.store(k, v)
    rng = np.random.default_rng(1)
    rq, rc_raw, rc_pca = [], [], []
    for q_idx in range(20):
        relevant_key = keys480[q_idx % len(keys480)]
        relevant_idx = keys480.index(relevant_key)
        query_vec = corpus480[relevant_key] + rng.normal(scale=noise, size=d_dim480)
        met = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
        rq.append(met["recall@1_quantum"])
        rc_raw.append(met["recall@1_classical"])
        rc_pca.append(classical_pca_recall1(query_vec, candidate_batch480, keys480, relevant_idx, n_qubits))
    elapsed = time.perf_counter() - t0
    row4[m] = {
        "quantum_pca": float(np.mean(rq)),
        "classical_raw": float(np.mean(rc_raw)),
        "classical_pca_vec": float(np.mean(rc_pca)),
    }
    print(f"  mult={m:>5.1f}  quantum(PCA)={row4[m]['quantum_pca']:.3f}  "
          f"classical(raw)={row4[m]['classical_raw']:.3f}  "
          f"classical(PCA-vec)={row4[m]['classical_pca_vec']:.3f}  ({elapsed:.1f}s)")
results["noise_sweep_480"] = row4

with open("scratch_pca_vector_classical_baseline_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nALL DONE")
