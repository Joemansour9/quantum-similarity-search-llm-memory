import pickle, time
import numpy as np
from agent_memory import AgentMemory
from q_encoder import build_feature_map, fit_projection_scale, fit_pca_projection_scale
from q_search import statevector_fidelity_from_circuits

with open("scratch_large_corpus_480.pkl", "rb") as f:
    d = pickle.load(f)
corpus, keys = d["corpus"], d["keys"]
d_dim = next(iter(corpus.values())).shape[0]
CORPUS_STD = 0.04882

# --- 1. Qubit sweep ---
print("=== Qubit sweep (480-item corpus, n_queries=20/cell) ===")
mult = 1.2
noise = mult * CORPUS_STD
qubit_results = {}
for n_qubits in (4, 6, 8, 10, 12, 14):
    for proj in ("random", "pca"):
        mem = AgentMemory(n_qubits=n_qubits, topology="chain", method="statevector", projection=proj)
        for k, v in corpus.items():
            mem.store(k, v)
        rng = np.random.default_rng(1)
        t0 = time.perf_counter()
        recalls = []
        for q_idx in range(20):
            relevant_key = keys[q_idx % len(keys)]
            query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
            m = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
            recalls.append(m["recall@1_quantum"])
        elapsed = time.perf_counter() - t0
        mean_recall = float(np.mean(recalls))
        qubit_results[(n_qubits, proj)] = mean_recall
        print(f"  n_qubits={n_qubits:2d}  {proj:6s}  Recall@1={mean_recall:.3f}  ({elapsed:.1f}s)")

print()
print(f"{'n_qubits':>8}  {'random':>8}  {'pca':>8}  {'gap':>6}")
for nq in sorted(set(k[0] for k in qubit_results)):
    r = qubit_results[(nq, "random")]; p = qubit_results[(nq, "pca")]
    print(f"{nq:>8}  {r:>8.3f}  {p:>8.3f}  {p-r:>6.3f}")

with open("scratch_large_corpus_qubit_sweep.pkl", "wb") as f:
    pickle.dump(qubit_results, f)

# --- 2. Query-noise sweep at n_qubits=4 ---
print("\n=== Query-noise sweep (480-item corpus, n_qubits=4, n_queries=20/cell) ===")
NOISE_MULTIPLIERS = (0.2, 1.2, 2, 4, 6, 8, 10, 12, 16, 20)
noise_results = {}
for m in NOISE_MULTIPLIERS:
    qn = m * CORPUS_STD
    row = {}
    for proj in ("random", "pca"):
        mem = AgentMemory(n_qubits=4, topology="chain", method="statevector", projection=proj)
        for k, v in corpus.items():
            mem.store(k, v)
        rng = np.random.default_rng(1)
        rq, rc = [], []
        for q_idx in range(20):
            relevant_key = keys[q_idx % len(keys)]
            query_vec = corpus[relevant_key] + rng.normal(scale=qn, size=d_dim)
            r = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
            rq.append(r["recall@1_quantum"]); rc.append(r["recall@1_classical"])
        row[proj] = float(np.mean(rq))
        row["classical"] = float(np.mean(rc))
    noise_results[m] = row
    print(f"  mult={m:>5.1f}  random={row['random']:.3f}  pca={row['pca']:.3f}  classical={row['classical']:.3f}")

with open("scratch_large_corpus_noise_sweep.pkl", "wb") as f:
    pickle.dump(noise_results, f)

# --- 3. Candidate-scarcity scan (all 480 queries, n_qubits=4, mult=1.2) ---
print("\n=== Candidate-scarcity scan: exact statevector fidelity, all 480 self-match queries ===")
n_qubits = 4
edges = [(i, i + 1) for i in range(n_qubits - 1)]
candidate_batch = np.stack([corpus[k] for k in keys])
rng = np.random.default_rng(1)
pca_fids, rand_fids = [], []
t0 = time.perf_counter()
for q_idx in range(len(keys)):
    relevant_key = keys[q_idx % len(keys)]
    query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
    candidate_vec = candidate_batch[q_idx % len(keys)]
    full_batch = np.vstack([query_vec[None, :], candidate_batch])

    R, lo, hi = fit_pca_projection_scale(full_batch, n_qubits)
    qc = build_feature_map(query_vec, n_qubits, (lo, hi), edges, R, entangler="rzz")
    fid, _ = statevector_fidelity_from_circuits(qc, candidate_vec, n_qubits, (lo, hi), edges, R, entangler="rzz")
    pca_fids.append(fid)

    R2, lo2, hi2 = fit_projection_scale(full_batch, n_qubits, seed=q_idx)
    qc2 = build_feature_map(query_vec, n_qubits, (lo2, hi2), edges, R2, entangler="rzz")
    fid2, _ = statevector_fidelity_from_circuits(qc2, candidate_vec, n_qubits, (lo2, hi2), edges, R2, entangler="rzz")
    rand_fids.append(fid2)

    if (q_idx + 1) % 100 == 0:
        print(f"  ... {q_idx+1}/480 done ({time.perf_counter()-t0:.0f}s elapsed)")

pca_fids = np.array(pca_fids)
rand_fids = np.array(rand_fids)
pca_above = int((pca_fids >= 0.5).sum())
rand_above = int((rand_fids >= 0.5).sum())
print(f"\nPCA: {pca_above}/480 queries clear sim_fid>=0.5 ({100*pca_above/480:.1f}%)")
print(f"Random: {rand_above}/480 queries clear sim_fid>=0.5 ({100*rand_above/480:.1f}%)")
print(f"(original 48-item corpus: 8/48 = 16.7% for random)")

with open("scratch_large_corpus_scarcity.pkl", "wb") as f:
    pickle.dump({"pca_fids": pca_fids, "rand_fids": rand_fids}, f)

print("\nALL DONE")
