import pickle, time
import numpy as np
from agent_memory import AgentMemory

SIZES = (96, 144, 192, 240, 320, 400)
CORPUS_STD = 0.04882
NOISE_MULTIPLIERS = (0.2, 1.2, 2.0, 4.0, 6.0, 8.0)

all_results = {}

for n in SIZES:
    with open(f"scratch_transition_corpus_{n}.pkl", "rb") as f:
        d = pickle.load(f)
    corpus, keys = d["corpus"], d["keys"]
    d_dim = next(iter(corpus.values())).shape[0]

    print(f"\n=== n_items={n} (n_qubits=4, PCA + classical, n=20/cell) ===")
    t0 = time.perf_counter()
    row_results = {}
    for mult in NOISE_MULTIPLIERS:
        noise = mult * CORPUS_STD
        mem = AgentMemory(n_qubits=4, topology="chain", method="statevector", projection="pca")
        for k, v in corpus.items():
            mem.store(k, v)
        rng = np.random.default_rng(1)
        rq, rc = [], []
        for q_idx in range(20):
            relevant_key = keys[q_idx % len(keys)]
            query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
            m = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
            rq.append(m["recall@1_quantum"]); rc.append(m["recall@1_classical"])
        pca_mean = float(np.mean(rq))
        classical_mean = float(np.mean(rc))
        gap = classical_mean - pca_mean
        row_results[mult] = {"pca": pca_mean, "classical": classical_mean, "gap": gap}
        print(f"  mult={mult:>5.1f}  pca={pca_mean:.3f}  classical={classical_mean:.3f}  gap={gap:.3f}")
    elapsed = time.perf_counter() - t0
    print(f"  ({elapsed:.1f}s for n_items={n})")
    all_results[n] = row_results

with open("scratch_transition_sweep_results.pkl", "wb") as f:
    pickle.dump(all_results, f)

print("\n\n=== SUMMARY: PCA-classical gap by corpus size ===")
header = "n_items".rjust(8) + "".join(f"  mult={m:<5.1f}" for m in NOISE_MULTIPLIERS)
print(header)
for n in SIZES:
    row = all_results[n]
    line = f"{n:>8}" + "".join(f"  {row[m]['gap']:>10.3f}" for m in NOISE_MULTIPLIERS)
    print(line)

print("\nDONE")
