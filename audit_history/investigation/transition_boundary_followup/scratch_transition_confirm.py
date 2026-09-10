import pickle, time
import numpy as np
from agent_memory import AgentMemory

CANDIDATE_SIZES = (144, 192)
CORPUS_STD = 0.04882
MULTS = (1.2, 2.0, 4.0)
N = 100

def wald_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return 1.96 * se

all_results = {}

for n_items in CANDIDATE_SIZES:
    with open(f"scratch_transition_corpus_{n_items}.pkl", "rb") as f:
        d = pickle.load(f)
    corpus, keys = d["corpus"], d["keys"]
    d_dim = next(iter(corpus.values())).shape[0]

    print(f"\n=== CONFIRMATION n_items={n_items} (n_qubits=4, PCA + classical, n={N}/cell) ===")
    t0 = time.perf_counter()
    row_results = {}
    for mult in MULTS:
        noise = mult * CORPUS_STD
        mem = AgentMemory(n_qubits=4, topology="chain", method="statevector", projection="pca")
        for k, v in corpus.items():
            mem.store(k, v)
        rng = np.random.default_rng(1)
        rq, rc = [], []
        for q_idx in range(N):
            relevant_key = keys[q_idx % len(keys)]
            query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
            m = mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
            rq.append(m["recall@1_quantum"]); rc.append(m["recall@1_classical"])
        pca_mean = float(np.mean(rq)); classical_mean = float(np.mean(rc))
        pca_ci = wald_ci(pca_mean, N); classical_ci = wald_ci(classical_mean, N)
        gap = classical_mean - pca_mean
        lo_p, hi_p = pca_mean - pca_ci, pca_mean + pca_ci
        lo_c, hi_c = classical_mean - classical_ci, classical_mean + classical_ci
        overlap = not (hi_p < lo_c or hi_c < lo_p)
        row_results[mult] = {
            "pca": pca_mean, "pca_ci": pca_ci,
            "classical": classical_mean, "classical_ci": classical_ci,
            "gap": gap, "overlap": overlap,
        }
        print(f"  mult={mult:>4.1f}  pca={pca_mean:.3f}+/-{pca_ci:.3f}  "
              f"classical={classical_mean:.3f}+/-{classical_ci:.3f}  gap={gap:.3f}  "
              f"{'OVERLAPPING' if overlap else 'non-overlapping'}")
    elapsed = time.perf_counter() - t0
    print(f"  ({elapsed:.1f}s for n_items={n_items})")
    all_results[n_items] = row_results

with open("scratch_transition_confirm_results.pkl", "wb") as f:
    pickle.dump(all_results, f)

print("\nDONE")
