import pickle, time
import numpy as np
from agent_memory import AgentMemory

SIZES = (96, 144, 192, 240, 320, 400)
CORPUS_STD = 0.04882
mult = 2.0

print("=== Timing calibration: n_qubits=4, pca projection, 5 queries/size ===")
for n in SIZES:
    with open(f"scratch_transition_corpus_{n}.pkl", "rb") as f:
        d = pickle.load(f)
    corpus, keys = d["corpus"], d["keys"]
    d_dim = next(iter(corpus.values())).shape[0]
    noise = mult * CORPUS_STD

    mem = AgentMemory(n_qubits=4, topology="chain", method="statevector", projection="pca")
    for k, v in corpus.items():
        mem.store(k, v)
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    for q_idx in range(5):
        relevant_key = keys[q_idx % len(keys)]
        query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)
        mem.evaluate_retrieval(query_vec, relevant_key=relevant_key, k_values=(1,), seed=q_idx)
    elapsed = time.perf_counter() - t0
    per_query = elapsed / 5
    print(f"  n={n:4d}  {per_query:.3f}s/query  (5 queries in {elapsed:.1f}s)")
