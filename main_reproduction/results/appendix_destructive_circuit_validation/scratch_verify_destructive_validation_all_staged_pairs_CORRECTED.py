"""
scratch_verify_destructive_validation_all_staged_pairs_CORRECTED.py

Regenerates the 144-pair destructive-circuit validation (appendix table
tab:destructive-validation-appendix) using CORRECTED, out-of-sample PCA
fitting, replacing scratch_verify_destructive_validation_all_staged_pairs_results.pkl.

Feasibility check performed before writing this script: of the
144 staged pairs, only 101 are PCA-projected -- the query-exclusion leak is
a PCA-fitting-specific issue (random projection's matrix R is drawn from a
fixed seed independent of the data batch; only PCA's R depends on batch
composition). The other 43 pairs are random-projection pairs (Test 2, Test 6,
and the "random" halves of Test 7/8/10/11/Priority 1's mixed-method tests),
confirmed via each pair's own meta/projection field in its source pkl. This
matches the project's already-established convention (see
scratch_verify_scarcity_oos.py, and scratch_build_corrected_circuits_readiness.py's
own "excluded_jobs" note: "Test2-random -- random projection, not PCA, out of
scope").

So: the 101 PCA pairs are rebuilt here from scratch with out-of-sample PCA
fitting (candidate_batch alone, query excluded -- same convention as
scratch_build_corrected_circuits_readiness.py, which already did exactly this
for real-hardware submission and whose corrected exact-fidelity values this
script's own fresh computation is cross-checked against). The 43 random
pairs are carried over UNCHANGED from the original staged circuits (their
exact fidelity never depended on the leak in the first place) -- each row is
tagged with a "corrected" boolean so the two kinds of rows are never
ambiguous.

Every one of the 144 circuits (rebuilt-PCA and unchanged-random alike) is
then freshly run through AerSimulator at 20,000 shots, matching the
original validation's shot count.
"""
import pickle
import time
import numpy as np

from qiskit_aer import AerSimulator
from sentence_transformers import SentenceTransformer

from q_encoder import fit_pca_projection_scale, project_to_n_features, build_feature_map
from q_graph import KnowledgeGraph
from q_search import (
    statevector_fidelity_from_circuits,
    destructive_swap_test_fidelity_from_circuits,
    destructive_swap_test_circuit,
)
from benchmark_realdata_embeddings_hard import CATEGORIES, _TEXTS_BY_CAT

SHOTS = 20000
MULT_SMALL = 1.2
CORPUS_STD = 0.04882
backend = AerSimulator()

t0 = time.time()

# ============================================================
# Reconstruct the historical pool=60 48-item corpus (all-MiniLM-L6-v2),
# identical to scratch_build_corrected_circuits_readiness.py.
# ============================================================
print("Reconstructing historical pool=60 corpus (all-MiniLM-L6-v2)...")
ORIGINAL_POOL_SIZE = 60
_model = SentenceTransformer("all-MiniLM-L6-v2")
_all_texts, _cat_ranges, _idx = [], {}, 0
for _cat in CATEGORIES:
    _t60 = _TEXTS_BY_CAT[_cat][:ORIGINAL_POOL_SIZE]
    _all_texts.extend(_t60)
    _cat_ranges[_cat] = (_idx, _idx + len(_t60))
    _idx += len(_t60)
_all_emb = _model.encode(_all_texts, batch_size=32, show_progress_bar=False)
_emb_by_cat = {c: list(_all_emb[s:e]) for c, (s, e) in _cat_ranges.items()}
print(f"  done ({time.time()-t0:.0f}s)")


def make_corpus48(seed, n_items=48, n_clusters=8):
    per_cluster = n_items // n_clusters
    rng = np.random.default_rng(seed)
    items, keys = {}, []
    for ci, cat in enumerate(CATEGORIES):
        pool = _emb_by_cat[cat]
        chosen = rng.choice(len(pool), size=per_cluster, replace=False)
        for j, i in enumerate(chosen):
            key = f"item_{ci * per_cluster + j}_c{ci}"
            items[key] = np.asarray(pool[i], dtype=float)
            keys.append(key)
    return items, keys


_corpus_cache = {}


def get_corpus48(seed):
    if seed not in _corpus_cache:
        _corpus_cache[seed] = make_corpus48(seed)
    return _corpus_cache[seed]


def query_vec_for_small(seed, q_idx, corpus, keys, d_dim, mult=MULT_SMALL):
    rng = np.random.default_rng(seed + 1)
    noise = mult * CORPUS_STD
    qv = None
    for i in range(q_idx + 1):
        rk = keys[i % len(keys)]
        qv = corpus[rk] + rng.normal(scale=noise, size=d_dim)
    return qv


def build_corrected_circuit(seed, q_idx, cand_key, n_qubits):
    """Out-of-sample PCA fit (candidate_batch alone, query excluded), then
    the destructive circuit + exact statevector fidelity for one pair."""
    corpus, keys = get_corpus48(seed)
    d_dim = next(iter(corpus.values())).shape[0]
    candidate_batch = np.stack([corpus[k] for k in keys])
    query_vec = query_vec_for_small(seed, q_idx, corpus, keys, d_dim)
    edges = KnowledgeGraph.chain(n_qubits).edges()

    R, lo, hi = fit_pca_projection_scale(candidate_batch, n_qubits)  # query excluded
    scale = (lo, hi)
    qc_query = build_feature_map(query_vec, n_qubits, scale, edges, R, entangler="rzz")

    cand_vec = corpus[cand_key]
    fid_exact, _ = statevector_fidelity_from_circuits(
        qc_query, cand_vec, n_qubits, scale, edges, R, entangler="rzz")

    return qc_query, cand_vec, scale, edges, R, fid_exact


def run_20k(qc_query, cand_vec, n_qubits, scale, edges, R):
    fid_shots, _ = destructive_swap_test_fidelity_from_circuits(
        qc_query, cand_vec, n_qubits, scale, edges, R, entangler="rzz",
        shots=SHOTS, backend=backend)
    return fid_shots


def run_20k_from_circuit(full_circuit, n_qubits):
    """For unchanged (random-projection) pairs: rerun the ORIGINAL saved
    circuit object directly, no rebuild."""
    from q_search import destructive_fidelity_from_counts
    job = backend.run(full_circuit, shots=SHOTS)
    counts = job.result().get_counts()
    return destructive_fidelity_from_counts(counts, n_qubits, SHOTS)


rows = []


def add_corrected_row(source_file, test, identity, n_qubits, seed, q_idx, cand_key):
    qc_query, cand_vec, scale, edges, R, fid_exact = build_corrected_circuit(seed, q_idx, cand_key, n_qubits)
    fid_shots = run_20k(qc_query, cand_vec, n_qubits, scale, edges, R)
    diff = abs(fid_exact - fid_shots)
    rows.append({
        "source_file": source_file, "test": test, "identity": identity,
        "n_qubits": n_qubits, "sim_fidelity": float(fid_exact),
        "local_20k_fidelity": float(fid_shots), "abs_diff": float(diff),
        "shots": SHOTS, "corrected": True,
    })
    print(f"[{len(rows):3d}] {test:24s} {identity:45s} n_q={n_qubits}  "
          f"sim={fid_exact:.4f}  local20k={fid_shots:.4f}  |diff|={diff:.4f}  [CORRECTED]")


def add_unchanged_row(source_file, test, identity, n_qubits, sim_fid, circuit):
    fid_shots = run_20k_from_circuit(circuit, n_qubits)
    diff = abs(sim_fid - fid_shots)
    rows.append({
        "source_file": source_file, "test": test, "identity": identity,
        "n_qubits": n_qubits, "sim_fidelity": float(sim_fid),
        "local_20k_fidelity": float(fid_shots), "abs_diff": float(diff),
        "shots": SHOTS, "corrected": False,
    })
    print(f"[{len(rows):3d}] {test:24s} {identity:45s} n_q={n_qubits}  "
          f"sim={sim_fid:.4f}  local20k={fid_shots:.4f}  |diff|={diff:.4f}  [unchanged, random-proj]")


def load(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


# ---- Test 1 (n=1, PCA) ----
d = load("scratch_hw_circuit_destructive.pkl")
assert d["relevant_key"] == "item_0_c0" and d["seed"] == 0 and d["q_idx"] == 0
add_corrected_row("scratch_hw_circuit_destructive.pkl", "Test 1",
                   f"{d['relevant_key']} seed={d['seed']} q_idx={d['q_idx']} mult={d['mult']} (PCA)",
                   d["n_qubits"], d["seed"], d["q_idx"], d["relevant_key"])

# ---- Test 2 (n=1, random -- unchanged) ----
d = load("scratch_hw_circuit_destructive_random.pkl")
add_unchanged_row("scratch_hw_circuit_destructive_random.pkl", "Test 2",
                   f"{d['relevant_key']} seed={d['seed']} q_idx={d['q_idx']} mult={d['mult']} (random)",
                   d["n_qubits"], d["sim_fidelity"], d["circuit"])

# ---- Test 6 (n=1, random -- unchanged) ----
d = load("scratch_hw_circuit_destructive_random_highfid.pkl")
add_unchanged_row("scratch_hw_circuit_destructive_random_highfid.pkl", "Test 6",
                   f"{d['relevant_key']} seed={d['seed']} q_idx={d['q_idx']} mult={d['mult']} (random, high-fid)",
                   d["n_qubits"], d["sim_fidelity"], d["circuit"])

# ---- Test 3 ranking (n=4, PCA) ----
d = load("scratch_hw_ranking_test.pkl")
assert d["relevant_key"] == "item_0_c0"
for cand in d["cand_keys"]:
    tag = "true match" if cand == d["relevant_key"] else "distractor"
    add_corrected_row("scratch_hw_ranking_test.pkl", "Test 3",
                       f"query={d['relevant_key']} vs cand={cand} ({tag})",
                       d["n_qubits"], 0, 0, cand)

# ---- Priority 2 / Test 14 cswap_strengthen (n=4, PCA, self-match) ----
d = load("scratch_hw_cswap_strengthen.pkl")
for item in d:
    add_corrected_row("scratch_hw_cswap_strengthen.pkl", "Priority 2",
                       f"{item['key']} q_idx={item['q_idx']}",
                       4, 0, item["q_idx"], item["key"])

# ---- Test 5 multi_retrieval (n=12, PCA) ----
d = load("scratch_hw_multi_retrieval.pkl")
for q_idx, qkey, ckey in d["meta"]:
    add_corrected_row("scratch_hw_multi_retrieval.pkl", "Test 5",
                       f"query={qkey} vs cand={ckey} q_idx={q_idx}",
                       d["n_qubits"], 0, q_idx, ckey)

# ---- Test 7 ngroup (n=6, MIXED: 3 pca corrected + 3 random unchanged) ----
d = load("scratch_hw_ngroup.pkl")
for (method, q_idx, key), circ, sim in zip(d["meta"], d["circuits"], d["sim_fidelities"]):
    identity = f"{key} q_idx={q_idx} ({method})"
    if method == "pca":
        add_corrected_row("scratch_hw_ngroup.pkl", "Test 7", identity, d["n_qubits"], 0, q_idx, key)
    else:
        add_unchanged_row("scratch_hw_ngroup.pkl", "Test 7", identity, d["n_qubits"], sim, circ)

# ---- Test 8 ngroup2 (n=28, MIXED: 14 pca corrected + 14 random unchanged) ----
d = load("scratch_hw_ngroup2.pkl")
for (method, seed, q_idx, key), circ, sim in zip(d["meta"], d["circuits"], d["sim_fidelities"]):
    identity = f"{key} seed={seed} q_idx={q_idx} ({method})"
    if method == "pca":
        add_corrected_row("scratch_hw_ngroup2.pkl", "Test 8", identity, d["n_qubits"], seed, q_idx, key)
    else:
        add_unchanged_row("scratch_hw_ngroup2.pkl", "Test 8", identity, d["n_qubits"], sim, circ)

# ---- Test 9 fill7 (n=7, PCA) ----
d = load("scratch_hw_pca_fill7.pkl")
for item in d:
    add_corrected_row("scratch_hw_pca_fill7.pkl", "Test 9 fill",
                       f"{item['key']} seed={item['seed']} q_idx={item['q_idx']} (PCA)",
                       4, item["seed"], item["q_idx"], item["key"])

# ---- Test 10/11/Priority 1: n8/n6/n5 final (n=16 each, MIXED: 8 pca + 8 random) ----
for fn, test, nq in [("scratch_hw_n8_final.pkl", "Test 10 (n_qubits=8)", 8),
                      ("scratch_hw_n6_final.pkl", "Test 11 (n_qubits=6)", 6),
                      ("scratch_hw_n5_final.pkl", "Priority 1 (n_qubits=5)", 5)]:
    d = load(fn)
    for method in ("pca", "random"):
        for item in d[method]:
            identity = f"{item['key']} seed={item['seed']} q_idx={item['q_idx']} ({method})"
            if method == "pca":
                add_corrected_row(fn, test, identity, nq, item["seed"], item["q_idx"], item["key"])
            else:
                add_unchanged_row(fn, test, identity, nq, item["sim"], item["circuit"])

# ---- Test 12 new: retrieval_extra (n=32, PCA) ----
d = load("scratch_hw_retrieval_extra.pkl")
for q_idx, qkey, ckey in d["meta"]:
    add_corrected_row("scratch_hw_retrieval_extra.pkl", "Test 12 (new)",
                       f"query={qkey} vs cand={ckey} q_idx={q_idx}",
                       d["n_qubits"], 0, q_idx, ckey)

print(f"\n=== DONE: {len(rows)} staged pairs re-simulated at {SHOTS} shots "
      f"({sum(r['corrected'] for r in rows)} corrected-PCA, "
      f"{sum(not r['corrected'] for r in rows)} unchanged-random) ===")

diffs = np.array([r["abs_diff"] for r in rows])
print(f"abs_diff: min={diffs.min():.4f}  max={diffs.max():.4f}  "
      f"mean={diffs.mean():.4f}  median={np.median(diffs):.4f}  std={diffs.std():.4f}")

# Confirm the Table hw-gatecount pair (Test 1, item_0_c0) now reads 0.9552 here too.
test1_row = next(r for r in rows if r["test"] == "Test 1")
print(f"\nTest 1 / item_0_c0 corrected exact fidelity: {test1_row['sim_fidelity']:.4f} "
      f"(Table hw-gatecount states 0.9552)")

out = {
    "description": (
        "CORRECTED (out-of-sample PCA fitting) re-verification of the destructive/"
        "Bell-basis circuit's shot-based fidelity estimator against exact Statevector "
        "fidelity, across all 144 staged pairs from the original 20-job hardware "
        "validation round. Replaces "
        "scratch_verify_destructive_validation_all_staged_pairs_results.pkl, which "
        "used the original (query-included / leaky) PCA circuits. Of the 144 pairs, "
        "101 are PCA-projected and are rebuilt here from scratch with the query "
        "excluded from the batch used to fit the projection (same convention as "
        "scratch_build_corrected_circuits_readiness.py); the remaining 43 pairs are "
        "random-projection pairs (Test 2, Test 6, and the random halves of Tests "
        "7/8/10/11/Priority 1), for which the query-inclusion leak does not apply "
        "(random projection's matrix is seed-drawn, independent of the data batch) "
        "-- these are carried over unchanged from the original staged circuits, "
        "each tagged corrected=False. Every one of the 144 circuits, rebuilt or "
        "unchanged, was freshly run through AerSimulator at 20,000 shots."
    ),
    "shots": SHOTS,
    "n_corrected_pca": int(sum(r["corrected"] for r in rows)),
    "n_unchanged_random": int(sum(not r["corrected"] for r in rows)),
    "rows": rows,
    "summary": {
        "n_pairs": len(rows),
        "abs_diff_min": float(diffs.min()),
        "abs_diff_max": float(diffs.max()),
        "abs_diff_mean": float(diffs.mean()),
        "abs_diff_median": float(np.median(diffs)),
        "abs_diff_std": float(diffs.std()),
    },
}
with open("scratch_verify_destructive_validation_all_staged_pairs_results.pkl", "wb") as f:
    pickle.dump(out, f)
print("\nSaved scratch_verify_destructive_validation_all_staged_pairs_results.pkl (REPLACED)")
