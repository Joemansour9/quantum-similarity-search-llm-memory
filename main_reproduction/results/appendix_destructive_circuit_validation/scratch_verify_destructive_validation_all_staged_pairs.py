"""
scratch_verify_destructive_validation_all_staged_pairs.py

Full, from-scratch, local-only (AerSimulator, no hardware/QPU) re-verification of
the destructive/Bell-basis circuit's shot-based fidelity estimator against exact
Statevector fidelity, across EVERY staged pair actually submitted to real
hardware in the original 20-job validation round (hardware_validation_2026-07-30.md).

This exists because no saved script or data file anywhere in the project backs
the paper's claim "diff=0.0001-0.0013 on the actual staged pairs used for the
hardware tests" -- the circuit-building step that would have produced this was
run inline and never saved (consistent with hardware/README.md's own
"circuit-building step run inline" notes on these jobs). The actual circuits and
their exact sim_fidelity values DO survive, pickled alongside pair identity, in
13 files -- this script re-simulates every one of them locally at 20,000 shots
(matching the paper's stated shot count) and saves the result this time.

Excluded by design: scratch_hw_circuit.pkl (Test 1's ancilla/CSWAP circuit --
verified via gate composition: cswap present, 9 qubits -- a different
construction, not part of this claim) and the *_built.pkl candidate-pool files
(scratch_hw_n5_built.pkl etc.) -- verified larger (24-27 items) than what was
actually submitted (8 final each); only the *_final.pkl subset was staged.
"""
import pickle
import numpy as np
from qiskit_aer import AerSimulator
from q_search import destructive_fidelity_from_counts

SHOTS = 20000
backend = AerSimulator()

def run_one(qc, n_qubits, shots=SHOTS):
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()
    return destructive_fidelity_from_counts(counts, n_qubits, shots)

rows = []

def add_row(source_file, test, identity, n_qubits, sim_fid, circuit):
    local_fid = run_one(circuit, n_qubits)
    diff = abs(sim_fid - local_fid)
    rows.append({
        "source_file": source_file, "test": test, "identity": identity,
        "n_qubits": n_qubits, "sim_fidelity": float(sim_fid),
        "local_20k_fidelity": float(local_fid), "abs_diff": float(diff),
        "shots": SHOTS,
    })
    print(f"[{len(rows):3d}] {test:24s} {identity:40s} n_q={n_qubits}  "
          f"sim={sim_fid:.4f}  local20k={local_fid:.4f}  |diff|={diff:.4f}")

def load(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)

# --- Single-pair files ---
d = load("scratch_hw_circuit_destructive.pkl")
add_row("scratch_hw_circuit_destructive.pkl", "Test 1",
        f"{d['relevant_key']} seed={d['seed']} q_idx={d['q_idx']} mult={d['mult']} (PCA)",
        d["n_qubits"], d["sim_fidelity"], d["circuit"])

d = load("scratch_hw_circuit_destructive_random.pkl")
add_row("scratch_hw_circuit_destructive_random.pkl", "Test 2",
        f"{d['relevant_key']} seed={d['seed']} q_idx={d['q_idx']} mult={d['mult']} (random)",
        d["n_qubits"], d["sim_fidelity"], d["circuit"])

d = load("scratch_hw_circuit_destructive_random_highfid.pkl")
add_row("scratch_hw_circuit_destructive_random_highfid.pkl", "Test 6",
        f"{d['relevant_key']} seed={d['seed']} q_idx={d['q_idx']} mult={d['mult']} (random, high-fid)",
        d["n_qubits"], d["sim_fidelity"], d["circuit"])

# --- Ranking test: 4 circuits (Test 3) ---
d = load("scratch_hw_ranking_test.pkl")
for i, (cand, circ, sim) in enumerate(zip(d["cand_keys"], d["circuits"], d["sim_fidelities"])):
    tag = "true match" if cand == d["relevant_key"] else "distractor"
    add_row("scratch_hw_ranking_test.pkl", "Test 3",
            f"query={d['relevant_key']} vs cand={cand} ({tag})",
            d["n_qubits"], sim, circ)

# --- CSWAP-strengthen: 4 destructive circuits (Priority 2) ---
d = load("scratch_hw_cswap_strengthen.pkl")
for item in d:
    add_row("scratch_hw_cswap_strengthen.pkl", "Priority 2",
            f"{item['key']} q_idx={item['q_idx']}",
            4, item["sim"], item["circuit_destr"])

# --- Multi-retrieval: 12 circuits (Test 5) ---
d = load("scratch_hw_multi_retrieval.pkl")
for (q_idx, qkey, ckey), circ, sim in zip(d["meta"], d["circuits"], d["sim_fidelities"]):
    add_row("scratch_hw_multi_retrieval.pkl", "Test 5",
            f"query={qkey} vs cand={ckey} q_idx={q_idx}",
            d["n_qubits"], sim, circ)

# --- ngroup: 6 circuits (Test 7) ---
d = load("scratch_hw_ngroup.pkl")
for (method, q_idx, key), circ, sim in zip(d["meta"], d["circuits"], d["sim_fidelities"]):
    add_row("scratch_hw_ngroup.pkl", "Test 7",
            f"{key} q_idx={q_idx} ({method})",
            d["n_qubits"], sim, circ)

# --- ngroup2: 28 circuits (Test 8) ---
d = load("scratch_hw_ngroup2.pkl")
for (method, seed, q_idx, key), circ, sim in zip(d["meta"], d["circuits"], d["sim_fidelities"]):
    add_row("scratch_hw_ngroup2.pkl", "Test 8",
            f"{key} seed={seed} q_idx={q_idx} ({method})",
            d["n_qubits"], sim, circ)

# --- pca_fill7: 7 circuits (Test 9 fill) ---
d = load("scratch_hw_pca_fill7.pkl")
for item in d:
    add_row("scratch_hw_pca_fill7.pkl", "Test 9 fill",
            f"{item['key']} seed={item['seed']} q_idx={item['q_idx']} (PCA)",
            4, item["sim"], item["circuit"])

# --- n8/n6/n5 final: 16 each (Tests 10, 11, Priority 1) ---
for fn, test in [("scratch_hw_n8_final.pkl", "Test 10 (n_qubits=8)"),
                  ("scratch_hw_n6_final.pkl", "Test 11 (n_qubits=6)"),
                  ("scratch_hw_n5_final.pkl", "Priority 1 (n_qubits=5)")]:
    d = load(fn)
    for method in ("pca", "random"):
        for item in d[method]:
            nq = 8 if "n8" in fn else (6 if "n6" in fn else 5)
            add_row(fn, test,
                    f"{item['key']} seed={item['seed']} q_idx={item['q_idx']} ({method})",
                    nq, item["sim"], item["circuit"])

# --- retrieval_extra: 32 circuits (Test 12 new) ---
d = load("scratch_hw_retrieval_extra.pkl")
for (q_idx, qkey, ckey), circ, sim in zip(d["meta"], d["circuits"], d["sim_fidelities"]):
    add_row("scratch_hw_retrieval_extra.pkl", "Test 12 (new)",
            f"query={qkey} vs cand={ckey} q_idx={q_idx}",
            d["n_qubits"], sim, circ)

print(f"\n=== DONE: {len(rows)} staged pairs re-simulated at {SHOTS} shots ===")
diffs = np.array([r["abs_diff"] for r in rows])
print(f"abs_diff: min={diffs.min():.4f}  max={diffs.max():.4f}  "
      f"mean={diffs.mean():.4f}  median={np.median(diffs):.4f}  std={diffs.std():.4f}")
print(f"Claimed range in the paper: 0.0001-0.0013")
n_within = int(((diffs >= 0.0001) & (diffs <= 0.0013)).sum())
print(f"Pairs actually falling within that claimed range: {n_within}/{len(rows)}")
n_exceed = int((diffs > 0.0013).sum())
print(f"Pairs exceeding the claimed upper bound (0.0013): {n_exceed}/{len(rows)}")

out = {
    "description": (
        "Full local-only (AerSimulator, no hardware) re-verification of the "
        "destructive/Bell-basis circuit's shot-based fidelity estimator against "
        "exact Statevector fidelity, across every one of the 144 staged pairs "
        "actually submitted to real hardware in the original 20-job validation "
        "round. Produced because no script or saved data anywhere in the "
        "project backed the paper's claim of 'diff=0.0001-0.0013 on the actual "
        "staged pairs' -- that computation was run inline and never saved. "
        "Shots=20000 matches the paper's stated convention."
    ),
    "shots": SHOTS,
    "rows": rows,
    "summary": {
        "n_pairs": len(rows),
        "abs_diff_min": float(diffs.min()),
        "abs_diff_max": float(diffs.max()),
        "abs_diff_mean": float(diffs.mean()),
        "abs_diff_median": float(np.median(diffs)),
        "abs_diff_std": float(diffs.std()),
        "n_within_claimed_range_0001_0013": n_within,
        "n_exceeding_claimed_upper_bound_0013": n_exceed,
    },
}
with open("scratch_verify_destructive_validation_all_staged_pairs_results.pkl", "wb") as f:
    pickle.dump(out, f)
print("\nSaved scratch_verify_destructive_validation_all_staged_pairs_results.pkl")
