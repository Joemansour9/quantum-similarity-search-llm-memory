import pickle, time
from collections import defaultdict
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from q_search import destructive_fidelity_from_counts

with open("scratch_hw_breakdown_circuits.pkl", "rb") as f:
    items = pickle.load(f)

n_qubits = 4
service = QiskitRuntimeService()
backend = service.backend("ibm_kingston")
print(f"Backend: ibm_kingston  pending_jobs={backend.status().pending_jobs}")

pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=0)
transpiled = [pm.run(it["circuit"]) for it in items]
gate_counts = [sum(v for k, v in t.count_ops().items() if k in ("ecr", "cz", "cx")) for t in transpiled]
print(f"2Q gate counts (re-confirmed at submission time): {gate_counts}")

shots = 1024
sampler = SamplerV2(mode=backend)
t0 = time.perf_counter()
job = sampler.run(transpiled, shots=shots)
print(f"Job ID: {job.job_id()}  ({len(transpiled)} PUBs, submitted, waiting...)")
result = job.result()
elapsed = time.perf_counter() - t0
print(f"wall_clock={elapsed:.1f}s")

for i, it in enumerate(items):
    creg_name = transpiled[i].cregs[0].name
    counts = result[i].data.__getattribute__(creg_name).get_counts()
    it["hw_fid"] = destructive_fidelity_from_counts(counts, n_qubits, shots)

by_query = defaultdict(list)
for it in items:
    by_query[it["q_idx"]].append(it)

print(f"\n--- RESULT: does hardware reproduce the simulation breakdown? ---")
for q_idx, rows in sorted(by_query.items()):
    true_key = rows[0]["relevant_key"]
    print(f"\nquery q_idx={q_idx} (true match = {true_key}):")
    for r in sorted(rows, key=lambda r: -r["hw_fid"]):
        marker = " <- TRUE MATCH" if r["is_true_match"] else ""
        print(f"  {r['cand_key']:12s} hw_fid={r['hw_fid']:.4f}  (sim destr val: {r['destr_sim_validation']:.4f}){marker}")
    hw_top1 = max(rows, key=lambda r: r["hw_fid"])
    sim_top1 = max(rows, key=lambda r: r["destr_sim_validation"])  # the wrong answer found in sim
    if hw_top1["is_true_match"]:
        verdict = "HARDWARE FIXED IT (picked true match, unlike simulation)"
    elif hw_top1["cand_key"] == sim_top1["cand_key"]:
        verdict = "HARDWARE REPRODUCED THE SAME WRONG ANSWER as simulation"
    else:
        verdict = f"HARDWARE FAILED DIFFERENTLY (picked {hw_top1['cand_key']}, sim picked {sim_top1['cand_key']})"
    print(f"  VERDICT: {verdict}")

with open("scratch_hw_breakdown_results.pkl", "wb") as f:
    pickle.dump(items, f)
