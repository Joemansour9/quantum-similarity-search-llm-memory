"""
scratch_verify_multiquery_wilson.py -- recomputes Table tab:hw-multiquery (Section 8.3,
"Multi-Query Retrieval Benchmark") from raw per-item data, and adds the Wilson 95% CI
that the manuscript reports alongside the 11/11 simulator and hardware counts.

This number was added to the paper during a later review pass without a saved script
behind it -- every other reported statistic in this project traces to a script like this
one; this file closes that gap.

Source files (corrected/leak-free, out-of-sample throughout):
  scratch_hw_corrected_Test5-multiretrieval_results.pkl   (3 queries, n_candidates=4 each)
  scratch_hw_corrected_Test12-batch1_results.pkl          (4 queries, n_candidates=4 each)
  scratch_hw_corrected_Test12-batch2_results.pkl          (4 queries, n_candidates=4 each)
Total: 11 queries, each compared against its true match plus three distractors.

For each query, top-1 by simulator fidelity and top-1 by hardware fidelity are computed
directly from each candidate's per-item fidelity (not read off a precomputed summary
flag), and checked against that query's true-match key.
"""
import pickle
from statsmodels.stats.proportion import proportion_confint

SOURCE_FILES = [
    "scratch_hw_corrected_Test5-multiretrieval_results.pkl",
    "scratch_hw_corrected_Test12-batch1_results.pkl",
    "scratch_hw_corrected_Test12-batch2_results.pkl",
]

queries = {}
for fname in SOURCE_FILES:
    with open(fname, "rb") as f:
        data = pickle.load(f)
    for item in data["items"]:
        key = (fname, item["seed"], item["q_idx"])
        queries.setdefault(key, []).append(item)

print(f"Total queries loaded: {len(queries)}")
assert len(queries) == 11, f"expected 11 queries, got {len(queries)}"

sim_correct = 0
hw_correct = 0
flips = 0
for key, candidates in queries.items():
    true_match_key = next(c["cand_key"] for c in candidates if c["is_true_match"])
    sim_top1 = max(candidates, key=lambda c: c["sim_statevector_fid_corrected"])["cand_key"]
    hw_top1 = max(candidates, key=lambda c: c["hw_fid_corrected"])["cand_key"]
    sim_ok = sim_top1 == true_match_key
    hw_ok = hw_top1 == true_match_key
    sim_correct += sim_ok
    hw_correct += hw_ok
    if sim_ok != hw_ok:
        flips += 1
    print(f"  {key}: true={true_match_key}  sim_top1={sim_top1} ({sim_ok})  "
          f"hw_top1={hw_top1} ({hw_ok})")

n = len(queries)
print(f"\nSimulator Recall@1: {sim_correct}/{n}")
print(f"Hardware Recall@1:   {hw_correct}/{n}")
print(f"Flips (sim correct XOR hw correct): {flips}")

sim_ci = proportion_confint(sim_correct, n, alpha=0.05, method="wilson")
hw_ci = proportion_confint(hw_correct, n, alpha=0.05, method="wilson")
print(f"\nWilson 95% CI, simulator: [{sim_ci[0]:.3f}, {sim_ci[1]:.3f}]")
print(f"Wilson 95% CI, hardware:  [{hw_ci[0]:.3f}, {hw_ci[1]:.3f}]")

results = {
    "n_queries": n,
    "sim_correct": sim_correct,
    "hw_correct": hw_correct,
    "flips": flips,
    "sim_wilson_95ci": sim_ci,
    "hw_wilson_95ci": hw_ci,
}
with open("scratch_verify_multiquery_wilson_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved scratch_verify_multiquery_wilson_results.pkl")
