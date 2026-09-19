"""
scratch_hw_gatecount_ttest.py -- recomputes the paired n=4 CSWAP-vs-destructive
degradation-gap comparison reported in Section 8.2 ("Fidelity Degradation Tracks
Gate Count") and the Hardware Validation Summary: mean degradation gap, sample
std, and paired t-test across the four query/candidate pairs (the original
single pair plus three additional pairs).

This statistic already existed in the manuscript before this review round, but
no script computing it survived anywhere in the project -- every other reported
statistic in this project traces to a script like this one; this file closes
that gap.

Source: scratch_hw_corrected_Test14-cswapvsdestr_results.pkl, the corrected
(leak-free, resubmitted) hardware job "Priority 2 -- CSWAP-vs-destructive, n=4
paired" (job dacoir3dd5gc73d6bts0, corrected rerun of d9lcqtjjf64c739jbkrg).
Degradation gap = exact simulator fidelity - hardware fidelity, per pair and
per circuit construction.
"""
import pickle
import numpy as np
from scipy import stats

with open("scratch_hw_corrected_Test14-cswapvsdestr_results.pkl", "rb") as f:
    data = pickle.load(f)

pairs = {}
for item in data["items"]:
    pairs.setdefault(item["q_idx"], {})[item["circuit_type"]] = item

destr_gaps, cswap_gaps = [], []
print(f"{'q_idx':>6} {'sim_fid':>10} {'destr_hw':>10} {'destr_gap':>10} {'cswap_hw':>10} {'cswap_gap':>10}")
for q_idx in sorted(pairs):
    p = pairs[q_idx]
    sim = p["destructive"]["sim_statevector_fid_corrected"]
    assert abs(sim - p["ancilla"]["sim_statevector_fid_corrected"]) < 1e-12
    d_gap = sim - p["destructive"]["hw_fid_corrected"]
    c_gap = sim - p["ancilla"]["hw_fid_corrected"]
    destr_gaps.append(d_gap)
    cswap_gaps.append(c_gap)
    print(f"{q_idx:>6} {sim:>10.6f} {p['destructive']['hw_fid_corrected']:>10.6f} "
          f"{d_gap:>10.6f} {p['ancilla']['hw_fid_corrected']:>10.6f} {c_gap:>10.6f}")

destr_gaps, cswap_gaps = np.array(destr_gaps), np.array(cswap_gaps)
diffs = cswap_gaps - destr_gaps
t_stat, p_val = stats.ttest_rel(cswap_gaps, destr_gaps)

print(f"\nDestructive: mean={destr_gaps.mean():.4f}  std(ddof=1)={destr_gaps.std(ddof=1):.4f}")
print(f"CSWAP:       mean={cswap_gaps.mean():.4f}  std(ddof=1)={cswap_gaps.std(ddof=1):.4f}")
print(f"Paired differences (CSWAP - destructive): {diffs}")
print(f"min/max paired diff: {diffs.min():.4f} / {diffs.max():.4f}")
print(f"Paired t-test: t={t_stat:.4f}  p={p_val:.4f}  df={len(diffs)-1}")

results = {
    "destr_gaps": destr_gaps.tolist(),
    "cswap_gaps": cswap_gaps.tolist(),
    "destr_mean": destr_gaps.mean(), "destr_std": destr_gaps.std(ddof=1),
    "cswap_mean": cswap_gaps.mean(), "cswap_std": cswap_gaps.std(ddof=1),
    "paired_diffs": diffs.tolist(),
    "t_stat": t_stat, "p_value": p_val, "df": len(diffs) - 1,
}
with open("scratch_hw_gatecount_ttest_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved scratch_hw_gatecount_ttest_results.pkl")
