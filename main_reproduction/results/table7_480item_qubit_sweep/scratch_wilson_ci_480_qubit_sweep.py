"""
scratch_wilson_ci_480_qubit_sweep.py — 95% Wilson score confidence
intervals for the 12 proportions (Random/PCA x n_qubits 4/6/8/10/12/14)
in the 480-item corpus qubit-count sweep (Table corpus-scale-qubit), n=20
query-runs per cell.

Source of the 12 proportions: scratch_pca_leakage_results.pkl's
"leak_free" dict (the out-of-sample-corrected numbers actually published
in the paper's Table corpus-scale-qubit) -- NOT
scratch_large_corpus_qubit_sweep.pkl, which is the original LEAKY archive
and differs from the published table at n_qubits=6/8/10 (PCA 0.55/0.80/0.90
vs. the published/leak-free 0.40/0.75/0.85). Confirmed by direct
comparison before computing anything below -- see the printed check.

Wilson score interval, standard closed form (same convention already used
elsewhere in the paper's own Wilson-interval citations, e.g. the noise
multiplier=1.2 random/PCA/classical intervals in Section 5.2):
    center = (p + z^2/(2n)) / (1 + z^2/n)
    half   = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)
    CI = (center - half, center + half)
z = 1.96 for 95%. n = 20 for every cell here (all six qubit counts).
"""
import math
import pickle
from datetime import datetime, timezone

Z = 1.96
N = 20

# --- Published proportions from the paper, for the exact-match check ---
PUBLISHED = {
    "random": {4: 0.050, 6: 0.000, 8: 0.100, 10: 0.150, 12: 0.200, 14: 0.350},
    "pca":    {4: 0.250, 6: 0.400, 8: 0.750, 10: 0.850, 12: 0.950, 14: 0.900},
}

with open("scratch_pca_leakage_results.pkl", "rb") as f:
    d = pickle.load(f)
leak_free = d["leak_free"]

print("=== Step 1: confirm the 12 proportions match exactly ===")
all_match = True
for proj in ("random", "pca"):
    for nq in (4, 6, 8, 10, 12, 14):
        actual = leak_free[(nq, proj)]
        expected = PUBLISHED[proj][nq]
        ok = abs(actual - expected) < 1e-12
        all_match = all_match and ok
        print(f"  n_qubits={nq:2d}  {proj:6s}  file={actual:.3f}  cited={expected:.3f}  match={ok}")
print(f"\nALL 12 PROPORTIONS MATCH EXACTLY: {all_match}")
if not all_match:
    raise SystemExit("Mismatch found -- stopping rather than computing CIs against the wrong numbers.")

# Sanity note on the OTHER file with a similar name, which does NOT match:
with open("scratch_large_corpus_qubit_sweep.pkl", "rb") as f:
    leaky_archived = pickle.load(f)
print("\n(For the record: scratch_large_corpus_qubit_sweep.pkl -- the original LEAKY archive, "
      "not used here -- differs from the above at n_qubits=6/8/10 PCA: "
      f"{leaky_archived[(6,'pca')]}/{leaky_archived[(8,'pca')]}/{leaky_archived[(10,'pca')]} "
      "vs. 0.400/0.750/0.850 here.)")


def wilson_interval(k: int, n: int, z: float = Z) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo, hi = (center - half) / denom, (center + half) / denom
    # Clip floating-point noise at the boundary (e.g. k=0 gives lo ~ -1e-17,
    # not a real negative probability) -- a proportion's CI can't leave [0, 1].
    return (max(0.0, lo), min(1.0, hi))


print("\n=== Step 2: Wilson 95% CI for each of the 12 proportions (n=20) ===")
results = {}
for proj in ("random", "pca"):
    for nq in (4, 6, 8, 10, 12, 14):
        p = leak_free[(nq, proj)]
        k = round(p * N)
        assert abs(k / N - p) < 1e-9, f"proportion {p} is not an exact k/{N} fraction"
        lo, hi = wilson_interval(k, N)
        results[(nq, proj)] = {
            "k": k, "n": N, "p": p,
            "ci_lower": lo, "ci_upper": hi, "ci_width": hi - lo,
        }
        print(f"  n_qubits={nq:2d}  {proj:6s}  {k:2d}/{N}={p:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

out = {
    "description": (
        "95% Wilson score confidence intervals for the 480-item corpus "
        "qubit-count sweep (Table corpus-scale-qubit), n=20 query-runs per cell. "
        "Proportions sourced from scratch_pca_leakage_results.pkl's 'leak_free' "
        "dict, matching the published table exactly."
    ),
    "source_file": "scratch_pca_leakage_results.pkl",
    "source_key": "leak_free",
    "z": Z,
    "n_per_cell": N,
    "formula": "Wilson score interval (Wilson 1927), standard closed form, no continuity correction",
    "generated": datetime.now(timezone.utc).isoformat(),
    "results": results,
}

with open("scratch_wilson_ci_480_qubit_sweep_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE -- saved scratch_wilson_ci_480_qubit_sweep_results.pkl")
