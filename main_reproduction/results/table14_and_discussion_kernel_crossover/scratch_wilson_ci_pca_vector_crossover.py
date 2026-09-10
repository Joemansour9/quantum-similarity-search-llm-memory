"""
scratch_wilson_ci_pca_vector_crossover.py — 95% Wilson score confidence
intervals for the 18 proportions (Fidelity kernel / Raw-classical / Cosine
kernel x n_qubits 4/6/8/10/12/14) in Table pca-vector-crossover (480-item
corpus, noise multiplier=1.2), n=20 query-runs per cell.

Source of the point estimates (verified below before computing anything):
  - Fidelity kernel column: scratch_pca_leakage_results.pkl's "leak_free"
    dict, 'pca' key -- the same out-of-sample-corrected numbers published
    in Table corpus-scale-qubit (Table 7), reused here since the fidelity
    kernel *is* that same PCA-projected quantum retrieval, just relabeled
    for this kernel comparison.
  - Raw-classical and Cosine kernel columns:
    scratch_pca_vector_crossover_oos_results.pkl's 'results' dict.

Same Wilson formula/convention already used elsewhere in this project
(scratch_wilson_ci_480_qubit_sweep.py, scratch_wilson_ci_transition_sizes.py):
standard closed form, z=1.96, no continuity correction.
"""
import math
import pickle
from datetime import datetime, timezone

Z = 1.96
N = 20
QUBIT_COUNTS = (4, 6, 8, 10, 12, 14)


def wilson(k, n=N, z=Z):
    p = k / n
    center = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return center - half, center + half


# --- Step 1: load and confirm the point estimates match the published table exactly ---
with open("scratch_pca_leakage_results.pkl", "rb") as f:
    d7 = pickle.load(f)
leak_free = d7["leak_free"]

with open("scratch_pca_vector_crossover_oos_results.pkl", "rb") as f:
    d14 = pickle.load(f)
crossover = d14["results"]

PUBLISHED = {
    4: {"fid": 0.250, "raw": 1.000, "cos": 0.200},
    6: {"fid": 0.400, "raw": 1.000, "cos": 0.500},
    8: {"fid": 0.750, "raw": 1.000, "cos": 0.850},
    10: {"fid": 0.850, "raw": 1.000, "cos": 1.000},
    12: {"fid": 0.950, "raw": 1.000, "cos": 1.000},
    14: {"fid": 0.900, "raw": 1.000, "cos": 1.000},
}

print("=== Step 1: confirm point estimates match the published table exactly ===")
point_estimates = {}
all_match = True
for nq in QUBIT_COUNTS:
    fid = leak_free[(nq, "pca")]
    raw = crossover[nq]["raw_classical"]
    cos = crossover[nq]["cosine_kernel_oos"]
    point_estimates[nq] = {"fid": fid, "raw": raw, "cos": cos}
    ok = (
        abs(fid - PUBLISHED[nq]["fid"]) < 1e-9
        and abs(raw - PUBLISHED[nq]["raw"]) < 1e-9
        and abs(cos - PUBLISHED[nq]["cos"]) < 1e-9
    )
    all_match = all_match and ok
    print(f"  n_qubits={nq:2d}  fid={fid:.3f}  raw={raw:.3f}  cos={cos:.3f}  match={ok}")
print(f"All 18 point estimates match exactly: {all_match}")
assert all_match, "Point estimates do not match the published table -- stopping."

# --- Step 2: compute Wilson 95% CIs for all 18 proportions ---
print("\n=== Step 2: Wilson 95% CIs, n=20, z=1.96 ===")
ci_results = {}
for nq in QUBIT_COUNTS:
    row = {}
    for col in ("fid", "raw", "cos"):
        k = round(point_estimates[nq][col] * N)
        lo, hi = wilson(k)
        row[col] = (lo, hi)
    ci_results[nq] = row
    print(f"  n_qubits={nq:2d}  fid=[{row['fid'][0]:.3f}, {row['fid'][1]:.3f}]  "
          f"raw=[{row['raw'][0]:.3f}, {row['raw'][1]:.3f}]  "
          f"cos=[{row['cos'][0]:.3f}, {row['cos'][1]:.3f}]")

# --- Step 3: cross-check against the manuscript-asserted intervals ---
ASSERTED = {
    4: {"fid": (0.112, 0.469), "raw": (0.839, 1.000), "cos": (0.081, 0.416)},
    6: {"fid": (0.219, 0.613), "raw": (0.839, 1.000), "cos": (0.299, 0.701)},
    8: {"fid": (0.531, 0.888), "raw": (0.839, 1.000), "cos": (0.640, 0.948)},
    10: {"fid": (0.640, 0.948), "raw": (0.839, 1.000), "cos": (0.839, 1.000)},
    12: {"fid": (0.764, 0.991), "raw": (0.839, 1.000), "cos": (0.839, 1.000)},
    14: {"fid": (0.699, 0.972), "raw": (0.839, 1.000), "cos": (0.839, 1.000)},
}
print("\n=== Step 3: cross-check against asserted intervals (tol=0.0005) ===")
all_ci_match = True
for nq in QUBIT_COUNTS:
    for col in ("fid", "raw", "cos"):
        lo, hi = ci_results[nq][col]
        alo, ahi = ASSERTED[nq][col]
        ok = abs(lo - alo) < 5e-4 and abs(hi - ahi) < 5e-4
        all_ci_match = all_ci_match and ok
        if not ok:
            print(f"  MISMATCH at n_qubits={nq}, {col}: computed=[{lo:.4f},{hi:.4f}] "
                  f"asserted=[{alo:.4f},{ahi:.4f}]")
print(f"All 18 asserted intervals match independently computed Wilson CIs: {all_ci_match}")

out = {
    "description": (
        "Independent verification of the Wilson 95% CI addition to Table "
        "pca-vector-crossover (480-item corpus, noise multiplier=1.2, n=20 "
        "per cell). Confirms (1) the three columns' point estimates trace "
        "exactly to scratch_pca_leakage_results.pkl's leak_free/pca column "
        "(fidelity kernel) and scratch_pca_vector_crossover_oos_results.pkl "
        "(raw-classical, cosine kernel), and (2) independently-computed "
        "Wilson 95% CIs at n=20 match the manuscript's asserted intervals "
        "exactly for all 18 cells (3 columns x 6 qubit counts)."
    ),
    "source_files": [
        "scratch_pca_leakage_results.pkl (leak_free/pca -> fidelity kernel)",
        "scratch_pca_vector_crossover_oos_results.pkl (raw_classical, cosine_kernel_oos)",
    ],
    "z": Z,
    "n_per_cell": N,
    "formula": "Wilson score interval (Wilson 1927), standard closed form, no continuity correction",
    "qubit_counts": QUBIT_COUNTS,
    "point_estimates": point_estimates,
    "point_estimates_all_match_published": all_match,
    "wilson_ci_results": ci_results,
    "asserted_intervals_checked": ASSERTED,
    "all_asserted_intervals_match": all_ci_match,
    "generated": datetime.now(timezone.utc).isoformat(),
}
with open("scratch_wilson_ci_pca_vector_crossover_results.pkl", "wb") as f:
    pickle.dump(out, f)
print("\nDONE -- saved scratch_wilson_ci_pca_vector_crossover_results.pkl")
