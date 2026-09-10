"""
scratch_wilson_ci_transition_sizes.py — PCA and random-projection Recall@1
proportions (individually, not just their gap) behind the corpus-size
transition table (Table corpus-transition: 96/144/192/240/320/400 items,
mult=1.2/2.0, n_qubits=4, n=20/cell, out-of-sample fitting), plus a 95%
Wilson score interval for each.

IMPORTANT: the paper's published "Gap" column for this table is
(classical - PCA), not (PCA - random) -- confirmed below by recomputing
both from the same source file and checking which one reproduces the
published 0.400/0.700/.../0.850 values. Classical recall is exactly 1.000
in all 12 cells here, so PCA was already recoverable as 1 - Gap; random
projection's numbers were never part of the published gap and are
reported here for the first time.

Source: scratch_pca_leakage_transition_sizes_results.pkl (the leak-free/
out-of-sample rerun that actually produced this table), key
[(n_items, mult)][('pca'|'random', 'leak_free')].
"""
import math
import pickle
from datetime import datetime, timezone

Z = 1.96
N = 20
SIZES = (96, 144, 192, 240, 320, 400)
MULTS = (1.2, 2.0)

# Published Table corpus-transition gap values, for the confirmation check
PUBLISHED_GAP = {
    96: {1.2: 0.400, 2.0: 0.700},
    144: {1.2: 0.400, 2.0: 0.750},
    192: {1.2: 0.550, 2.0: 0.800},
    240: {1.2: 0.700, 2.0: 0.850},
    320: {1.2: 0.750, 2.0: 0.900},
    400: {1.2: 0.850, 2.0: 0.850},
}

with open("scratch_pca_leakage_transition_sizes_results.pkl", "rb") as f:
    d = pickle.load(f)

print("=== Step 1: confirm what the published 'Gap' column actually measures ===")
gap_is_cls_minus_pca = True
gap_is_pca_minus_rand = True
for n_items in SIZES:
    for mult in MULTS:
        row = d[(n_items, mult)]
        pca = row[("pca", "leak_free")]
        rand = row[("random", "leak_free")]
        cls = row["classical"]
        published = PUBLISHED_GAP[n_items][mult]
        g1 = cls - pca
        g2 = pca - rand
        ok1 = abs(g1 - published) < 1e-9
        ok2 = abs(g2 - published) < 1e-9
        gap_is_cls_minus_pca = gap_is_cls_minus_pca and ok1
        gap_is_pca_minus_rand = gap_is_pca_minus_rand and ok2
        print(f"  n={n_items:4d} mult={mult:>4.1f}  pca={pca:.3f}  random={rand:.3f}  classical={cls:.3f}  "
              f"published_gap={published:.3f}  (classical-pca)={g1:.3f} match={ok1}  "
              f"(pca-random)={g2:.3f} match={ok2}")

print(f"\nGap column = (classical - PCA) for all 12 cells: {gap_is_cls_minus_pca}")
print(f"Gap column = (PCA - random) for all 12 cells:     {gap_is_pca_minus_rand}")
if not gap_is_cls_minus_pca:
    raise SystemExit("Expected (classical - PCA) to reproduce the published gap -- it didn't. Stopping.")


def wilson_interval(k: int, n: int, z: float = Z) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo, hi = (center - half) / denom, (center + half) / denom
    return (max(0.0, lo), min(1.0, hi))


print("\n=== Step 2: PCA and random Recall@1, with 95% Wilson CI (n=20) ===")
results = {}
for n_items in SIZES:
    for mult in MULTS:
        row = d[(n_items, mult)]
        cell = {}
        for proj in ("pca", "random"):
            p = row[(proj, "leak_free")]
            k = round(p * N)
            assert abs(k / N - p) < 1e-9, f"{proj} proportion {p} at n={n_items},mult={mult} is not an exact k/{N}"
            lo, hi = wilson_interval(k, N)
            cell[proj] = {"k": k, "n": N, "p": p, "ci_lower": lo, "ci_upper": hi, "ci_width": hi - lo}
        cell["classical"] = row["classical"]
        results[(n_items, mult)] = cell
        print(f"  n={n_items:4d} mult={mult:>4.1f}  "
              f"PCA {cell['pca']['k']:2d}/{N}={cell['pca']['p']:.3f} CI=[{cell['pca']['ci_lower']:.3f}, {cell['pca']['ci_upper']:.3f}]   "
              f"Random {cell['random']['k']:2d}/{N}={cell['random']['p']:.3f} CI=[{cell['random']['ci_lower']:.3f}, {cell['random']['ci_upper']:.3f}]")

out = {
    "description": (
        "PCA and random-projection Recall@1 proportions (individually) behind "
        "the corpus-size transition table (Table corpus-transition: 96-400 "
        "items, mult=1.2/2.0, n_qubits=4, n=20/cell, out-of-sample fitting), "
        "with 95% Wilson score CIs for each. The paper's published 'Gap' "
        "column for this table is (classical - PCA), not (PCA - random); "
        "classical recall is exactly 1.000 in all 12 cells, confirmed above "
        "against the published gap values before computing anything."
    ),
    "source_file": "scratch_pca_leakage_transition_sizes_results.pkl",
    "source_key": "('pca'|'random', 'leak_free') per (n_items, mult)",
    "gap_definition_confirmed": "classical - pca",
    "z": Z,
    "n_per_cell": N,
    "formula": "Wilson score interval (Wilson 1927), standard closed form, no continuity correction",
    "generated": datetime.now(timezone.utc).isoformat(),
    "results": results,
}

with open("scratch_wilson_ci_transition_sizes_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE -- saved scratch_wilson_ci_transition_sizes_results.pkl")
