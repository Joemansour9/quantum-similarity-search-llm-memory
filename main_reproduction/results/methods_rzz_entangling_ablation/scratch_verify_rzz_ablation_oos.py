"""
scratch_verify_rzz_ablation_oos.py — out-of-sample (leak-free) rerun of
the entangling-layer ablation (Methods section): entangled (Ry+RZZ) vs.
plain product-state (Ry only) encoding, PCA projection, 48-item hard
corpus, n_qubits=4 and 8, full noise sweep.

The original scratch_check_rzz_ablation.py deliberately used
query-included (leaky) PCA fitting, reasoning the ablation question
(does the entangling layer help retrieval) is orthogonal to the leakage
question. That reasoning may be fine on its own, but the paper's
Limitations section claims -- without exception -- that every reported
result uses out-of-sample fitting. This rerun closes that gap instead of
carving out an exception: identical methodology (same corpus
construction, same n_seeds=5 x n_queries=10 = 50/cell, same 8
multipliers, same two qubit counts), only the fitting step changes to
out-of-sample (candidate_batch alone, query excluded).
"""
import pickle
import numpy as np
from benchmark_realdata_embeddings_hard import make_hard_real_corpus, CORPUS_PER_DIM_STD
from q_encoder import fit_pca_projection_scale, build_feature_map
from q_search import statevector_fidelity_from_circuits, recall_at_k

N_QUBITS_LIST = (4, 8)
NOISE_MULTIPLIERS = (0.2, 1.2, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0)
N_SEEDS = 5
N_QUERIES = 10

# Published (leaky) reference, from scratch_rzz_ablation_results.pkl
PUBLISHED = {
    (4, 0.2): {"rzz": 1.000, "none": 1.000}, (4, 1.2): {"rzz": 0.880, "none": 1.000},
    (4, 2.0): {"rzz": 0.980, "none": 1.000}, (4, 4.0): {"rzz": 0.960, "none": 0.960},
    (4, 6.0): {"rzz": 0.840, "none": 0.860}, (4, 8.0): {"rzz": 0.620, "none": 0.620},
    (4, 12.0): {"rzz": 0.360, "none": 0.360}, (4, 20.0): {"rzz": 0.160, "none": 0.160},
    (8, 0.2): {"rzz": 1.000, "none": 1.000}, (8, 1.2): {"rzz": 0.920, "none": 1.000},
    (8, 2.0): {"rzz": 1.000, "none": 1.000}, (8, 4.0): {"rzz": 0.960, "none": 0.960},
    (8, 6.0): {"rzz": 0.860, "none": 0.860}, (8, 8.0): {"rzz": 0.620, "none": 0.620},
    (8, 12.0): {"rzz": 0.360, "none": 0.360}, (8, 20.0): {"rzz": 0.160, "none": 0.160},
}

print("=== Out-of-sample RZZ-entangling-layer ablation: Recall@1, rzz vs no-entangler ===")
print("(48-item hard corpus, PCA projection, n_qubits=4 and 8, n=50/cell, query excluded from fit)")
results = {}
for n_qubits in N_QUBITS_LIST:
    print(f"\n--- n_qubits={n_qubits} ---")
    print(f"{'mult':>6}  {'rzz':>8}  {'no-entangler':>13}  {'delta':>8}")
    for mult in NOISE_MULTIPLIERS:
        noise = mult * CORPUS_PER_DIM_STD
        for variant, edges_override in (("rzz", None), ("none", [])):
            recalls = []
            for seed in range(N_SEEDS):
                corpus, keys = make_hard_real_corpus(48, 8, seed=seed)
                d_dim = next(iter(corpus.values())).shape[0]
                candidate_batch = np.stack([corpus[k] for k in keys])
                rng = np.random.default_rng(seed + 1)
                for q_idx in range(N_QUERIES):
                    relevant_key = keys[q_idx % len(keys)]
                    relevant_idx = keys.index(relevant_key)
                    query_vec = corpus[relevant_key] + rng.normal(scale=noise, size=d_dim)

                    # OUT-OF-SAMPLE: fit on candidate_batch only, query excluded
                    R, lo, hi = fit_pca_projection_scale(candidate_batch, n_qubits)
                    scale = (lo, hi)
                    edges = edges_override if edges_override is not None else [(i, i + 1) for i in range(n_qubits - 1)]
                    qc_query = build_feature_map(query_vec, n_qubits, scale, edges, R, entangler="rzz")
                    fids = np.zeros(len(keys))
                    for i in range(len(keys)):
                        fid, _ = statevector_fidelity_from_circuits(
                            qc_query, candidate_batch[i], n_qubits, scale, edges, R, entangler="rzz"
                        )
                        fids[i] = fid
                    rank = np.argsort(-fids)
                    recalls.append(recall_at_k(rank, relevant_idx, 1))
            results[(n_qubits, mult, variant)] = float(np.mean(recalls))
        r_rzz = results[(n_qubits, mult, "rzz")]
        r_none = results[(n_qubits, mult, "none")]
        print(f"{mult:>6.1f}  {r_rzz:>8.3f}  {r_none:>13.3f}  {r_none-r_rzz:>+8.3f}")

print("\n=== Before / after: leaky (published) vs. out-of-sample ===")
print(f"{'n_qubits':>8}  {'mult':>6}  {'rzz leaky':>10}  {'rzz OOS':>8}  {'none leaky':>11}  {'none OOS':>9}  {'delta leaky':>12}  {'delta OOS':>10}")
for n_qubits in N_QUBITS_LIST:
    for mult in NOISE_MULTIPLIERS:
        pub = PUBLISHED[(n_qubits, mult)]
        rzz_oos = results[(n_qubits, mult, "rzz")]
        none_oos = results[(n_qubits, mult, "none")]
        delta_leaky = pub["none"] - pub["rzz"]
        delta_oos = none_oos - rzz_oos
        print(f"{n_qubits:>8}  {mult:>6.1f}  {pub['rzz']:>10.3f}  {rzz_oos:>8.3f}  {pub['none']:>11.3f}  {none_oos:>9.3f}  "
              f"{delta_leaky:>+12.3f}  {delta_oos:>+10.3f}")

print("\n=== Does the headline claim survive? ===")
d4_12 = results[(4, 1.2, "none")] - results[(4, 1.2, "rzz")]
d8_12 = results[(8, 1.2, "none")] - results[(8, 1.2, "rzz")]
print(f"At mult=1.2: n_qubits=4 delta={d4_12:+.3f}, n_qubits=8 delta={d8_12:+.3f}")
print(f"Published claim: '0.08-0.12 absolute' at both qubit counts")
in_range = all(0.08 - 1e-9 <= d <= 0.12 + 1e-9 for d in (d4_12, d8_12))
print(f"Out-of-sample deltas still fall in 0.08-0.12 range: {in_range}")

never_improves = True
other_cells_within_noise = True
for n_qubits in N_QUBITS_LIST:
    for mult in NOISE_MULTIPLIERS:
        d = results[(n_qubits, mult, "none")] - results[(n_qubits, mult, "rzz")]
        if d < -1e-9:
            never_improves = False
        if mult != 1.2 and abs(d) > 0.03:
            other_cells_within_noise = False
print(f"Entangling layer never improves Recall@1 at any tested condition: {never_improves}")
print(f"Every other cell (mult != 1.2) is identical or within ~0.03 of the other: {other_cells_within_noise}")

out = {
    "description": (
        "Out-of-sample rerun of the RZZ entangling-layer ablation "
        "(Methods section): entangled (Ry+RZZ) vs. product-state (Ry "
        "only) encoding, PCA projection, 48-item hard corpus, n_qubits=4 "
        "and 8, full noise sweep. The published version "
        "(scratch_rzz_ablation_results.pkl) deliberately used "
        "query-included (leaky) PCA fitting; this rerun uses out-of-sample "
        "fitting throughout so the Limitations section's blanket claim "
        "('every reported result... uses strict out-of-sample projection') "
        "holds without exception."
    ),
    "source_of_leaky_baseline": "scratch_rzz_ablation_results.pkl",
    "n_qubits_list": N_QUBITS_LIST,
    "noise_multipliers": NOISE_MULTIPLIERS,
    "n_seeds": N_SEEDS,
    "n_queries": N_QUERIES,
    "results": results,
}

with open("scratch_rzz_ablation_oos_results.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE -- saved scratch_rzz_ablation_oos_results.pkl")
