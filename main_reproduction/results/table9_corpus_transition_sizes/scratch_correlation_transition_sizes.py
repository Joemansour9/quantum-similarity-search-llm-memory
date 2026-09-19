"""
scratch_correlation_transition_sizes.py -- recomputes the Pearson and Spearman
correlation between corpus size and the PCA-vs.-classical Recall@1 gap at noise
multiplier 1.2, reported in Section 7.1 ("PCA's departure from classical retrieval
under query noise is therefore better read as a genuine scale effect...").

This statistic was added to the paper during a later review pass (responding to a
comment that the original "corpus-scale-independent" framing contradicted the
paper's own data) without a saved script behind it -- every other reported
statistic in this project traces to a script like this one; this file closes
that gap.

Seven points, corpus size vs. gap at multiplier 1.2, out-of-sample throughout:
  48 items:  from table5_hard48_nq8_full_noise_sweep/scratch_table5_minilm_noisesweep_nq8_CLEAN.pkl
  96-400:    from this folder's scratch_table9_transition_sizes_CLEAN.pkl
gap = classical Recall@1 - PCA Recall@1, at multiplier 1.2 in every case.
"""
import pickle
from scipy import stats

with open("../table5_hard48_nq8_full_noise_sweep/scratch_table5_minilm_noisesweep_nq8_CLEAN.pkl", "rb") as f:
    table5 = pickle.load(f)["leak_free_only"][1.2]
gap_48 = table5["classical"] - table5[("pca",)]

with open("scratch_table9_transition_sizes_CLEAN.pkl", "rb") as f:
    table9 = pickle.load(f)["leak_free_only"]

sizes = [48, 96, 144, 192, 240, 320, 400]
gaps = [gap_48] + [
    table9[(size, 1.2)]["classical"] - table9[(size, 1.2)][("pca",)]
    for size in (96, 144, 192, 240, 320, 400)
]

print("size  gap")
for s, g in zip(sizes, gaps):
    print(f"{s:>5} {g:.3f}")

pearson_r, pearson_p = stats.pearsonr(sizes, gaps)
spearman_rho, spearman_p = stats.spearmanr(sizes, gaps)

print(f"\nPearson  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
print(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.6f}")

results = {
    "sizes": sizes,
    "gaps": gaps,
    "pearson_r": pearson_r,
    "pearson_p": pearson_p,
    "spearman_rho": spearman_rho,
    "spearman_p": spearman_p,
}
with open("scratch_correlation_transition_sizes_results.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved scratch_correlation_transition_sizes_results.pkl")
