import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Springer Nature figure rules: sans-serif (Arial), 8-9 pt lettering at final size, no in-figure title.
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

synthetic_q = [4, 6, 8, 10, 12, 14]
synthetic_random = [0.520, 0.660, 0.680, 0.780, 0.860, 0.860]
synthetic_pca =    [0.560, 0.860, 0.980, 0.960, 0.960, 0.960]

easy_q = [4, 6, 8, 10, 12, 14]
easy_random = [0.120, 0.200, 0.360, 0.320, 0.360, 0.420]
easy_pca =    [0.980, 1.000, 1.000, 1.000, 1.000, 1.000]

hard_q = [4, 6, 8, 10, 12, 14]
hard_random = [0.180, 0.120, 0.220, 0.380, 0.380, 0.420]
hard_pca =    [0.660, 0.920, 0.940, 1.000, 1.000, 1.000]

fig, ax = plt.subplots(figsize=(4.4, 4.0))

ax.plot(synthetic_q, synthetic_random, 'o:', color='C0', alpha=0.55, ms=4, lw=1.2, label='Synthetic, Random')
ax.plot(synthetic_q, synthetic_pca, 's:', color='C3', alpha=0.55, ms=4, lw=1.2, label='Synthetic, PCA')
ax.plot(easy_q, easy_random, 'o-', color='C0', ms=4, lw=1.2, label='Real (easy), Random')
ax.plot(easy_q, easy_pca, 's-', color='C3', ms=4, lw=1.2, label='Real (easy), PCA')
ax.plot(hard_q, hard_random, 'o--', color='navy', ms=4, lw=1.2, label='Real (hard), Random')
ax.plot(hard_q, hard_pca, 's--', color='darkred', ms=4, lw=1.2, label='Real (hard), PCA')

ax.set_xlabel("Number of qubits")
ax.set_ylabel("Recall@1")
ax.set_xticks([4, 6, 8, 10, 12, 14])
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, columnspacing=1.5, handlelength=1.8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("fig_qubit_sweep.pdf", bbox_inches="tight")
print("Saved fig_qubit_sweep.pdf")
