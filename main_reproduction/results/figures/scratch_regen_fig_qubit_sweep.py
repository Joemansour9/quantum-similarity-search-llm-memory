import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

synthetic_q = [4, 6, 8, 10, 12, 14]
synthetic_random = [0.520, 0.660, 0.680, 0.780, 0.860, 0.860]
synthetic_pca =    [0.560, 0.860, 0.980, 0.960, 0.960, 0.960]

easy_q = [4, 6, 8, 10, 12, 14]
easy_random = [0.120, 0.200, 0.360, 0.320, 0.360, 0.420]
easy_pca =    [0.980, 1.000, 1.000, 1.000, 1.000, 1.000]

hard_q = [4, 6, 8, 10, 12, 14]
hard_random = [0.180, 0.120, 0.220, 0.380, 0.380, 0.420]
hard_pca =    [0.660, 0.920, 0.940, 1.000, 1.000, 1.000]

fig, ax = plt.subplots(figsize=(7.5, 6))

ax.plot(synthetic_q, synthetic_random, 'o-', color='C0', alpha=0.55, label='Synthetic, Random')
ax.plot(synthetic_q, synthetic_pca, 's-', color='C3', alpha=0.55, label='Synthetic, PCA')
ax.plot(easy_q, easy_random, 'o-', color='C0', label='Real (easy), Random')
ax.plot(easy_q, easy_pca, 's-', color='C3', label='Real (easy), PCA')
ax.plot(hard_q, hard_random, '^-', color='navy', label='Real (hard), Random')
ax.plot(hard_q, hard_pca, '^-', color='darkred', label='Real (hard), PCA')

ax.set_xlabel("Number of qubits")
ax.set_ylabel("Recall@1")
ax.set_title("Qubit-count sweep: PCA vs. random projection")
ax.set_xticks([4, 6, 8, 10, 12, 14])
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='center right', fontsize=8, framealpha=0.9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("fig_qubit_sweep.pdf")
print("Saved fig_qubit_sweep.pdf")
