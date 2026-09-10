import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
pca_dev =    [0.270, 0.326]
random_dev = [0.518, 0.602]
ratios = [random_dev[i] / pca_dev[i] for i in range(2)]

x = np.arange(2)
width = 0.35

fig, ax = plt.subplots(figsize=(6, 6))
ax.bar(x - width/2, pca_dev, width, color='C3', label='PCA')
ax.bar(x + width/2, random_dev, width, color='C0', label='Random')

ax.set_yscale('log')
ax.set_ylim(0.15, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Mean |quantum - classical| deviation (log scale)")
ax.set_title("Query-noise dissociation replicates across embedding models\n(n-qubits=8, matched)")
ax.legend(loc='upper left')

for i, r in enumerate(ratios):
    ymax = max(pca_dev[i], random_dev[i])
    ax.text(x[i], ymax * 1.2, f"{r:.2f}x", ha='center', fontweight='bold')

fig.tight_layout()
fig.savefig("fig_embedding_replication.pdf")
print("Saved fig_embedding_replication.pdf")
print("ratios:", ratios)
