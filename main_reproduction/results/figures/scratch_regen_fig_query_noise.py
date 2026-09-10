import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mult = [0.2, 1.2, 4.0, 6.0, 8.0, 10.0, 16.0, 20.0]
random_ = [0.920, 0.220, 0.100, 0.040, 0.000, 0.000, 0.020, 0.020]
pca =     [1.000, 0.940, 0.340, 0.200, 0.160, 0.140, 0.080, 0.060]
classical=[1.000, 1.000, 0.960, 0.860, 0.580, 0.460, 0.220, 0.160]

fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.plot(mult, classical, 'k--', label='Classical')
ax.plot(mult, pca, 's-', color='C3', label='PCA (quantum)')
ax.plot(mult, random_, 'o-', color='C0', label='Random (quantum)')

ax.set_xlabel("Noise multiplier")
ax.set_ylabel("Recall@1")
ax.set_title("Query-noise sweep: PCA tracks classical at low noise, diverges from\nmultiplier 4.0 onward; random collapses\n(hard corpus, n-qubits=8)")
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("fig_query_noise.pdf")
print("Saved fig_query_noise.pdf")
