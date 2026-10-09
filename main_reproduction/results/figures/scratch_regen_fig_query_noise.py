import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Springer Nature figure rules: sans-serif (Arial), 8-9 pt lettering at final size, no in-figure title.
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

mult = [0.2, 1.2, 4.0, 6.0, 8.0, 10.0, 16.0, 20.0]
random_ = [0.920, 0.220, 0.100, 0.040, 0.000, 0.000, 0.020, 0.020]
pca =     [1.000, 0.940, 0.340, 0.200, 0.160, 0.140, 0.080, 0.060]
classical=[1.000, 1.000, 0.960, 0.860, 0.580, 0.460, 0.220, 0.160]

fig, ax = plt.subplots(figsize=(4.4, 3.3))

ax.plot(mult, classical, 'k--', lw=1.2, label='Classical')
ax.plot(mult, pca, 's-', color='C3', ms=4, lw=1.2, label='PCA (quantum)')
ax.plot(mult, random_, 'o-', color='C0', ms=4, lw=1.2, label='Random (quantum)')

ax.set_xlabel("Noise multiplier")
ax.set_ylabel("Recall@1")
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("fig_query_noise.pdf")
print("Saved fig_query_noise.pdf")
