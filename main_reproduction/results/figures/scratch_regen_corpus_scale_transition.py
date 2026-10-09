import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Springer Nature figure rules: sans-serif (Arial), 8-9 pt lettering at final size, no in-figure title.
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42,
})

sizes = [96, 144, 192, 240, 320, 400]
gap_12 = [0.400, 0.400, 0.550, 0.700, 0.750, 0.850]
gap_20 = [0.700, 0.750, 0.800, 0.850, 0.900, 0.850]

fig, ax = plt.subplots(figsize=(4.4, 3.4))

ax.plot(sizes, gap_12, 'o-', color='C0', ms=4, lw=1.2, label='Noise mult. = 1.2')
ax.plot(sizes, gap_20, 's--', color='C3', ms=4, lw=1.2, label='Noise mult. = 2.0')

# 48-item baseline: Table noise-hard's mult.=1.2 gap (0.940 vs. 1.000), unconnected to the curves
ax.plot(48, 0.06, '*', color='green', markersize=11, zorder=5)
ax.annotate("48 items", xy=(48, 0.06), xytext=(70, 0.17), fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# 480-item points: different multipliers (4.0, 6.0), unconnected to the mult.=1.2/2.0 curves
ax.plot(480, 0.95, 'D', color='C0', markersize=5.5, zorder=5)
ax.plot(480, 0.65, 'D', color='C3', markersize=5.5, zorder=5)
ax.text(468, 0.95, 'mult. 4.0', fontsize=8, ha='right', va='center')
ax.text(468, 0.65, 'mult. 6.0', fontsize=8, ha='right', va='center')
ax.plot([480, 480], [0.65, 0.95], ':', color='gray', lw=0.8, zorder=1)

ax.set_xlabel("Candidate-pool size (documents)")
ax.set_ylabel("PCA - classical Recall@1 gap")
ax.set_ylim(-0.02, 1.05)
ax.set_xlim(0, 530)
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("corpus_scale_transition.pdf")
print("Saved corpus_scale_transition.pdf")
