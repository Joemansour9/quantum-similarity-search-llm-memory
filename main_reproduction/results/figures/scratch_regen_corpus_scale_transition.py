import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sizes = [96, 144, 192, 240, 320, 400]
gap_12 = [0.400, 0.400, 0.550, 0.700, 0.750, 0.850]
gap_20 = [0.700, 0.750, 0.800, 0.850, 0.900, 0.850]

fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.plot(sizes, gap_12, 'o-', color='C0', label='Noise mult. = 1.2')
ax.plot(sizes, gap_20, 'o-', color='C3', label='Noise mult. = 2.0')

# 48-item baseline: Table noise-hard's mult.=1.2 gap (0.940 vs. 1.000), unconnected to the curves
ax.plot(48, 0.06, '*', color='green', markersize=16, zorder=5)
ax.annotate("48-item baseline\n(mult.=1.2, Table\nnoise-hard: 0.940\nvs. 1.000)", xy=(48, 0.06),
            xytext=(60, 0.20), fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray'))

# 480-item point: different multipliers (4.0, 6.0), also unconnected to the mult=1.2/2.0 curves
ax.plot(480, 0.95, 'D', color='C0', markersize=8, zorder=5)
ax.plot(480, 0.65, 'D', color='C3', markersize=8, zorder=5)
ax.annotate("480-item, n=100/cell\n(mult.=4.0: gap 0.95;\nmult.=6.0: gap 0.65 —\ndifferent multipliers,\nnot connected to the\nmult.=1.2/2.0 curves)",
            xy=(480, 0.80), xytext=(340, 0.55), fontsize=7.5, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlabel("Candidate-pool size (documents)")
ax.set_ylabel("PCA − classical Recall@1 gap")
ax.set_title("PCA-vs.-classical Recall@1 gap vs. candidate-pool size")
ax.set_ylim(-0.02, 1.05)
ax.set_xlim(0, 520)
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("corpus_scale_transition.pdf")
print("Saved corpus_scale_transition.pdf")
