"""Figure: the five integration tests each run the full pipeline.

Data: photon-mosaic-pipeline CI run 31396896852, fast Ubuntu leg (py3.13,
20 min total), per-test wall times from the job log timestamps. The same
leg's slow sibling (py3.12, 133 min) multiplies every bar by ~6.7.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#0a7a3d"
INK, MUTED, RULE = "#111111", "#52514e", "#d8d7d2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": RULE, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

tests = [
    ("test_snakemake_dry_run", 12, GREEN, "a real dry run: 12 s"),
    ("test_snakemake_execution", 250, ORANGE, "duplicate of the CLI test below"),
    ("test_snakemake_with_contrast", 227, BLUE, ""),
    ("test_photon_mosaic_pipeline_cli_dry_run", 227, ORANGE, "named 'dry run', runs the whole pipeline"),
    ("test_photon_mosaic_pipeline_cli", 227, BLUE, ""),
    ("test_incremental_processing", 227, BLUE, ""),
]
labels = [t[0] for t in tests][::-1]
vals = [t[1] for t in tests][::-1]
cols = [t[2] for t in tests][::-1]
notes = [t[3] for t in tests][::-1]
y = list(range(len(tests)))

fig, ax = plt.subplots(figsize=(11.5, 4.3))
fig.subplots_adjust(left=0.30, top=0.80, bottom=0.16, right=0.985)
ax.barh(y, vals, height=0.55, color=cols)
for i, (v, n) in enumerate(zip(vals, notes)):
    ax.text(v + 6, i, f"{v} s", va="center", fontsize=9.5, color=INK, fontweight="bold")
    if n:
        ax.text(v + 48, i, n, va="center", fontsize=9, color=MUTED, style="italic")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5, family="DejaVu Sans Mono")
ax.set_xlabel("seconds on the FAST CI leg (the slow leg multiplies every bar by ~6.7)")
ax.set_xlim(0, 560)
fig.text(0.012, 0.955, "Five integration tests = five full pipeline runs over 11 sessions",
         fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.012, 0.885, "orange = the two runs that buy no coverage   \u2022   green = what a genuine dry run costs",
         fontsize=10, color=MUTED, ha="left", va="top")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.xaxis.grid(True, color=RULE, linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)
fig.savefig("ci-fig-redundant-runs.png", dpi=170)
print("wrote ci-fig-redundant-runs.png")
