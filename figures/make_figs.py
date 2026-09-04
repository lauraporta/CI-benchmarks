"""Render the two diagnosis figures as PNGs for a GitHub comment.

Data are the measured values from CI-benchmarks runs 33526475990 (per-rule
timing) and 32872113181 / 32869548446 (full-suite distribution), plus the
local Apple M2 calibration. Light background on purpose: GitHub comments
render on white in light mode and do not invert images.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8983"
INK, MUTED, RULE = "#111111", "#52514e", "#d8d7d2"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=RULE, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)


# ---- Figure 1: per-rule cost per session, by machine drawn ----------------
rows = [
    ("EPYC 9V74  (run1)", 27.6, 2.3),
    ("Intel Xeon (run6)", 39.3, 2.6),
    ("EPYC 7763  (run4)", 128.5, 2.7),
    ("EPYC 7763  (run5)", 134.5, 2.6),
    ("EPYC 7763  (run3)", 135.5, 2.7),
    ("EPYC 9V74  (run2)", 162.5, 2.6),
    ("local M2, MPS", 14.0, 2.0),
]
labels = [r[0] for r in rows][::-1]
s2p = [r[1] for r in rows][::-1]
prep = [r[2] for r in rows][::-1]
y = list(range(len(rows)))

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.barh([i + 0.19 for i in y], s2p, height=0.34, color=BLUE, label="suite2p rule")
ax.barh([i - 0.19 for i in y], prep, height=0.34, color=ORANGE, label="preprocessing rule")
for i, v in zip(y, s2p):
    ax.text(v + 2.5, i + 0.19, f"{v:.1f}s", va="center", fontsize=9, color=INK, fontweight="bold")
for i, v in zip(y, prep):
    ax.text(v + 2.5, i - 0.19, f"{v:.1f}s", va="center", fontsize=8.5, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("mean seconds per session (11 sessions per run)")
ax.set_xlim(0, 190)
ax.set_title(
    "Where pipeline time goes: the suite2p rule is 92-98% of it\n"
    "preprocessing is flat everywhere; all machine-to-machine spread is inside suite2p",
    fontsize=11.5,
    fontweight="bold",
    loc="left",
    pad=12,
)
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig("ci-fig1-per-rule.png", dpi=170)

# ---- Figure 2: full-suite wall time by machine drawn ---------------------
pts = [
    ("Intel Xeon 8573C", 21.4, "ubuntu"),
    ("AMD EPYC 9V74", 24.0, "ubuntu"),
    ("Intel Xeon 8370C", 32.5, "ubuntu"),
    ("AMD EPYC 7763", 120.8, "ubuntu"),
    ("AMD EPYC 7763", 124.5, "ubuntu"),
    ("AMD EPYC 9V74", 143.8, "windows"),
    ("mixed EPYC", 142.7, "windows"),
    ("mixed EPYC", 145.1, "windows"),
    ("mixed EPYC", 146.9, "windows"),
    ("mixed EPYC", 147.8, "windows"),
    ("Apple M1 (macOS)", 63.7, "macos"),
]
colors = {"ubuntu": BLUE, "windows": ORANGE, "macos": GREY}
fig, ax = plt.subplots(figsize=(9, 3.9))
seen = set()
for i, (cpu, mins, os_) in enumerate(pts):
    lbl = os_ if os_ not in seen else None
    seen.add(os_)
    ax.scatter(mins, -i, s=90, color=colors[os_], zorder=3, label=lbl)
    ax.text(mins + 3, -i, f"{mins:.0f} min", va="center", fontsize=9, color=INK)
ax.set_yticks([-i for i in range(len(pts))])
ax.set_yticklabels([c for c, _, _ in pts], fontsize=9.5)
ax.set_xlabel("full test-suite wall time (minutes), identical code")
ax.set_xlim(0, 185)
ax.set_title(
    "The machine lottery: same suite, same commit, 21 min to 148 min\n"
    "CPU model is the strongest predictor - but note the 9V74 that took 144 min",
    fontsize=11.5,
    fontweight="bold",
    loc="left",
    pad=12,
)
# whitespace band in the middle-left of the plot; keeps clear of the labels
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.28, 0.42), fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig("ci-fig2-machine-lottery.png", dpi=170)
print("wrote ci-fig1-per-rule.png ci-fig2-machine-lottery.png")
