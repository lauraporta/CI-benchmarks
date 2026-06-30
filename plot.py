"""
Render a single summary PNG from the bench-*.json artifacts.

For each workload (cellpose_eval, blas_matmul) and each (OS, python) leg, plot
the `default` vs `pin1` minimum times: a translucent bar at the best (min) time
and a scatter point per repeated run. Reading it:

- pin1 (green) sitting *below* default (red) on the same leg  -> pinning is
  causally faster (oversubscription is real).
- default points spread out while pin1 points cluster tight   -> pinning also
  kills the run-to-run variance (the #74 complaint).

Usage:  python plot.py artifacts
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze import load  # noqa: E402

COLORS = {"default": "tab:red", "pin1": "tab:green", "pin2": "tab:blue"}
OFFSETS = {"default": -0.22, "pin1": 0.0, "pin2": 0.22}


def main(root):
    rows = load(Path(root))
    if not rows:
        print("No bench-*.json found under", root)
        # still emit an (empty) figure so the artifact upload has something
        plt.figure().savefig("bench_summary.png")
        return

    tests = sorted({r["test"] for r in rows})
    settings = [s for s in ("default", "pin1", "pin2")
                if any(r["setting"] == s for r in rows)]

    fig, axes = plt.subplots(len(tests), 1, figsize=(12, 5 * len(tests)),
                             squeeze=False)
    for ax, test in zip(axes[:, 0], tests):
        legs = sorted({(r["os"], r["py"]) for r in rows if r["test"] == test})
        labelled = set()
        for i, (os_, py) in enumerate(legs):
            for setting in settings:
                vals = [r["min"] for r in rows
                        if r["test"] == test and r["os"] == os_
                        and r["py"] == py and r["setting"] == setting]
                if not vals:
                    continue
                xpos = i + OFFSETS.get(setting, 0.0)
                lbl = setting if setting not in labelled else None
                labelled.add(setting)
                ax.bar(xpos, min(vals), width=0.2,
                       color=COLORS.get(setting, "gray"), alpha=0.3)
                ax.scatter([xpos] * len(vals), vals, s=28, zorder=3,
                           color=COLORS.get(setting, "gray"), label=lbl)
        ax.set_xticks(range(len(legs)))
        ax.set_xticklabels([f"{o}\npy{p}" for o, p in legs], fontsize=8)
        ax.set_ylabel("min eval time (s)")
        ax.set_title(f"{test}  -  default (oversubscribed) vs pinned threads")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="threads")

    fig.suptitle("Thread-pinning benchmark: bar = best run, dots = each run "
                 "(spread = variance)", y=1.0)
    fig.tight_layout()
    fig.savefig("bench_summary.png", dpi=120, bbox_inches="tight")
    print("wrote bench_summary.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
