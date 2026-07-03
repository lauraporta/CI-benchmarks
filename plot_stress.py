"""
Plot the local contention stress test: cellpose min time vs background CPU load,
with CI reference bands overlaid. Shows that steady contention on a fast local
box sweeps it through the CI runners' range -- i.e. shared-runner contention is
sufficient to explain CI-level slowness, and variable contention explains the
variance.

Usage:
    .venv/bin/python plot_stress.py local_artifacts/stress.json --out rootcause_contention.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CI cellpose_eval mean-of-min references (from the CI artifacts, default threads)
CI_REF = {"CI macOS (~36s)": 36, "CI ubuntu (~106s)": 106,
          "CI windows (~111s)": 111}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stress")
    ap.add_argument("--out", default="rootcause_contention.png")
    args = ap.parse_args()

    d = json.loads(open(args.stress).read())
    conds = d["conditions"]
    xs = [f"{c['burners']} burners\n(demand {c['demand']}/{d['cpu_count']})"
          for c in conds]
    mins = [c["min"] for c in conds]
    maxs = [c["max"] for c in conds]
    errs = [mx - mn for mn, mx in zip(mins, maxs)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(xs, mins, yerr=[[0] * len(errs), errs], capsize=5,
                  color="#7d3dd1", zorder=3)
    idle = mins[0]
    for b, c in zip(bars, conds):
        ax.text(b.get_x() + b.get_width() / 2, c["min"],
                f"{c['min']:.0f}s\n{c['min'] / idle:.2f}x",
                ha="center", va="bottom", fontsize=10, zorder=4)

    for label, y in CI_REF.items():
        ax.axhline(y, ls="--", lw=1.2, color="#888", zorder=1)
        ax.text(len(xs) - 0.5, y, "  " + label, va="center", ha="left",
                fontsize=9, color="#555")

    ax.set_ylabel("cellpose_eval min time (s)")
    ax.set_title("Reproducing CI slowness locally by contention alone\n"
                 "cellpose pinned to 4 threads (CI width); background burners "
                 "steal cores.\nSame machine, 55s → 125s (2.3x) — sweeps through "
                 "the CI runners' range.")
    ax.set_ylim(0, max(maxs) * 1.15)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
