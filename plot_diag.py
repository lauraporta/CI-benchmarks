"""
Render the diagnostics figure from the diag-*.json artifacts:
  (top)    torch matmul / conv2d / attention per OS  -> localizes the ubuntu
           slowdown to the attention kernel (matmul/conv are equal across OS).
  (bottom) suite2p registration vs full per OS, dots = each run -> registration
           is tight everywhere; the full step's spread is the runner-luck variance.

Usage:  python plot_diag.py artifacts
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FN = re.compile(r"diag-(?P<os>.+?)-py(?P<py>[\d.]+)-run(?P<run>\d+)\.json")
OSES = ["macos-latest", "ubuntu-latest", "windows-latest"]
OSCOL = {"macos-latest": "tab:green", "ubuntu-latest": "tab:red",
         "windows-latest": "tab:blue"}


def load(root):
    import json
    rows = []
    for p in Path(root).rglob("diag-*.json"):
        m = FN.search(p.name)
        if not m:
            continue
        for b in json.loads(p.read_text())["benchmarks"]:
            rows.append({"os": m["os"], "py": m["py"],
                         "test": b["name"].split("[")[0],
                         "min": b["stats"]["min"]})
    return rows


def _grouped(ax, rows, tests, labels, logy):
    width = 0.26
    for i, os_ in enumerate(OSES):
        xs, ys = [], []
        for j, t in enumerate(tests):
            vals = [r["min"] for r in rows if r["test"] == t and r["os"] == os_]
            if vals:
                xs.append(j + (i - 1) * width)
                ys.append(min(vals))
                ax.scatter([j + (i - 1) * width] * len(vals), vals, s=14,
                           color=OSCOL[os_], zorder=3, edgecolor="k", linewidth=0.3)
        ax.bar(xs, ys, width=width, color=OSCOL[os_], alpha=0.35,
               label=os_.split("-")[0])
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(labels)
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()


def main(root):
    rows = load(root)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    if not rows:
        for ax in axes:
            ax.text(0.5, 0.5, "NO DATA\n(no diag-*.json found)", ha="center",
                    va="center", fontsize=20, color="crimson", transform=ax.transAxes)
        fig.savefig("diag_summary.png", dpi=120)
        print("NO DATA — wrote placeholder diag_summary.png")
        return

    _grouped(axes[0],
             rows,
             ["test_torch_matmul", "test_torch_conv2d", "test_torch_attention"],
             ["matmul (4096²)", "conv2d", "attention (SDPA)"],
             logy=True)
    axes[0].set_ylabel("min time (s, log)")
    axes[0].set_title("torch ops by OS — matmul/conv equal; ubuntu ATTENTION ~3x slow")

    _grouped(axes[1],
             rows,
             ["test_suite2p_register", "test_suite2p_full"],
             ["suite2p register (numba)", "suite2p full (+cellpose)"],
             logy=True)
    axes[1].set_ylabel("min time (s, log)")
    axes[1].set_title("suite2p step by OS — registration tight; full = runner-luck spread")

    fig.suptitle("Diagnostics: why ubuntu is slow (attention) + where variance lives "
                 "(not registration)", y=1.0)
    fig.tight_layout()
    fig.savefig("diag_summary.png", dpi=120, bbox_inches="tight")
    print("wrote diag_summary.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
