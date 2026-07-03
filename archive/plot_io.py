"""
Render the I/O experiment figure from io-*.json artifacts: per OS, raw_file_io vs
snakemake_io -- bar = mean, dots = each run, and the cross-run CV annotated. If
snakemake_io's dots are far more spread (higher CV) than raw_file_io's, the
variance lives in snakemake orchestration / --latency-wait, not the disk.

Usage:  python plot_io.py artifacts
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FN = re.compile(r"io-(?P<os>.+?)-py(?P<py>[\d.]+)-run(?P<run>\d+)\.json")
OSES = ["macos-latest", "ubuntu-latest", "windows-latest"]
TESTS = ["test_raw_file_io", "test_snakemake_io"]
COL = {"test_raw_file_io": "tab:gray", "test_snakemake_io": "tab:purple"}


def load(root):
    rows = []
    for p in Path(root).rglob("io-*.json"):
        m = FN.search(p.name)
        if not m:
            continue
        for b in json.loads(p.read_text())["benchmarks"]:
            rows.append({"os": m["os"], "test": b["name"].split("[")[0],
                         "min": b["stats"]["min"], "mean": b["stats"]["mean"]})
    return rows


def main(root):
    rows = load(root)
    fig, ax = plt.subplots(figsize=(11, 6))
    if not rows:
        ax.text(0.5, 0.5, "NO DATA\n(no io-*.json found)", ha="center",
                va="center", fontsize=20, color="crimson", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig("io_summary.png", dpi=120)
        print("NO DATA — wrote placeholder io_summary.png")
        return

    width = 0.38
    for j, test in enumerate(TESTS):
        for i, os_ in enumerate(OSES):
            vals = [r["mean"] for r in rows if r["test"] == test and r["os"] == os_]
            if not vals:
                continue
            x = i + (j - 0.5) * width
            m = statistics.mean(vals)
            cv = 100 * statistics.pstdev(vals) / m if len(vals) > 1 and m else 0
            ax.bar(x, m, width=width * 0.9, color=COL[test], alpha=0.35,
                   label=test.replace("test_", "") if i == 0 else None)
            ax.scatter([x] * len(vals), vals, s=22, color=COL[test], zorder=3,
                       edgecolor="k", linewidth=0.3)
            ax.annotate(f"CV {cv:.0f}%", (x, max(vals)), ha="center",
                        va="bottom", fontsize=8)
    ax.set_xticks(range(len(OSES)))
    ax.set_xticklabels([o.split("-")[0] for o in OSES])
    ax.set_ylabel("time per run (s)")
    ax.set_title("File I/O vs snakemake orchestration — mean (bar), each run (dots), "
                 "cross-run CV")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("io_summary.png", dpi=120, bbox_inches="tight")
    print("wrote io_summary.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
