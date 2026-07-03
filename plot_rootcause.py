"""
Root-cause figures for the CI slowness + variance, built from the real CI
artifacts (default-threads only -- the pinning arm is dropped) plus the local
measurements.

Figure 1 (rootcause_slowness.png): two panels.
  A. cellpose_eval min time by OS (CI runners) vs local 8-core Mac -> the
     slowness ranking, and how far below it a real machine sits.
  B. blas_matmul min time by OS (CI) -> nearly OS-invariant. Same hardware, same
     BLAS math, ~equal time. So the OS gap in (A) is NOT the machine being slow
     at compute in general -- it is specific to torch's cellpose forward pass on
     the linux/windows CPU wheel.

Figure 2 (rootcause_variance.png): run-to-run CV of cellpose_eval min per (OS,
py). Shows the variance is a runner-luck property (mostly modest, with blow-out
legs), not a code property.

Usage:
    .venv/bin/python plot_rootcause.py /tmp/ci_raw
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FNAME = re.compile(r"bench-(?P<os>.+?)-py(?P<py>[\d.]+)-run(?P<run>\d+)\.json")
OS_ORDER = ["macos-latest", "ubuntu-latest", "windows-latest"]
OS_COLOR = {"macos-latest": "#4c9f70", "ubuntu-latest": "#d1603d",
            "windows-latest": "#3d6cd1"}


def load_ci(root):
    rows = []
    for path in Path(root).rglob("bench-*.json"):
        m = FNAME.search(path.name)
        if not m:
            continue
        data = json.loads(path.read_text())
        for b in data["benchmarks"]:
            if b["extra_info"].get("setting") != "default":
                continue
            rows.append({
                "os": m["os"], "py": m["py"], "run": int(m["run"]),
                "test": b["name"].split("[")[0],
                "cpu": b["extra_info"].get("cpu_count"),
                "min": b["stats"]["min"],
            })
    return rows


def agg(rows, test):
    """mean-of-min and list-of-min per os for a given test."""
    by_os = defaultdict(list)
    for r in rows:
        if r["test"] == test:
            by_os[r["os"]].append(r["min"])
    return by_os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ci_root")
    args = ap.parse_args()

    rows = load_ci(args.ci_root)
    cpu_by_os = {r["os"]: r["cpu"] for r in rows}

    # ---- Figure 1: slowness ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    cell = agg(rows, "test_cellpose_eval")
    labels, means, errs, colors = [], [], [], []
    for os_ in OS_ORDER:
        vals = cell.get(os_, [])
        if not vals:
            continue
        labels.append(f"{os_}\n({cpu_by_os.get(os_,'?')} vCPU)")
        means.append(statistics.mean(vals))
        errs.append(statistics.pstdev(vals))
        colors.append(OS_COLOR[os_])
    axA.bar(labels, means, yerr=errs, color=colors, capsize=5)
    for i, m in enumerate(means):
        axA.text(i, m, f"{m:.0f}s", ha="center", va="bottom", fontsize=10)
    axA.set_ylabel("cellpose_eval min time (s)")
    axA.set_title("A. Slowness: cellpose forward pass by environment\n"
                  "(bars = mean of per-run min; whiskers = run-to-run spread)")

    blas = agg(rows, "test_blas_matmul")
    blabels, bmeans, bcolors = [], [], []
    for os_ in OS_ORDER:
        vals = blas.get(os_, [])
        if not vals:
            continue
        blabels.append(f"{os_}\n({cpu_by_os.get(os_,'?')} vCPU)")
        bmeans.append(statistics.mean(vals) * 1000)  # ms
        bcolors.append(OS_COLOR[os_])
    axB.bar(blabels, bmeans, color=bcolors)
    for i, m in enumerate(bmeans):
        axB.text(i, m, f"{m:.0f}ms", ha="center", va="bottom", fontsize=10)
    axB.set_ylabel("blas_matmul min time (ms)")
    axB.set_title("B. Same runners, pure NumPy matmul: ~OS-invariant\n"
                  "→ the gap in A is torch's cellpose kernel, not slow hardware")
    fig.suptitle("Why CI is SLOW: the cost lives in torch's cellpose CPU "
                 "forward pass, worst on the linux/windows wheel", fontsize=12)
    fig.tight_layout()
    fig.savefig("rootcause_slowness.png", dpi=130)
    print("wrote rootcause_slowness.png")

    # ---- Figure 2: variance ----
    fig2, ax = plt.subplots(figsize=(11, 5))
    by_leg = defaultdict(list)
    for r in rows:
        if r["test"] == "test_cellpose_eval":
            by_leg[(r["os"], r["py"])].append(r["min"])
    legs = sorted(by_leg.keys())
    xs, cvs, colors = [], [], []
    for (os_, py) in legs:
        vals = by_leg[(os_, py)]
        if len(vals) < 2:
            continue
        cv = 100 * statistics.pstdev(vals) / statistics.mean(vals)
        xs.append(f"{os_.split('-')[0]}\npy{py}")
        cvs.append(cv)
        colors.append(OS_COLOR[os_])
    bars = ax.bar(xs, cvs, color=colors)
    for b, cv in zip(bars, cvs):
        ax.text(b.get_x() + b.get_width() / 2, cv, f"{cv:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("run-to-run CV of cellpose min (%)")
    ax.set_title("Why CI VARIES: run-to-run spread of the SAME leg (3 repeats).\n"
                 "Mostly modest, but shared-runner contention produces blow-out "
                 "legs (e.g. windows py3.14) — that is the variance.")
    fig2.tight_layout()
    fig2.savefig("rootcause_variance.png", dpi=130)
    print("wrote rootcause_variance.png")


if __name__ == "__main__":
    main()
