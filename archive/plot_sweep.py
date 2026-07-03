"""
Plot the local thread-count sweeps (time vs threads) for both workloads, with a
vertical marker at os.cpu_count(). The shape past that marker is the answer to
"does oversubscription thrash on THIS box?": flat/falling = no, rising = yes.

Usage:
    .venv/bin/python plot_sweep.py local_artifacts/sweep_cellpose.json \
                                    local_artifacts/sweep_blas.json \
                                    --out local_sweep.png
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(paths):
    series = {}
    ncpu = None
    for p in paths:
        d = json.loads(open(p).read())
        ncpu = d.get("cpu_count", ncpu)
        for b in d["benchmarks"]:
            series.setdefault(b["workload"], []).append((b["threads"], b["min"]))
    for w in series:
        series[w].sort()
    return series, ncpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="local_sweep.png")
    args = ap.parse_args()

    series, ncpu = load(args.paths)
    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    for ax, (name, pts) in zip(axes[0], sorted(series.items())):
        xs = [t for t, _ in pts]
        ys = [v for _, v in pts]
        best = min(ys)
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=2)
        if ncpu:
            ax.axvline(ncpu, color="crimson", ls="--", lw=1.2,
                       label=f"os.cpu_count()={ncpu}")
        ax.set_title(name)
        ax.set_xlabel("torch + BLAS threads")
        ax.set_ylabel("min time (s)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        # annotate slowdown vs best at the extremes
        ax.annotate(f"{ys[-1] / best:.2f}x best",
                    (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(-10, 8), fontsize=9, color="#555")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9)
    fig.suptitle("Local thread-count sweep — time vs threads "
                 "(marker = physical core count)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
