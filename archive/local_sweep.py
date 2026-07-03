"""
Local thread-count sweep: draw the oversubscription curve on THIS machine.

The CI matrix can only compare 3 fixed thread settings (default/pin1/pin2) and
it lands on 3-4 core runners, so it can never see what happens *past* the core
count. Locally we own the box, so we sweep torch+BLAS thread counts across a
range that straddles the physical core count and time both workloads at each
point. That turns the yes/no "is oversubscription real?" question into a curve:

  time(threads) for cellpose_eval and blas_matmul, min over repeats.

Oversubscription, if it thrashes, shows up as the curve turning back *up* once
threads exceed os.cpu_count(). If the curve keeps falling (or flattens) past the
core count, oversubscription is not hurting on this hardware -- which is exactly
what CI's default-beats-pin1 result implied.

Usage:
    .venv/bin/python local_sweep.py [--threads 1,2,4,6,8,12,16] [--repeats 3]
                                    [--out local_artifacts/sweep.json]
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_limits


def time_min(fn, repeats):
    """Return the min wall-time of `fn` over `repeats` calls (min = least
    contended sample, same statistic pytest-benchmark headlines)."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return min(samples), statistics.mean(samples), samples


def make_cellpose():
    from cellpose.models import CellposeModel

    rng = np.random.default_rng(0)
    image = rng.random((256, 256), dtype=np.float32)
    model = CellposeModel(pretrained_model="cpsam", gpu=False)
    # Warm up once outside the timed region (weights download + lazy init).
    model.eval(image, diameter=None, cellprob_threshold=0.0, flow_threshold=0.4)

    def run():
        model.eval(image, diameter=None, cellprob_threshold=0.0,
                   flow_threshold=0.4)

    return run


def make_blas():
    rng = np.random.default_rng(1)
    a = rng.random((2000, 2000), dtype=np.float64)
    b = rng.random((2000, 2000), dtype=np.float64)

    def run():
        return a @ b

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,2,4,6,8,12,16",
                    help="comma-separated thread counts to sweep")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", choices=["blas", "cellpose", "both"],
                    default="both", help="which workload(s) to sweep")
    ap.add_argument("--out", default="local_artifacts/sweep.json")
    args = ap.parse_args()

    threads = [int(t) for t in args.threads.split(",")]
    ncpu = os.cpu_count()
    print(f"os.cpu_count() = {ncpu}   sweeping threads = {threads}   "
          f"repeats = {args.repeats}\n")

    import torch
    print(f"torch {torch.__version__}  numpy {np.__version__}\n")

    workloads = {}
    if args.only in ("blas", "both"):
        workloads["blas_matmul"] = make_blas()
    if args.only in ("cellpose", "both"):
        workloads["cellpose_eval"] = make_cellpose()

    results = {"cpu_count": ncpu, "threads": threads, "benchmarks": []}
    header = f"{'workload':<16}{'threads':<9}{'min_s':<11}{'mean_s':<11}{'vs_best':<8}"
    for name, fn in workloads.items():
        print(header)
        best = None
        rows = []
        for n in threads:
            torch.set_num_threads(n)
            with threadpool_limits(limits=n):
                mn, mean, samples = time_min(fn, args.repeats)
            best = mn if best is None else min(best, mn)
            rows.append((n, mn, mean, samples))
        for n, mn, mean, samples in rows:
            flag = "  <-- past ncpu" if n > ncpu else ""
            print(f"{name:<16}{n:<9}{mn:<11.4f}{mean:<11.4f}"
                  f"{mn / best:<8.2f}{flag}")
            results["benchmarks"].append({
                "workload": name, "threads": n, "min": mn, "mean": mean,
                "samples": samples,
            })
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
