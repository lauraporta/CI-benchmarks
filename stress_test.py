"""
Stress test: reproduce the CI *variance* mechanism on a local machine.

A hosted CI runner is a small, SHARED box: few vCPUs, and unknown neighbour jobs
competing for them. The theory is that this variable contention -- not any
threading knob -- is what makes the same test leg swing run-to-run. We can
recreate the mechanism locally by pinning cellpose to a CI-like width (4 threads)
and then stealing cores with background CPU burners, watching what the
contention does to both the runtime and its spread.

For each burner count we run cellpose `--reps` times and report min / mean / max
and the coefficient of variation (CV). Rising CV under contention = the variance
is contention, reproduced.

Usage:
    .venv/bin/python stress_test.py --threads 4 --burners 0,4,8 --reps 3
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

# A pure-CPU busy loop in its own process = one saturated core, no GIL sharing
# with the benchmark process.
BURNER = "x = 0\nwhile True:\n    x = (x * x + 1) % 2147483647\n"


def start_burners(n):
    return [subprocess.Popen([sys.executable, "-c", BURNER]) for _ in range(n)]


def stop_burners(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def make_cellpose():
    from cellpose.models import CellposeModel

    rng = np.random.default_rng(0)
    image = rng.random((256, 256), dtype=np.float32)
    model = CellposeModel(pretrained_model="cpsam", gpu=False)
    model.eval(image, diameter=None, cellprob_threshold=0.0, flow_threshold=0.4)

    def run():
        model.eval(image, diameter=None, cellprob_threshold=0.0,
                   flow_threshold=0.4)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4,
                    help="pin cellpose to this many threads (CI-like width)")
    ap.add_argument("--burners", default="0,4,8",
                    help="comma-separated background CPU-burner counts")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="local_artifacts/stress.json")
    args = ap.parse_args()

    burner_counts = [int(b) for b in args.burners.split(",")]
    ncpu = os.cpu_count()
    print(f"os.cpu_count() = {ncpu}   cellpose threads = {args.threads}   "
          f"burners = {burner_counts}   reps = {args.reps}\n")

    import torch
    torch.set_num_threads(args.threads)
    from threadpoolctl import threadpool_limits

    run = make_cellpose()

    print(f"{'burners':<9}{'demand':<8}{'min_s':<10}{'mean_s':<10}"
          f"{'max_s':<10}{'cv_%':<8}{'x_vs_idle':<10}")
    results = {"cpu_count": ncpu, "threads": args.threads, "conditions": []}
    idle_min = None
    for nb in burner_counts:
        procs = start_burners(nb)
        # let the burners spin up and saturate their cores
        time.sleep(2)
        samples = []
        with threadpool_limits(limits=args.threads):
            for _ in range(args.reps):
                t0 = time.perf_counter()
                run()
                samples.append(time.perf_counter() - t0)
        stop_burners(procs)

        mn, mean, mx = min(samples), statistics.mean(samples), max(samples)
        cv = 100 * statistics.pstdev(samples) / mean if mean else float("nan")
        demand = args.threads + nb  # total compute threads competing for ncpu
        idle_min = mn if idle_min is None else idle_min
        print(f"{nb:<9}{demand:<8}{mn:<10.2f}{mean:<10.2f}{mx:<10.2f}"
              f"{cv:<8.1f}{mn / idle_min:<10.2f}")
        results["conditions"].append({
            "burners": nb, "demand": demand, "min": mn, "mean": mean,
            "max": mx, "cv_pct": cv, "samples": samples,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
