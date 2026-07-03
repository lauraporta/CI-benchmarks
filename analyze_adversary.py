"""
ADVERSARY analysis for the matmul-control experiment (adversary/matmul-control).

Attacks ROOT_CAUSE.md's headline: "the Linux/Windows torch *wheel* runs
cellpose's CPU math ~3x slower than macOS; the hardware is equally fast -- proven
by an OS-invariant NumPy matmul."

The existing control (test_blas_matmul) is a NumPy *float64* GEMM: a different
library and precision than cpsam's actual hot op (a torch *float32* attention
matmul). We added test_torch_matmul (same framework + dtype + kernel family as
cpsam) and persisted per-leg confounds (torch version, CPU model).

Pre-registered decision rules (write predictions BEFORE the data):

  R1 (control validity / claim D -> A):
     Compare the per-OS gap of test_torch_matmul vs test_blas_matmul.
     - If torch-f32 matmul shows an OS gap of similar magnitude to cellpose
       (>= ~2x macOS->linux) while numpy-f64 matmul stays flat (< ~1.3x):
         => the numpy control was BLIND to the effect. The slowness is torch's
            float32 GEMM backend on the linux/windows wheel. SUPPORTS the
            "it's the wheel" mechanism but FALSIFIES the paper's control:
            "a plain NumPy matmul is equally fast" does NOT prove hardware parity
            for the code that is slow.
     - If BOTH matmuls stay flat (< ~1.3x) but cellpose is still ~3x:
         => the 3x is NOT torch's GEMM at all; it lives in cellpose's non-GEMM
            path. WEAKENS claim A as written ("the torch build runs the CPU math
            3x slower").
     - If torch-f32 matmul is flat and numpy matmul is flat and cellpose ~3x,
       AND torch versions differ across legs (see R2): the gap is confounded.

  R2 (identical-packages premise / claim A):
     If torch (or numpy) VERSION differs across the legs being compared
     (esp. macOS wheel vs linux wheel, or across python versions):
         => "same OS, identical packages" is FALSE/unverified; the OS gap
            conflates wheel-build with version. WEAKENS A.

  R3 (faster-physical-machine premise / claim C):
     If a leg's 3 runs share one CPU model yet its cellpose min swings > 3x:
         => "it landed on a faster physical machine" is NOT supported by the
            hardware record. WEAKENS C. If the CPU model DOES vary run-to-run,
            C survives for that leg.

Usage:
    python analyze_adversary.py artifacts
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

FNAME = re.compile(r"bench-(?P<os>.+?)-py(?P<py>[\d.]+)-run(?P<run>\d+)\.json")
OS_ORDER = ["macos-latest", "ubuntu-latest", "windows-latest"]
TESTS = ["test_cellpose_eval", "test_blas_matmul", "test_torch_matmul"]


def load_bench(root):
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
                "min": b["stats"]["min"],
            })
    return rows


def load_env(root):
    envs = {}
    for path in Path(root).rglob("env-*.json"):
        d = json.loads(path.read_text())
        pkgs = {}
        for line in d.get("pip_freeze", []):
            if "==" in line:
                name, _, ver = line.partition("==")
                pkgs[name.lower()] = ver
        envs[d.get("leg")] = {
            "cpu_model": d.get("cpu_model", "?"),
            "torch": pkgs.get("torch", "?"),
            "numpy": pkgs.get("numpy", "?"),
            "cellpose": pkgs.get("cellpose", "?"),
            "os_cpu_count": d.get("os_cpu_count"),
        }
    return envs


def per_os_mean_of_min(rows, test):
    by_os = defaultdict(list)
    for r in rows:
        if r["test"] == test:
            by_os[r["os"]].append(r["min"])
    return {o: statistics.mean(v) for o, v in by_os.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()

    rows = load_bench(args.root)
    envs = load_env(args.root)

    print("=" * 70)
    print("R1  per-OS mean-of-min, and gap vs macOS (the control-validity test)")
    print("=" * 70)
    base = {}
    for test in TESTS:
        means = per_os_mean_of_min(rows, test)
        mac = means.get("macos-latest")
        base[test] = mac
        print(f"\n{test}:")
        for os_ in OS_ORDER:
            if os_ not in means:
                continue
            v = means[os_]
            gap = f"{v / mac:.2f}x" if mac else "n/a"
            unit = "s" if test == "test_cellpose_eval" else "ms"
            scale = 1 if test == "test_cellpose_eval" else 1000
            print(f"   {os_:<16} {v * scale:8.1f} {unit}   gap vs macOS = {gap}")
    print("\n-> If torch_matmul gap tracks cellpose but blas_matmul stays flat,")
    print("   the NumPy control was blind: ROOT_CAUSE's control is FALSIFIED.")

    print("\n" + "=" * 70)
    print("R2  package versions per leg (the 'identical packages' premise)")
    print("=" * 70)
    torch_vers = defaultdict(set)
    for leg, e in sorted(envs.items()):
        print(f"   {leg:<34} torch={e['torch']:<14} numpy={e['numpy']:<10} "
              f"cellpose={e['cellpose']}")
        os_ = leg.split("-py")[0]
        torch_vers[os_].add(e["torch"])
    print()
    for os_, vers in sorted(torch_vers.items()):
        flag = "  <-- MULTIPLE torch versions within this OS!" if len(vers) > 1 else ""
        print(f"   {os_:<16} torch versions seen: {sorted(vers)}{flag}")

    print("\n" + "=" * 70)
    print("R3  CPU model vs cellpose swing per leg (the 'faster machine' premise)")
    print("=" * 70)
    by_leg = defaultdict(list)
    for r in rows:
        if r["test"] == "test_cellpose_eval":
            by_leg[(r["os"], r["py"])].append((r["run"], r["min"]))
    for (os_, py), runs in sorted(by_leg.items()):
        if len(runs) < 2:
            continue
        mins = [m for _, m in runs]
        swing = max(mins) / min(mins)
        models = {envs.get(f"{os_}-py{py}-run{run}", {}).get("cpu_model", "?")
                  for run, _ in runs}
        verdict = ""
        if swing > 3 and len(models) == 1:
            verdict = "  <-- >3x swing on ONE cpu model: 'faster machine' NOT supported"
        elif len(models) > 1:
            verdict = "  (cpu model varies across runs -> 'faster machine' plausible)"
        print(f"   {os_}-py{py}: swing={swing:.1f}x  cpu_models={models}{verdict}")


if __name__ == "__main__":
    main()
