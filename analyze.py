"""
Aggregate the bench-*.json artifacts produced by the CI matrix into a table
that answers the two questions:

  1. Is pinning causally faster *within a job*?  -> the default/pin1 ratio,
     measured on the same runner back-to-back (immune to runner luck).
  2. Does pinning shrink the cross-run variance? -> spread of `default` vs
     `pin1` minimum times across the repeated runs.

Usage:
    # download all artifacts from the workflow run into ./artifacts first, e.g.
    #   gh run download <run-id> -D artifacts
    python analyze.py artifacts
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FNAME = re.compile(r"bench-(?P<os>.+?)-py(?P<py>[\d.]+)-run(?P<run>\d+)\.json")


def load(root: Path):
    rows = []
    for path in root.rglob("bench-*.json"):
        m = FNAME.search(path.name)
        if not m:
            continue
        data = json.loads(path.read_text())
        for b in data["benchmarks"]:
            rows.append({
                "os": m["os"],
                "py": m["py"],
                "run": int(m["run"]),
                "test": b["name"].split("[")[0],
                "setting": b["extra_info"].get("setting", "?"),
                "cpu": b["extra_info"].get("cpu_count"),
                "min": b["stats"]["min"],
                "mean": b["stats"]["mean"],
                "stddev": b["stats"]["stddev"],
            })
    return rows


def main(root):
    rows = load(Path(root))
    if not rows:
        print("No bench-*.json found under", root)
        return

    # 1) Within-job causal effect: default/pin1 min-time ratio per (os,py,run,test)
    by_key = defaultdict(dict)
    for r in rows:
        by_key[(r["os"], r["py"], r["run"], r["test"])][r["setting"]] = r

    print("== within-job effect (default min / pin1 min); >1 means pinning is faster ==")
    print(f"{'os':<16}{'py':<6}{'run':<5}{'test':<18}{'cpu':<5}"
          f"{'default_s':<11}{'pin1_s':<10}{'ratio':<7}")
    ratios = defaultdict(list)
    for (os_, py, run, test), s in sorted(by_key.items()):
        if "default" in s and "pin1" in s:
            d, p = s["default"]["min"], s["pin1"]["min"]
            ratio = d / p if p else float("nan")
            ratios[(os_, py, test)].append(ratio)
            print(f"{os_:<16}{py:<6}{run:<5}{test:<18}{str(s['default']['cpu']):<5}"
                  f"{d:<11.3f}{p:<10.3f}{ratio:<7.2f}")

    # 2) Cross-run variance, default vs pin1 (coefficient of variation of min)
    print("\n== cross-run spread of min time (lower CV = more reproducible) ==")
    print(f"{'os':<16}{'py':<6}{'test':<18}{'setting':<9}{'mean_s':<10}{'cv_%':<7}{'n':<3}")
    grp = defaultdict(list)
    for r in rows:
        grp[(r["os"], r["py"], r["test"], r["setting"])].append(r["min"])
    for (os_, py, test, setting), vals in sorted(grp.items()):
        if len(vals) >= 2:
            mean = statistics.mean(vals)
            cv = 100 * statistics.pstdev(vals) / mean if mean else float("nan")
            print(f"{os_:<16}{py:<6}{test:<18}{setting:<9}{mean:<10.3f}{cv:<7.1f}{len(vals):<3}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
