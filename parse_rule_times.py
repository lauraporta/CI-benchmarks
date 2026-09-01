"""Parse a snakemake --cores 1 log into per-rule wall times.

Snakemake prints a `[Mon Aug 25 17:00:00 2026]` timestamp line immediately
before each `rule <name>:` block and before each `Finished job N.` line.
With --cores 1 execution is strictly sequential, so each job's wall time is
finish_ts - start_ts of the enclosing pair. Output: a per-job CSV on stdout
plus a per-rule summary (count, total, mean) so the job log itself answers
"which rule ate the minutes" without downloading artifacts.
"""

import re
import sys
from datetime import datetime

# Snakemake may prefix lines with "INFO:snakemake.workflow:" depending on
# how logging is wired; tolerate it everywhere.
P = r"^(?:INFO:[\w.]+:)?"
TS = re.compile(P + r"\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4})\]")
RULE = re.compile(P + r"(?:checkpoint |local)?rule (\w+):")
# FIN deliberately NOT prefix-tolerant: snakemake also emits an
# "INFO:snakemake.logging:Finished jobid:" duplicate that can arrive BEFORE
# the timestamped bare copy — matching it would pair the finish with a stale
# timestamp and report 0 s. Only the bare line follows a fresh [timestamp].
FIN = re.compile(r"^Finished jobid: \d+|^Finished job \d+")
WC = re.compile(r"^\s+wildcards: (.+)")


def parse_ts(s):
    return datetime.strptime(re.sub(r" +", " ", s), "%a %b %d %H:%M:%S %Y")


def main(path):
    last_ts = None
    current = None  # [rule, wildcards, start_ts]
    jobs = []
    with open(path, errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if m:
                last_ts = parse_ts(m.group(1))
                continue
            m = RULE.match(line)
            if m and last_ts is not None:
                current = [m.group(1), "", last_ts]
                continue
            m = WC.match(line)
            if m and current:
                current[1] = m.group(1).strip()[:60]
                continue
            if FIN.match(line) and current and last_ts is not None:
                rule, wc, start = current
                jobs.append((rule, wc, (last_ts - start).total_seconds()))
                current = None

    print("rule,wildcards,seconds")
    for rule, wc, secs in jobs:
        print(f'{rule},"{wc}",{secs:.0f}')

    print("\n=== per-rule summary ===")
    agg = {}
    for rule, _, secs in jobs:
        agg.setdefault(rule, []).append(secs)
    total = sum(s for v in agg.values() for s in v) or 1
    for rule, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        print(
            f"{rule:<24} n={len(v):<3} total={sum(v):7.0f}s "
            f"mean={sum(v)/len(v):6.1f}s share={sum(v)/total*100:5.1f}%"
        )
    print(f"{'ALL':<24} total={total:7.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
