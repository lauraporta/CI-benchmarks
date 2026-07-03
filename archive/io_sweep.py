"""
Local resource sweep for the I/O experiment (PR #3 topic). Two axes:

1. raw disk throughput vs payload size -- write+fsync+read at 10..400 MB. Tells
   us whether disk time is a fixed cost or scales with the data the pipeline
   actually moves (registered stacks are big).
2. snakemake wall-time vs --cores -- does handing the orchestrator more cores
   speed a trivial 3-rule DAG, or is it fixed spawn/DAG overhead (so cores don't
   help and the only lever is *fewer invocations*)?

Usage:
    PATH="$PWD/.venv/bin:$PATH" .venv/bin/python io_sweep.py
"""

import json
import os
import shutil
import statistics
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

SNAKEFILE = textwrap.dedent(
    """
    rule all:
        input: "out/c.npy"
    rule a:
        input: "in.npy"
        output: "out/a.npy"
        run:
            import numpy as np
            np.save(output[0], np.load(input[0]) + 1)
    rule b:
        input: "out/a.npy"
        output: "out/b.npy"
        run:
            import numpy as np
            np.save(output[0], np.load(input[0]) * 2)
    rule c:
        input: "out/b.npy"
        output: "out/c.npy"
        run:
            import numpy as np
            np.save(output[0], np.load(input[0]) - 1)
    """
).lstrip()


def time_min(fn, repeats=3):
    s = [(_t(fn)) for _ in range(repeats)]
    return min(s), statistics.mean(s), s


def _t(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def raw_io_size_sweep(sizes_mb, repeats=3):
    print(f"{'raw_file_io':<14}{'MB':<7}{'min_s':<10}{'MB/s':<9}")
    out = []
    for mb in sizes_mb:
        data = np.random.default_rng(0).integers(
            -30000, 30000, (mb * 1024 * 1024 // 2,), dtype=np.int16)

        def run():
            with tempfile.NamedTemporaryFile(delete=False) as f:
                path = f.name
                f.write(data.tobytes())
                f.flush()
                os.fsync(f.fileno())
            np.fromfile(path, dtype=np.int16)
            os.unlink(path)

        mn, mean, _ = time_min(run, repeats)
        print(f"{'':<14}{mb:<7}{mn:<10.4f}{mb / mn:<9.0f}")
        out.append({"size_mb": mb, "min": mn, "mean": mean, "mb_per_s": mb / mn})
    print()
    return out


def snakemake_cores_sweep(cores_list, repeats=3):
    print(f"{'snakemake_io':<14}{'cores':<7}{'min_s':<10}{'mean_s':<10}")
    out = []
    for cores in cores_list:
        def run():
            work = Path(tempfile.mkdtemp(prefix="smk"))
            try:
                (work / "Snakefile").write_text(SNAKEFILE)
                np.save(work / "in.npy",
                        np.random.default_rng(0).random((2000, 2000)))
                cmd = ["snakemake", "--cores", str(cores),
                       "-s", str(work / "Snakefile"), "--nolock"]
                r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-1500:])
            finally:
                shutil.rmtree(work, ignore_errors=True)

        mn, mean, _ = time_min(run, repeats)
        print(f"{'':<14}{cores:<7}{mn:<10.4f}{mean:<10.4f}")
        out.append({"cores": cores, "min": mn, "mean": mean})
    print()
    return out


def main():
    print(f"os.cpu_count() = {os.cpu_count()}\n")
    results = {
        "cpu_count": os.cpu_count(),
        "raw_io_size": raw_io_size_sweep([10, 50, 100, 200, 400]),
        "snakemake_cores": snakemake_cores_sweep([1, 2, 4, 8]),
    }
    os.makedirs("local_artifacts", exist_ok=True)
    with open("local_artifacts/io_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote local_artifacts/io_sweep.json")


if __name__ == "__main__":
    main()
