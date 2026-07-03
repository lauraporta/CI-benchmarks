"""
Does file read/write + snakemake orchestration introduce run-to-run variance,
independent of the cellpose/suite2p compute?

Prior experiments showed cellpose and suite2p-registration are stable within-OS,
leaving runner-luck on compute as the variance source. But the real pipeline also
spawns snakemake, walks a DAG, writes/reads intermediate files, and polls with
--latency-wait. This probes that layer with NO heavy compute:

- raw_file_io:  write ~50 MB + fsync + read back (pure disk throughput).
- snakemake_io: a minimal 3-rule DAG (read -> transform -> write, trivial numpy)
                run via the `snakemake` subprocess with the pipeline's flags
                (--cores 1, --latency-wait 30 on CI, --nolock).

The tell: if snakemake_io has a much higher CV than raw_file_io (and the compute
benches), the variance lives in snakemake orchestration / latency-wait, not the
disk or the math.
"""

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

SIZE_MB = 50

# A minimal DAG mirroring the pipeline shape: read an input, write a chain of
# intermediate outputs. Trivial numpy so the cost is spawn + DAG + I/O, not math.
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


def test_raw_file_io(benchmark, tmp_path):
    """Pure disk: write ~50 MB with fsync, read it back."""
    data = np.random.default_rng(0).integers(
        -30000, 30000, (SIZE_MB * 1024 * 1024 // 2,), dtype=np.int16
    )

    def run():
        p = tmp_path / "blob.bin"
        with open(p, "wb") as f:
            f.write(data.tobytes())
            f.flush()
            os.fsync(f.fileno())
        n = np.fromfile(p, dtype=np.int16).size
        p.unlink()
        return n

    benchmark.extra_info["cpu_count"] = os.cpu_count()
    benchmark.pedantic(run, rounds=5, iterations=1)


def _run_snakemake(work):
    (work / "Snakefile").write_text(SNAKEFILE)
    # ~32 MB input so each rule moves a non-trivial amount through disk.
    np.save(work / "in.npy", np.random.default_rng(0).random((2000, 2000)))
    cmd = ["snakemake", "--cores", "1", "-s", str(work / "Snakefile")]
    if os.getenv("CI"):
        cmd += ["--nolock", "--latency-wait", "30"]
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"snakemake failed:\n{r.stderr[-2000:]}")


def test_snakemake_io(benchmark, tmp_path_factory):
    """snakemake orchestration + I/O with trivial compute."""

    def run():
        work = tmp_path_factory.mktemp("smk")
        try:
            _run_snakemake(work)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    benchmark.extra_info["cpu_count"] = os.cpu_count()
    benchmark.pedantic(run, rounds=5, iterations=1)
