"""
Where does the *run-to-run* variance live? The isolated cellpose bench was
stable within each OS, so the historical 5x same-leg blow-up is not cellpose.
The next stage is suite2p registration, which is numba-heavy and so far
unmeasured. We run a real suite2p step on a synthetic 1-session movie with
registration-only and full (registration + cellpose detection) timed
separately, repeated, to see (a) whether registration carries the variance and
(b) whether it is also disproportionately slow on ubuntu.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

# Force cellpose onto CPU on every OS (matches the pipeline's CI behaviour and
# keeps the cross-OS comparison a pure-CPU one; no MPS confound on macOS).
import cellpose.core  # noqa: E402

cellpose.core.use_gpu = lambda *args, **kwargs: False

from suite2p import run_s2p  # noqa: E402
from suite2p.default_ops import default_ops  # noqa: E402


def _synth_movie(path, n=250, h=128, w=128, ncells=12, seed=0):
    """A small movie with a few time-varying Gaussian blobs so registration has
    signal and cellpose has something to find."""
    rng = np.random.default_rng(seed)
    mov = rng.normal(100, 5, (n, h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cys = rng.integers(12, h - 12, ncells)
    cxs = rng.integers(12, w - 12, ncells)
    for cy, cx in zip(cys, cxs):
        trace = 150 * (1 + np.sin(np.arange(n) / 12 + cx)) / 2
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))
        mov += trace[:, None, None] * blob[None]
    tifffile.imwrite(path, mov.astype(np.int16))


def _ops(data_path, save, roidetect):
    ops = default_ops()
    ops.update(
        roidetect=roidetect,
        spikedetect=roidetect,
        do_registration=1,
        anatomical_only=4,
        diameter=0,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        pretrained_model="cpsam",
    )
    ops["data_path"] = [str(data_path)]
    ops["save_path0"] = str(save)
    ops["save_folder"] = str(save)
    ops["fast_disk"] = str(save)
    return ops


@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("rawmovie")
    _synth_movie(d / "recording.tif")
    return d


def _run(data_path, roidetect):
    save = Path(tempfile.mkdtemp(prefix="s2p_"))
    try:
        run_s2p(ops=_ops(data_path, save, roidetect))
    finally:
        shutil.rmtree(save, ignore_errors=True)


def test_suite2p_register(benchmark, data_path):
    """Registration only (numba) -- does this stage carry the variance?"""
    benchmark.extra_info["cpu_count"] = os.cpu_count()
    benchmark.pedantic(lambda: _run(data_path, False), rounds=3, iterations=1)


def test_suite2p_full(benchmark, data_path):
    """Registration + cellpose detection -- the full pipeline step."""
    benchmark.extra_info["cpu_count"] = os.cpu_count()
    benchmark.pedantic(lambda: _run(data_path, True), rounds=3, iterations=1)
