"""
Controlled benchmark: is the photon-mosaic-pipeline CI runtime variance caused
by thread oversubscription in the cellpose-SAM CPU forward pass?

Background
----------
On photon-mosaic-pipeline CI, the same 21-test suite took 1733 s on ubuntu
py3.14 but 8219 s on ubuntu py3.12 (same OS, identical resolved packages,
identical ~33 s install). The blow-up is entirely inside the test compute --
cellpose-SAM + suite2p running on CPU -- and the slow leg varies run to run.
Hypothesis: torch, the BLAS backend and numba each default to grabbing every
core, so on a 4-core runner you get ~3x oversubscription that thrashes badly
when the (shared) runner is loaded -- and how badly depends on which physical
machine you landed on, which is why it looks random.

Design
------
The killer confound is *which runner you got*. To remove it we compare control
(default threads) vs treatment (pinned threads) **back-to-back in the same
process, on the same runner**: if pinning wins within a job, that is causal
evidence for the threading mechanism, independent of runner luck. pytest-
benchmark repeats each condition and reports min / mean / stddev. The workflow
then repeats the whole job across OS x python x run-index to sample the
runner-luck dimension.

Two workloads are timed:
- ``cellpose_eval``  -- the real hot op (a cpsam forward pass on CPU).
- ``blas_matmul``    -- a pure NumPy matmul, to isolate whether BLAS thread
                        oversubscription alone reproduces the effect.

Every run first prints the thread environment so we can *see* whether
oversubscription is even happening (e.g. torch=4, BLAS=4, numba=4 on 4 cores).
"""

import os

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

# Thread settings compared within each process. ``None`` = leave at the
# (oversubscribing) default; an int pins torch + BLAS to that many threads.
THREAD_SETTINGS = {
    "default": None,
    "pin1": 1,
    "pin2": 2,
}


def _print_environment():
    import torch

    print("\n=== thread environment ===")
    print("platform           :", os.sys.platform)
    print("os.cpu_count()     :", os.cpu_count())
    print("torch              :", torch.__version__)
    print("numpy              :", np.__version__)
    try:
        import cellpose

        print("cellpose           :", cellpose.__version__)
    except Exception as exc:  # pragma: no cover - informational only
        print("cellpose version   : (unavailable)", exc)
    print("torch.num_threads  :", torch.get_num_threads())
    print("torch.num_interop  :", torch.get_num_interop_threads())
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        print(f"{var:<19}:", os.environ.get(var, "(unset)"))
    for pool in threadpool_info():
        print("  BLAS pool        :", pool.get("internal_api"),
              "num_threads=", pool.get("num_threads"),
              "(", pool.get("filepath", "").split("/")[-1], ")")
    print("=== end environment ===\n")


@pytest.fixture(scope="session", autouse=True)
def _env_banner():
    _print_environment()


@pytest.fixture(scope="session")
def image():
    # Content is irrelevant for timing a fixed-size ViT forward pass; a fixed
    # seed keeps the work identical across legs. 256x256 ~ the registered
    # max-projection the pipeline feeds cellpose.
    rng = np.random.default_rng(0)
    return rng.random((256, 256), dtype=np.float32)


@pytest.fixture(scope="session")
def cellpose_model(image):
    from cellpose.models import CellposeModel

    # gpu=False on every OS: we are measuring the CPU compute the runners
    # actually do (no GPU on hosted runners; no MPS confound on macOS).
    model = CellposeModel(pretrained_model="cpsam", gpu=False)
    # Warm up once (downloads weights + triggers lazy init) outside any timed
    # region so the benchmarked rounds are steady-state.
    model.eval(image, diameter=None, cellprob_threshold=0.0, flow_threshold=0.4)
    return model


def _apply_threads(n):
    """Pin torch (and return a BLAS-limit context) to ``n`` threads, or to the
    oversubscribing default when ``n`` is None."""
    import torch

    if n is None:
        torch.set_num_threads(os.cpu_count() or 1)
        return threadpool_limits(limits=None)  # no-op context: BLAS at default
    torch.set_num_threads(n)
    return threadpool_limits(limits=n)


@pytest.mark.parametrize("setting", list(THREAD_SETTINGS))
def test_cellpose_eval(benchmark, cellpose_model, image, setting):
    """Time a cpsam CPU forward pass under default vs pinned threads."""
    n = THREAD_SETTINGS[setting]

    def run():
        with _apply_threads(n):
            return cellpose_model.eval(
                image, diameter=None, cellprob_threshold=0.0,
                flow_threshold=0.4,
            )[0]

    benchmark.extra_info["setting"] = setting
    benchmark.extra_info["threads"] = "default" if n is None else str(n)
    benchmark.extra_info["cpu_count"] = os.cpu_count()
    masks = benchmark(run)
    # Sanity only: the op produced an array. Mask count is not asserted (it is
    # not what we measure).
    assert masks is not None


@pytest.mark.parametrize("setting", list(THREAD_SETTINGS))
def test_blas_matmul(benchmark, setting):
    """Time a pure BLAS matmul under default vs pinned threads -- isolates
    whether BLAS oversubscription alone reproduces the effect."""
    n = THREAD_SETTINGS[setting]
    rng = np.random.default_rng(1)
    a = rng.random((2000, 2000), dtype=np.float64)
    b = rng.random((2000, 2000), dtype=np.float64)

    def run():
        with threadpool_limits(limits=n):  # None = default (all cores)
            return a @ b

    benchmark.extra_info["setting"] = setting
    benchmark.extra_info["threads"] = "default" if n is None else str(n)
    benchmark.extra_info["cpu_count"] = os.cpu_count()
    out = benchmark(run)
    assert out.shape == (2000, 2000)


@pytest.mark.parametrize("setting", list(THREAD_SETTINGS))
def test_torch_matmul(benchmark, setting):
    """ADVERSARY CONTROL. The existing ``test_blas_matmul`` control is a **NumPy
    float64** GEMM -- a *different library* (OpenBLAS / Accelerate / MKL,
    whichever numpy resolved) at a *different precision* than the code that is
    actually slow. cpsam is a SAM/ViT transformer: its hot op is a **torch
    float32** matmul inside attention. If the OS slowness lives in torch's
    float32 GEMM backend on the linux/windows wheel, a NumPy-float64 matmul is
    blind to it and would look OS-invariant *even though the hardware/wheel is
    not*. This control uses the same framework (torch), dtype (float32) and
    kernel family (GEMM) as cpsam, so it can *see* an effect the numpy control
    cannot -- distinguishing 'the torch wheel is slow' from 'the numpy BLAS is
    fine.'"""
    import torch

    n = THREAD_SETTINGS[setting]
    torch.manual_seed(1)
    # Same 2000x2000 shape as test_blas_matmul so the ONLY differences vs that
    # control are library (numpy->torch) and dtype (float64->float32). Looped so
    # the timed region is comfortably measurable and dominated by the GEMM.
    a = torch.rand(2000, 2000, dtype=torch.float32)
    b = torch.rand(2000, 2000, dtype=torch.float32)

    def run():
        with _apply_threads(n):
            with torch.no_grad():
                out = a
                for _ in range(8):
                    out = out @ b
            return out

    benchmark.extra_info["setting"] = setting
    benchmark.extra_info["threads"] = "default" if n is None else str(n)
    benchmark.extra_info["cpu_count"] = os.cpu_count()
    out = benchmark(run)
    assert out.shape == (2000, 2000)
