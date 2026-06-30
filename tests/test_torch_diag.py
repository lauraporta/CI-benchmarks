"""
Why is cellpose CPU inference ~7x slower on the ubuntu runner than on
windows/macOS, when the pure NumPy BLAS matmul is the same across all three?

We (1) dump torch's build/threading config per OS as an artifact, and (2)
benchmark torch-native ops at default threads -- matmul (torch's own BLAS),
conv2d (oneDNN/MKLDNN), and scaled-dot-product attention (the ViT core) -- to
localize the slowdown to a specific op class / backend rather than guessing.
"""

import os
import platform
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _tag():
    return (f"{os.environ.get('RUNNER_OS', platform.system())}"
            f"-py{sys.version_info.major}.{sys.version_info.minor}")


def test_dump_torch_config():
    """Write torch's build + threading config to an artifact file (and stdout).
    Not a benchmark -- this is the 'what backend is torch actually using' probe.
    """
    lines = [
        f"tag                 : {_tag()}",
        f"platform            : {platform.platform()}",
        f"processor           : {platform.processor()}",
        f"python              : {sys.version.split()[0]}",
        f"os.cpu_count        : {os.cpu_count()}",
        f"torch               : {torch.__version__}",
        f"torch.num_threads   : {torch.get_num_threads()}",
        f"torch.interop       : {torch.get_num_interop_threads()}",
        f"mkldnn.is_available : {torch.backends.mkldnn.is_available()}",
        f"mkl.is_available    : {torch.backends.mkl.is_available()}",
        f"openmp.is_available : {torch.backends.openmp.is_available()}",
        "=== torch.__config__.parallel_info() ===",
        torch.__config__.parallel_info(),
        "=== torch.__config__.show() ===",
        torch.__config__.show(),
    ]
    text = "\n".join(lines)
    print("\n" + text)
    Path(f"torch-config-{_tag()}.txt").write_text(text)


@pytest.fixture(autouse=True)
def _default_threads():
    # Measure at the runner's natural default (all cores), which is what the
    # pipeline actually uses.
    torch.set_num_threads(os.cpu_count() or 1)


@pytest.mark.parametrize("size", [4096])
def test_torch_matmul(benchmark, size):
    """torch's own linear algebra (vs the NumPy BLAS matmul in the other file --
    if this is 7x slower on ubuntu but NumPy isn't, torch ships a slower BLAS)."""
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    benchmark.extra_info["cpu_count"] = os.cpu_count()
    benchmark.extra_info["torch_threads"] = torch.get_num_threads()
    benchmark(lambda: torch.mm(a, b))


def test_torch_conv2d(benchmark):
    """A conv (cellpose's CNN stem) -- exercises the oneDNN/MKLDNN path."""
    x = torch.randn(1, 32, 256, 256)
    w = torch.randn(64, 32, 3, 3)
    benchmark.extra_info["cpu_count"] = os.cpu_count()
    benchmark(lambda: F.conv2d(x, w, padding=1))


def test_torch_attention(benchmark):
    """Scaled-dot-product attention -- the core of the cpsam transformer."""
    q = torch.randn(1, 8, 1024, 64)
    k = torch.randn(1, 8, 1024, 64)
    v = torch.randn(1, 8, 1024, 64)
    benchmark.extra_info["cpu_count"] = os.cpu_count()
    benchmark(lambda: F.scaled_dot_product_attention(q, k, v))
