# cellpose-ci-bench

A controlled experiment to find **why photon-mosaic-pipeline CI runtime is slow
and varies wildly across runners** (the same 21-test suite has swung from ~29 min
to ~137 min on the *same* Ubuntu runner — same OS, identical packages). The
blow-up is entirely in the cellpose-SAM + suite2p CPU compute.

## Answer

Two separate causes, both supported by the data:

- **Slow** — the prebuilt torch package for **Linux/Windows** runs cellpose's CPU
  math ~3× slower than the macOS one. It's the torch build, not the hardware: a
  plain NumPy matmul is equally fast on all three operating systems.
- **Variable** — on CI you share a small machine with other jobs, and they steal
  CPU by a different amount each run. Reproduced locally: adding background CPU
  load alone stretches cellpose from 55 s to 125 s.

Full write-up, figures, and reproduce/stress-test steps: **[ROOT_CAUSE.md](ROOT_CAUSE.md)**.

## Layout

```
tests/test_thread_bench.py        cellpose forward pass + plain NumPy matmul (the compute experiment)
.github/workflows/bench.yml       runs it across every OS × Python version × repeat
plot_rootcause.py                 builds the slowness + variance figures from CI results
stress_test.py / plot_stress.py   imitate a busy CI machine locally
ROOT_CAUSE.md                     the findings
archive/                          retired experiments (thread-pinning, I/O) — ruled out, kept for the record
```

## Recommendation (photon-mosaic-pipeline #74)

Trim the integration fixture (fewer sessions → less cellpose compute). It's the
one change that helps both problems: less time in the slow math, and less time
exposed to other jobs stealing CPU. Don't pin thread counts (it's only ever
slower).
