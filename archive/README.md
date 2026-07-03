# Archive — retired experiments

These are the intermediate experiments that were run and then **ruled out** as
the cause of the CI slowness/variance. They are kept for provenance; the live
conclusion is in [`../ROOT_CAUSE.md`](../ROOT_CAUSE.md). Nothing here runs in CI
(the workflow was moved out of `.github/workflows/`).

## Thread-pinning / oversubscription — *refuted*
The first hypothesis: torch/BLAS/numba oversubscribe cores on small runners and
thrash. **Wrong** — pinning threads was *slower* on every runner and locally, and
oversubscribing past the core count costs nothing.
- `analyze.py` — within-job `default/pin1` ratio + cross-run CV table
- `plot.py` — the default-vs-pinned bar figure
- `local_sweep.py`, `plot_sweep.py`, `local_sweep.png` — local thread-count sweep
  showing cellpose saturates by ~4 cores and is flat beyond (no oversubscription
  penalty)

_(The compute test `tests/test_thread_bench.py` still measures the `default`
arm — that's what the slowness figure uses — so it stayed in the main repo.)_

## File I/O + snakemake — *ruled out*
Does the file/DAG/orchestration layer carry the variance? **No** — snakemake is a
fixed ~6.5 s overhead independent of `--cores`; raw disk scales with payload but
is trivial next to the compute.
- `test_io_bench.py` — raw file I/O + snakemake DAG benchmark
- `io.yml` — the CI workflow (3 OS × {3.12,3.14} × 5 runs)
- `io_sweep.py` — local sweep over payload size and snakemake `--cores`
- `plot_io.py`, `io_summary.png` — the I/O summary figure
- `IO_VARIANCE.md` — the write-up

## Superseded notes
- `LOCAL_RESULTS.md` — the first, longer local-run log, superseded by the concise
  `../ROOT_CAUSE.md`
