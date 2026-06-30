# cellpose-ci-bench

A controlled experiment to find out **why photon-mosaic-pipeline CI runtime
varies so wildly across runners** (e.g. the same 21-test suite took 1733 s on
ubuntu py3.14 but 8219 s on ubuntu py3.12 — same OS, identical resolved
packages, ~33 s install both times). The blow-up is entirely in the
cellpose-SAM + suite2p CPU compute, and the slow leg changes from run to run.

**Hypothesis:** torch, the BLAS backend, and numba each default to using every
core, so on a 4-core hosted runner you get ~3× thread oversubscription that
thrashes — and *how much* it thrashes depends on how loaded the physical runner
is, which is why it looks random and untied to any one OS/Python.

## The design (why it's trustworthy)

The dominant confound is **which physical runner you landed on**. So the
benchmark compares **control (default threads) vs treatment (pinned threads)
back-to-back in the same process, on the same runner**. If pinning wins *within
a job*, that's causal evidence for the threading mechanism, independent of
runner luck. The workflow then repeats the whole job across
`OS × python × run-index` to sample the runner-luck dimension.

Two workloads, both on CPU:
- `cellpose_eval` — a cpsam forward pass (the real pipeline hot op).
- `blas_matmul` — a pure NumPy matmul (isolates BLAS oversubscription alone).

Each job first prints the **thread environment** (`os.cpu_count()`,
`torch.get_num_threads()`, the BLAS pool size, the `*_NUM_THREADS` vars) so you
can *see* whether oversubscription is even happening.

## Run it

1. Create an empty GitHub repo and push this directory to it (`main`).
2. The `bench` workflow runs on push, or trigger it manually (Actions →
   *bench* → *Run workflow*). Re-run it a few times to collect more samples.
3. Download and aggregate the results:
   ```bash
   gh run download <run-id> -D artifacts
   python analyze.py artifacts
   ```

## Reading the output

- **`within-job effect`** — `default_min / pin1_min`. A ratio consistently `>1`
  means pinning is causally faster on the same hardware → oversubscription is
  real. `≈1` means threads aren't the cause and we look elsewhere
  (runner contention, I/O, snakemake `--cores 1`).
- **`cross-run spread`** — coefficient of variation of the min time across the
  repeated runs, for `default` vs `pin1`. If `pin1` has a much lower CV, pinning
  also *stabilises* the runtime (kills the variance), which is the actual #74
  complaint.

## What a positive result would justify (in photon-mosaic-pipeline)

- Pin thread pools in the test/CI environment
  (`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`,
  `torch.set_num_threads`, `NUMBA_NUM_THREADS`).
- Keep the fixture trim (less compute → less exposure to the effect).
- No need to drop cross-OS coverage.
