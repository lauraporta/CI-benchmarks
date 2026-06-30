# Results

_To be filled in once the benchmark matrix has run a few times._

Download and aggregate:

```bash
gh run download <run-id> -D artifacts
python analyze.py artifacts      # tables
python plot.py artifacts         # bench_summary.png
```

## What we're looking for

| Signal | Where | Meaning |
|---|---|---|
| `default/pin1` ratio > 1, **within a job** | `analyze.py` table 1 | pinning is causally faster on the same runner → thread oversubscription is real |
| `pin1` cross-run CV ≪ `default` CV | `analyze.py` table 2 | pinning also stabilises runtime (the #74 variance) |
| `torch.num_threads == cpu_count`, BLAS == cpu_count | job log env banner | oversubscription is actually happening |
| `blas_matmul` shows the same pattern | both | BLAS alone reproduces it (not just torch) |

## Findings

### Local sanity run (1× macOS, 8-core, uncontended) — N=1

| workload | default | pin1 | pin2 |
|---|---|---|---|
| `cellpose_eval` | **54.1 s** | 116.5 s | 74.3 s |
| `blas_matmul` | 50.8 ms | 50.6 ms | 50.6 ms |

On a machine with spare cores and no noisy neighbour, **default (all threads) is
fastest and pinning is up to 2× slower** — parallelism just helps. The BLAS
matmul is unaffected (Apple Accelerate; may differ from OpenBLAS/MKL on the
linux runners).

**Implication:** "blanket-pin to 1 thread" is *not* a safe universal fix — it
would slow down any leg with cores to spare. The CI question is therefore about
**variance under contention**: does `default` swing wildly across runs while
`pin*` stays tight? That's what the matrix runs will show. The likely right fix
is a *moderate* pin (e.g. cores/2) and/or just the fixture trim, not pin1.

### CI matrix

- _pending first runs_
