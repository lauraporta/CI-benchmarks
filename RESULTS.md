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

- _pending first runs_
