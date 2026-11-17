# E2 bottleneck: per-batch time vs executors

Source: `docs/results/e2_cluster_readonly.json`. Steady full batches only (first batch of each run and partial batches dropped); mean seconds per batch, pooled over the repetitions.

| executors | runs | batches | rows/batch | `addBatch` | `commitOffsets` | `getBatch` | `latestOffset` | `queryPlanning` | `triggerExecution` | `walCommit` | step `count(read only)` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 18 | 199,988 | 6.78 | 0.05 | 0.00 | 0.01 | 0.04 | 6.93 | 0.04 | 6.69 |
| 3 | 2 | 18 | 199,988 | 3.68 | 0.04 | 0.00 | 0.01 | 0.04 | 3.82 | 0.04 | 3.59 |
| 5 | 2 | 18 | 199,988 | 2.93 | 0.05 | 0.00 | 0.01 | 0.04 | 3.08 | 0.04 | 2.83 |

Fit T(n) = a + b / n on the per-n means (n = executors); serial fraction = a / (a + b).

| quantity | a (s, serial) | b (s × n, parallel) | serial fraction | R² | points |
| --- | ---: | ---: | ---: | ---: | ---: |
| `durationMs.triggerExecution` | 2.18 | 4.76 | 0.31 | 0.999 | 3 |
| `durationMs.addBatch` | 2.03 | 4.77 | 0.30 | 0.999 | 3 |
| `step.count(read only)` | 1.93 | 4.77 | 0.29 | 0.999 | 3 |
