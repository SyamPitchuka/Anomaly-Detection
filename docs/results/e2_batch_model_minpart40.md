# E2 bottleneck: per-batch time vs executors

Source: `docs/results/e2_cluster_minpart40.json`. Steady full batches only (first batch of each run and partial batches dropped); mean seconds per batch, pooled over the repetitions.

| executors | runs | batches | rows/batch | `addBatch` | `commitOffsets` | `getBatch` | `latestOffset` | `queryPlanning` | `triggerExecution` | `walCommit` | step `count(stateful scoring)` | step `toPandas(test)` | step `write_scores` | step `toPandas(latency)` | step `kafka(anomalies)` | step `write_latency` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 18 | 199,988 | 13.93 | 0.04 | 0.00 | 0.01 | 0.10 | 14.13 | 0.04 | 13.29 | 0.12 | 0.00 | 0.33 | 0.00 | 0.01 |
| 3 | 2 | 18 | 199,988 | 7.19 | 0.04 | 0.00 | 0.01 | 0.10 | 7.38 | 0.04 | 6.48 | 0.13 | 0.00 | 0.37 | 0.00 | 0.01 |
| 5 | 2 | 18 | 199,988 | 6.23 | 0.04 | 0.00 | 0.01 | 0.11 | 6.44 | 0.04 | 5.51 | 0.15 | 0.00 | 0.36 | 0.00 | 0.01 |

Fit T(n) = a + b / n on the per-n means (n = executors); serial fraction = a / (a + b).

| quantity | a (s, serial) | b (s × n, parallel) | serial fraction | R² | points |
| --- | ---: | ---: | ---: | ---: | ---: |
| `durationMs.triggerExecution` | 4.32 | 9.77 | 0.31 | 0.998 | 3 |
| `durationMs.addBatch` | 4.11 | 9.79 | 0.30 | 0.998 | 3 |
| `step.count(stateful scoring)` | 3.37 | 9.89 | 0.25 | 0.998 | 3 |
