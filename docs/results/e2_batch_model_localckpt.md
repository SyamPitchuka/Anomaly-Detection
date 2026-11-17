# E2 bottleneck: per-batch time vs executors

Source: `docs/results/e2_cluster_localckpt.json`. Steady full batches only (first batch of each run and partial batches dropped); mean seconds per batch, pooled over the repetitions.

| executors | runs | batches | rows/batch | `addBatch` | `commitOffsets` | `getBatch` | `latestOffset` | `queryPlanning` | `triggerExecution` | `walCommit` | step `count(stateful scoring)` | step `toPandas(test)` | step `write_scores` | step `toPandas(latency)` | step `kafka(anomalies)` | step `write_latency` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 18 | 199,988 | 13.82 | 0.04 | 0.00 | 0.01 | 0.10 | 14.01 | 0.05 | 13.16 | 0.11 | 0.00 | 0.34 | 0.00 | 0.01 |
| 3 | 2 | 18 | 199,988 | 7.49 | 0.04 | 0.00 | 0.01 | 0.10 | 7.69 | 0.04 | 6.82 | 0.13 | 0.00 | 0.34 | 0.00 | 0.01 |
| 5 | 2 | 18 | 199,988 | 5.77 | 0.04 | 0.00 | 0.01 | 0.10 | 5.97 | 0.05 | 5.04 | 0.14 | 0.00 | 0.39 | 0.00 | 0.01 |

Fit T(n) = a + b / n on the per-n means (n = executors); serial fraction = a / (a + b).

| quantity | a (s, serial) | b (s × n, parallel) | serial fraction | R² | points |
| --- | ---: | ---: | ---: | ---: | ---: |
| `durationMs.triggerExecution` | 4.18 | 9.87 | 0.30 | 0.998 | 3 |
| `durationMs.addBatch` | 3.98 | 9.87 | 0.29 | 0.998 | 3 |
| `step.count(stateful scoring)` | 3.26 | 9.95 | 0.25 | 0.997 | 3 |
