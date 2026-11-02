## Status of this measurement (2026-11-03): incomplete, both 100x attempts stopped by the RAM guard

The plan was three runs of up to 25 minutes each: speed-up 100 (`--limit 56000`), 500 (`--limit 280000`)
and 0 = max (`--limit 400000`). Only the **100x** run was attempted, twice, and both attempts were stopped
by the rule "stop if free RAM < 1 GB":

| attempt | score_job master | stopped at | free RAM before start | free RAM between batches | lowest free RAM |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `local[*]` (12 Python workers) | 70 s | 3.75 GB | ≈ 1.8 GB | 0.83 GB |
| 2 | `local[4]` | 110 s | ≈ 3.7 GB | ≈ 1.8 GB | 0.81 GB |

Free RAM stayed close to 1.8 GB and then fell by about 1 GB once per micro-batch (every 30 s; see
`lag.csv` in each run folder). Lowering the master from `local[*]` to `local[4]` did not change the dip. The
500x and max runs were **not** attempted: their micro-batches are larger, so they would hit the same limit.
The numbers above therefore come from 1 and 2 scored micro-batches only (196 and 560 windows, all from the
first ~35 event-minutes of the train part). Treat them as a first look, not as the result.

## What the partial data already shows

* **Every micro-batch costs about 11 s, whatever it contains.** From score_job's query progress: `addBatch`
  took 11.2 s for batch 0 (**0 input rows**), 10.6 s for 341 rows, 11.1 s for 1,547 rows and 10.5 s for 1,400
  rows. The cost is fixed per batch, not per record. As a result, **processing latency p95 is 11.7–12.6 s,
  just above the 10 s goal even before any wait for the trigger**.
* **Total latency is 28–45 s at 100x.** It is that ~11 s plus up to 30 s waiting for the next trigger
  (records arrive at a steady 47 records/s, and the trigger fires every 30 s) plus the 6.6 s event-time wait
  of the 10-minute watermark. The measured "watermark wait" column (15–34 s) is the 6.6 s plus the trigger
  wait.
* **Lag is a sawtooth.** It grows by 47 records/s × 30 s ≈ 1,400 records between commits and drops at each
  commit, peaking near 2,000. The stream kept up with 100x (the lag did not grow from batch to batch).

## Likely cause of both problems (hypothesis, not verified)

A fixed ~11 s per batch, even with no rows, plus a ~1 GB memory spike once per batch, matches the Python
workers of `applyInPandasWithState` being started again in every micro-batch. Each new worker would import
PyTorch and load the models (`get_models()` caches per process, so a new process reloads everything),
4 workers × a few hundred MB. Ways to check:
1. log `os.getpid()` and the time `get_models()` takes inside `score_group`, and compare across batches;
2. look at whether `spark.python.worker.reuse=true` holds for the stateful pandas operator on Windows.

If this is confirmed and fixed, both the 11 s floor and the 1 GB spike should go away, and the runs can be
repeated with the same command.

## To repeat the runs

Free some RAM first. During these runs WSL (`vmmemWSL`, 1.3 GB) and Edge (about 1.5 GB in total) were open
as well as Docker Desktop. Then:

```
python experiments/e2e_latency_runs.py                                   # 100, 500, max; local[4]
python experiments/e2e_latency_runs.py --runs 100:56000 --trigger "5 seconds" --tag t5   # shorter trigger
python src/streaming/latency_report.py --notes docs/results/e2e_latency_notes.md
```

The p95 < 10 s goal also needs a trigger well below 30 s: with a 30 s trigger, records wait on average
15 s for the next micro-batch.
