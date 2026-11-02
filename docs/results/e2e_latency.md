# End-to-end latency: Kafka -> score_job

Measured on 2026-11-03 with `python experiments/e2e_latency_runs.py`, report by `python src/streaming/latency_report.py`.
Each run: `src/producer/replay.py --source smd --speedup S --limit N` and
`src/streaming/score_job.py --input kafka --reset` started together on an emptied `telemetry.smd` topic
(`OMP_NUM_THREADS=1`). Laptop: i5-13420H (12 threads), 16 GB RAM, Windows 11, Kafka 3.7.0 in Docker Desktop.

| run | speed-up | master | status | records in topic | windows | total p50 / p95 / max (s) | processing p50 / p95 / max (s) | batch run time p50 / max (s) | watermark wait p50 / p95 (s) | wait 11 min / speed-up (s) | peak lag (records) | p95 total < 10 s | p95 processing < 10 s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| speedup_100_local_star | 100x | local[*] | out of memory | 3,204 | 196 | 28.2 / 37.2 / 37.2 | 12.6 / 12.6 / 12.6 | 12.6 / 12.6 | 15.6 / 24.6 | 6.60 | 2,028 | no | no |
| speedup_100_local4 | 100x | local[4] | out of memory | 5,052 | 560 | 31.4 / 44.7 / 45.3 | 11.3 / 11.7 / 11.7 | 11.4 / 12.0 | 20.1 / 33.6 | 6.60 | 1,911 | no | no |

![latency CDF and consumer lag](e2e_latency.png)

## What the columns mean

* **total** = `emit_ts_ms − max_ingest_ts_ms`: from the moment the producer sent the *newest record of a
  window* to the moment the scored window was written (scores file and, for anomalies, the `anomalies` topic).
  This is what a user would call end-to-end latency, and it contains the watermark wait below.
* **processing** = `emit_ts_ms − batch_max_ingest_ts_ms`: from the newest record of that machine in the
  micro-batch that closed the window to the write: the time that record waited in Kafka for the next trigger
  (short while the producer is still sending, since Spark fixes a batch's offsets when it starts; up to one
  trigger interval when the producer stopped before the trigger) plus the micro-batch run time (Kafka read,
  parse, state update, AE + KMeans + PCA scoring, Parquet write, Kafka write) and any queueing behind a
  previous long batch.
* **batch run time** = `triggerExecution` of the micro-batches that had input rows.
* **watermark wait** = `batch_max_ingest_ts_ms − max_ingest_ts_ms` (measured): the 11 minutes of event time
  explained below converted to wall time, *plus* the time the closing records sat in Kafka until the next
  trigger fired (up to one trigger interval, 30 s by default). **11 min / speed-up** is the event-time part
  alone.
* **peak lag** = the largest number of records in `telemetry.smd` that no committed micro-batch had
  processed yet (broker end offsets minus the offsets of the newest committed batch in the checkpoint),
  sampled every 5 s.

## Why the 10-minute watermark is reported separately

A 5-minute window `[s, s+5)` cannot be scored when its last record (minute `s+4`) arrives: with a
10-minute watermark it is complete only once that machine has sent a record with event time `>= s+15`.
That is **11 minutes of event time** of pure waiting, by design (it lets out-of-order records join their
window). How long that is in wall-clock seconds depends only on how fast event time runs:

| replay speed-up | event-time wait in wall seconds (11 min / speed-up) |
| --- | ---: |
| 1x (real time) | 660 |
| 100x | 6.6 |
| 500x | 1.32 |
| max (0) | ≈ 0 (bounded by the producer rate) |

In production (speed-up 1) every window is therefore scored at the earliest 11 minutes after its newest
record, whatever the cluster does; this is a correctness choice, not a performance property. The
proposal's goal **p95 < 10 s** is about the stream, so it should be read against the **processing**
column; the **total** column shows the sum at each replay speed.

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

