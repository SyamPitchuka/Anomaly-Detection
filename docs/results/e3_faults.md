# E3 – fault tolerance: does the shared checkpoint give exactly-once output?

Result: **PASS.** After one executor kill and one driver kill with a restart on a different VM, the stream output
(Parquet `data/scores_stream`) and the `anomalies` topic are identical to a clean run on the same input.
Lost rows 0, duplicate rows 0, value mismatches 0. Data: [e3_faults.json](e3_faults.json).

## Setup
- Cluster: spark-master (Kafka, NFS, first driver) + spark-w1..w5, 5 executors x 2 cores, checkpoint on the NFS share.
- Input: the whole SMD set, 1,416,825 records, replayed at a steady rate (`--speedup 3000`, about 1,400 records/s,
  about 17 min). The same records go into a clean run and a fault run. Train rows come first and write no Parquet, so
  faults are only injected once test windows are being written.
- score_job: trigger 5 s, at most 20,000 records per micro-batch.
- Script: `experiments/e3_faults.sh` (branch `samritha/e3-faults`), run as
  `bash experiments/e3_faults.sh`. Run id `e3_full2`.

## Faults
| step | what | when |
| --- | --- | --- |
| 1 | `kill -9` one executor JVM on spark-w1 | batch 96 in flight (checkpoint `offsets/96` written, `commits/96` not yet; last commit 95) |
| 2 | `kill -9` the driver on spark-master, start a new driver on spark-w1 with the same checkpoint, no `--reset` | batch 101 in flight (last commit 100) |

## Recovery time
| fault | measure | seconds |
| --- | --- | ---: |
| executor kill | kill -> commit of the batch that was in flight (batch 96) | 15.7 |
| driver kill, restart on spark-w1 | kill -> `score_job started` on the new driver | 27.3 |
| driver kill, restart on spark-w1 | kill -> commit of the batch that was in flight (batch 101, replayed by the new driver) | 46.1 |

Single measurements, taken by polling the checkpoint `commits/` directory every 0.2 s. The driver numbers include
`spark-submit` start-up, executor registration and model loading on the new VM.

## Exactly-once check (fault run vs clean run, key = entity_id + window_start)
| layer | clean rows | fault rows | duplicates in fault run | lost | extra | value mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parquet (`scores_stream`) | 141,624 | 141,624 | 0 | 0 | 0 | 0 |
| topic `anomalies`, de-duplicated | 1,158 | 1,158 | 0 | 0 | 0 | 0 |

- The `anomalies` topic had 1,158 records and 1,158 distinct keys, so the replayed batch did not publish twice here. The
  Kafka sink is at-least-once in general; consumers de-duplicate on (entity_id, window_start).
- Value mismatch: `ae_score` differs by more than 1e-6 or `is_anomaly` differs for a key present in both runs.
- The fault run wrote 126 score files and the clean run 136: both runs replay in real time, so micro-batch boundaries differ.
  The rows are the same.

## Caveats
- One run, one executor kill and one driver kill. Recovery times are single measurements; repeat the run for error bars.
- The comparison is only valid because both runs got the same records. If a window were late in one run only, it would
  show up as lost or extra (see `lost_sample` in the JSON).
- Kafka (one broker) and the NFS share are not failed in this test; the first driver and Kafka share the master VM.
