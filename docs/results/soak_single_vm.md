# Soak test: score_job on the standalone cluster (single VM), 30 min at max replay speed

**Result: PASS.** 63 micro-batches over 30 min (plus ~3 min of drain), **0 failed batches**, 0 exceptions in the driver log, Kafka lag
bounded at 250-310 k records (about one batch) with no upward trend, and the backlog fully drained at the end. No OOM kills.

Run: 2026-11-14 18:53-19:27 UTC, `soak_test.sh -e 1 -c 2 -d 30` on `spark-master`. Raw per-minute samples: [`soak_single_vm.csv`](soak_single_vm.csv).

## Setup

| | |
| --- | --- |
| Cluster | Spark 3.5.1 standalone, `spark://spark-master:7077`, **single VM** (`VM_PROFILE=vm_8gb_2cpu_single`): master + 1 worker on `spark-master` (Azure `Standard_D2as_v4`, 2 vCPU / 8 GB) |
| Executors | **1 executor x 2 cores**, 1536 MB + 1024 MB overhead |
| Submit | `infra/cluster/submit.sh -e 1 -c 2` -> `spark-submit --deploy-mode client --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 --executor-cores 2 --total-executor-cores 2` |
| Checkpoints | `/srv/spark-shared/Anomaly-Detection/checkpoints/score/source=smd` (shared storage; NFS export of `/srv/spark-shared`) |
| Models | `models/{ae,kmeans,pca}_smd` from the shared repo, checksum-checked on every worker before submit |
| Kafka | 1 broker `kafka-1` (apache/kafka:3.7.0, KRaft), `spark-master:9092`, `telemetry.smd` 8 partitions, RF 1 |
| Load | `replay.py --source smd --speedup 0` (max speed), looped 10x with event times shifted per pass; topic recreated and `--reset` at start |
| Health before the run | `deploy.sh all` -> health check PASS; `check_spark_cluster.py` PASS; `check_kafka_cluster.py` (wraps `tools/check_kafka.py`) PASS |

## Throughput

| | min | mean | max (rec/s, 29 steady minutes) |
| --- | --- | --- | --- |
| produced into `telemetry.smd` | 5,615 | 7,304 | 7,765 |
| consumed by `score_job` | 6,046 | **7,328** | 7,766 |

* 13.07 M records produced, 13.07 M consumed (all of them; the last batches drained the topic to 0).
* 2.61 M windows scored, 10,449 anomalies published to `anomalies`.
* Steady-state batch duration: mean 22.1 s, max 26.5 s (2 batches per minute); state store stays at 28 rows (one per SMD machine).

## Lag (produced - consumed, cumulative)

| minute | 1 | 2 | 5 | 10 | 15 | 20 | 25 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag (records) | 351 k | 263 k | 265 k | 254 k | 277 k | 263 k | 280 k | 308 k |

The lag is about one micro-batch of input (~40 s at 7.3 k rec/s) and does not grow: the job keeps up with the producer.

## Caveats

* **Single VM, not a scaling result.** Kafka, the producer, the driver and the only executor share 2 vCPUs, so the producer (~7.3 k rec/s) is itself
  CPU-bound and the job's ceiling on this box is not measured. E2 / E3 need the 3-VM layout (`spark-w1`, `spark-w2`, see `infra/cluster/AZURE.md`).
* **SMD, not Alibaba.** `score_job.py` consumes `telemetry.smd` only (SMD autoencoder, 38 metrics); replaying Alibaba would load Kafka but give the
  job 0 rows. There is also no Alibaba raw layer on the VM yet.
* The asked-for `submit.sh 4 2` (4 executors x 2 cores = 8 cores) does not fit the 6 vCPU quota in any layout; the 3-VM cluster holds 2 x 2.
