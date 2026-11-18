# E7: HDFS as the checkpoint store

Spark Structured Streaming keeps its offsets, commits and state store in the checkpoint directory. Until now that was `checkpoints/` on the NFS share served by spark-master: one server, one disk. E7 puts the checkpoint on HDFS instead (Apache Hadoop 3.3.6: NameNode on spark-master, DataNodes on spark-master, spark-w2 and spark-w5, `dfs.replication = 2`; `deploy.sh hdfs`) with `submit.sh / drain_test.sh -k hdfs://spark-master:9000/...`. Without `-k` the pipeline is unchanged (NFS).

Measured 2026-11-18 with `experiments/e7_hdfs.sh` (5 executors × 2 cores, `maxOffsetsPerTrigger` 200,000, 32 topic partitions).

## (a) Clean drain, HDFS checkpoints

- 1,416,825 of 1,416,825 records consumed in 8 micro-batches, drain 70.8 s (20,017 records/s); status `drained`.
- score rows written: 141,624; anomalies (`is_anomaly`): 1,158; distinct anomalies on the `anomalies` topic: 1,158.
- checkpoint `hdfs://spark-master:9000/spark-anomaly/checkpoints/e7/e7_run1_clean`.

## (b) Same drain, DataNode on spark-w5 stopped

After micro-batch 3 had committed, `systemctl stop hadoop-datanode` on spark-w5; it stayed down until the drain had finished and fsck had been sampled.

- drain: 1,416,825 of 1,416,825 records, 8 batches, 76.7 s (18,467 records/s); status `drained`.
- vs (a): Parquet rows 141,624 / 141,624, lost 0, extra 0, duplicates 0, value mismatches 0; anomalies 1,158 / 1,158; topic (de-duplicated) lost 0, extra 0, mismatches 0.
- NameNode declared the DataNode dead 90.9 s after the stop; fsck first showed under-replicated blocks at 90.9 s, at most 59 (of 179 blocks), missing blocks at most 0.
- DataNode restarted at 164.3 s (under-replicated then: 4); fsck HEALTHY with 0 under-replicated and all DataNodes live 5.0 s after the restart.

| t after stop (s) | phase | live DataNodes | fsck status | blocks | under-replicated | missing |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| -7.9 | before | 3 | HEALTHY | 75 | 0 | 0 |
| 0.0 | outage | 3 | HEALTHY | 99 | 0 | 0 |
| 90.9 | outage | 2 | HEALTHY | 179 | 59 | 0 |
| 101.1 | outage | 2 | HEALTHY | 179 | 44 | 0 |
| 110.6 | outage | 2 | HEALTHY | 179 | 34 | 0 |
| 120.2 | outage | 2 | HEALTHY | 179 | 22 | 0 |
| 130.0 | outage | 2 | HEALTHY | 179 | 13 | 0 |
| 139.6 | outage | 2 | HEALTHY | 179 | 4 | 0 |
| 169.3 | restarted | 3 | HEALTHY | 179 | 0 | 0 |

- [x] fault run drained
- [x] scores and anomalies identical to (a)
- [x] fsck never reported a missing or corrupt block
- [x] fsck reported under-replicated blocks while the DataNode was down
- [x] after the restart: all DataNodes live, 0 under-replicated, HEALTHY

**(b): PASS**

## (c) Throughput: HDFS vs NFS checkpoints (5 × 2, 2,000,000-record backlog, replicate 4)

| checkpoint store | runs drained | records/s | steady records/s | drain (s) | batch mean (s) | first batch (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HDFS (this run) | 3/3 | 23,646 (22,342–24,690) | 31,368 (29,059–32,873) | 84.7 (81.0–89.5) | 7.70 (7.36–8.14) | 27.2 (26.2–27.7) |
| NFS (E2 baseline, e2_cluster_5w.json) | 3/3 | 24,342 (24,097–24,712) | 32,374 (32,045–32,955) | 82.2 (80.9–83.0) | 7.47 (7.35–7.54) | 26.6 (26.3–26.8) |

- HDFS records/s = 0.971 × the E2 NFS baseline (mean of means).

## Caveats

- The NameNode is still a single point of failure: HDFS removes the single NFS server from the DataNode side (blocks are replicated twice over three VMs), but every checkpoint read and write needs the one NameNode on spark-master. An HA NameNode (two NameNodes, three JournalNodes, ZooKeeper failover) is future work; the cluster has no memory left for it on the 8 GB workers.
- spark-master is also the Spark master, the driver host, the Kafka broker and the NFS server, so losing that VM still stops the pipeline; E7 tests the loss of a storage node, not of the master.
- Only a DataNode process was stopped (systemctl stop), not a VM or a disk, and only one of three; with replication 2 a second simultaneous DataNode loss can make blocks missing.
- The NameNode counts a stopped DataNode's replicas as live until it declares the node dead (2 x recheck interval + 10 heartbeats = 90 s here; Hadoop's default is 10.5 min), so under-replication appears in fsck only from then on, and the NameNode then re-replicates onto the two remaining DataNodes.
- The NFS throughput baseline is the E2 sweep (docs/results/e2_cluster_5w.json) from an earlier day; since then Cassandra (E6) and the HDFS daemons run on spark-w2, spark-w5 and spark-master.

**Result: PASS** ((a) drained; (b) identical output, no block missing, under-replication seen and healed)
