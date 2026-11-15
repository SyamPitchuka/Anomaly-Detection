# E2 on the Azure cluster: which file is which

Cluster: 6 Azure VMs (India South Central) in 2 subscriptions joined by VNet peering; `spark-master` (D2as_v4) runs Kafka, NFS and the driver; 5 workers (3 D2as_v4, 2 E2as_v4), 2 vCPU + 4 GB Spark each. Every run is a backlog drain of `telemetry.smd` (32 partitions) by `infra/cluster/drain_test.sh`; each set has `.md` (table), `.png` (plot) and `.json` (every run). `CLUSTER` below is that one-line description passed as `--cluster "..."`.

- `e2_cluster_2w.*`: 2-worker pilot, 1×2 and 2×2, 2,000,000 records, 3 repeats: `python experiments/e2_scaling_sweep.py --configs 1x2 2x2 --repeats 3 --cluster "$CLUSTER"`
- `e2_cluster_5w.*`: strong scaling 1–5 workers (1×2 … 5×2), 2,000,000 records, replicate 4 (112 machines), 3 repeats: `python experiments/e2_scaling_sweep.py --configs 1x2 2x2 3x2 4x2 5x2 --repeats 3 --cluster "$CLUSTER"`
- `e2_cluster_batch800k.*`: 800k-record micro-batches (`maxOffsetsPerTrigger` 800,000), 4,000,000 records, 1×2 vs 5×2, 2 repeats: `python experiments/e2_scaling_sweep.py --configs 1x2 5x2 --repeats 2 --records 4000000 --max-offsets-per-trigger 800000 --cluster "$CLUSTER"`
- `e2_cluster_weak.*`: weak scaling, 28 → 140 machines (replicate = workers), 1,000,000 records per base config, 2 repeats: `python experiments/e2_scaling_sweep.py --weak --configs 1x2 2x2 3x2 4x2 5x2 --repeats 2 --records 1000000 --cluster "$CLUSTER"`
- `e2_drain_runs.csv`: one row per `drain_test.sh` run of all the sets above (and earlier trial runs), appended by `infra/cluster/drain_monitor.py` to `data/scaling/drain_runs.csv`.
- Each sweep writes `e2_cluster.{md,png,json}`; the files were renamed after the run (and the image link in each `.md` fixed to match).
