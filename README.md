# Scalable Real-Time Anomaly Detection on Data-Centre Telemetry

**Team SPARK** (DA 231o Data Engineering at Scale, IISc Bengaluru, Prof. Yogesh Simmhan):
Syamprasad P, Samritha D, Venkatesh P, Raghav Mishra.

A streaming system that scores every server's five-minute window of telemetry as it arrives, names the
metrics at fault, groups alarms into incidents and analyses incidents as a co-failure graph. It runs on a
six-VM Azure cluster with Kafka, Spark Structured Streaming, HDFS, Cassandra, GraphFrames and Neo4j.

```
Raw Parquet (SMD, Alibaba) -> replay producer (event time, key = machine) -> Kafka (keyed partitions)
  -> stateful Structured Streaming job (applyInPandasWithState, per-machine 5-min windows and watermark;
     Tier 1 z-score, KMeans, PCA, autoencoder; checkpoints on NFS or HDFS)
  -> scores layer (Parquet) + Kafka `anomalies` + Cassandra
  -> incidents (top-3 metrics) -> co-failure graph -> GraphFrames Pregel (groups, roots) and Neo4j GDS (Cypher)
```

## Headline results

| Question | Result | Evidence |
| --- | --- | --- |
| Accuracy within the alarm budget (SMD) | Autoencoder: composite F1 0.523, **1.18 false alarms / machine / day** (KMeans 3.46, PCA 3.50, Tier 1 215) | [e1.md](docs/results/e1.md) |
| Effect of the causal drift fix | Composite F1 0.236 -> 0.525, false alarms 36.98 -> 1.18 | [ae_ablation.md](docs/ae_ablation.md) |
| Robustness | 3 seeds: F1 0.516 ± 0.002; held-out threshold: 0.61 false alarms / day | [e1_robustness.md](docs/results/e1_robustness.md) |
| Stream = batch | Max score difference 4.1e-6 (autoencoder), 5.8e-11 (Tier 1); 100% of flags identical | [tier1_stream.md](docs/results/tier1_stream.md) |
| Latency (Linux VM) | Processing p95 3.5 s at 100x, 2.3 s at 500x replay; 10.4 s at maximum speed | [e2e_latency_linux.md](docs/results/e2e_latency_linux.md) |
| Strong scaling (6 VMs) | 11,823 -> 24,342 records/s, **2.06x on 5 workers**; batch time fits T(n) = 3.6 + 9.4/n s | [e2_cluster_5w.md](docs/results/e2_cluster_5w.md) |
| Larger micro-batches / weak scaling | 41,230 records/s (800k batches); weak-scaling efficiency 56% | [batch800k](docs/results/e2_cluster_batch800k.md), [weak](docs/results/e2_cluster_weak.md) |
| Where the ceiling sits | Fixed cost per task and micro-batch; not the broker, driver or NFS | [readonly](docs/results/e2_batch_model_readonly.md), [localckpt](docs/results/e2_batch_model_localckpt.md), [minpart40](docs/results/e2_batch_model_minpart40.md) |
| Fault tolerance | Executor kill and driver restart: output identical to a clean run (141,624 rows, 1,158 anomalies) | [e3_faults.md](docs/results/e3_faults.md) |
| HDFS checkpoints | DataNode lost mid-run: identical output, healthy 5 s after restart | [e7_hdfs.md](docs/results/e7_hdfs.md) |
| Cassandra and CAP | One node down: ONE stays available, QUORUM and ALL refuse; replicas converge | [e6_cap.md](docs/results/e6_cap.md) |
| Culprit metrics | A true culprit is in the incident's top 3 in 94.5% of covered faults (random 53.7%) | [attribution.md](docs/results/attribution.md) |
| Second dataset (Alibaba) | Autoencoder catches more slow drifts than Tier 1 (0.60 vs 0.47), 3x sooner | [e5.md](docs/results/e5.md), [e5b_alibaba_ae.md](docs/results/e5b_alibaba_ae.md) |
| Graph analytics (E8, E8c) | Pregel = Neo4j on every graph; Alibaba: 90 co-failure edges vs 47 by chance; injected cascades: group F1 0.91-0.95, root accuracy 0.50-0.67 | [e8_graph.md](docs/results/e8_graph.md) |

Every number above comes from a committed result file produced by a script in this repository.

## Repository layout

| Folder | What is in it |
| --- | --- |
| `contracts/` | Frozen schemas for every layer and Kafka message (checked by `tools/check_contracts.py`) |
| `src/producer/` | SMD and Alibaba conversion, replay into Kafka, fault and cascade injection |
| `src/detection/` | Tier 1, KMeans, PCA, autoencoder, shared evaluation (`evaluate.py`) |
| `src/streaming/` | Stateful scoring job (`score_job.py`), latency report |
| `src/incidents/` | Incident grouping and culprit-metric evaluation |
| `src/serving/` | Cassandra store |
| `src/dashboard/` | Streamlit dashboard of scores and incidents |
| `infra/cluster/` | Azure cluster scripts: deploy, Kafka, Spark, NFS, HDFS, Cassandra, health check (`AZURE.md`) |
| `experiments/` | One script per experiment (E1 to E8c) |
| `docs/results/` | Result tables (`.md`), raw numbers (`.json`) and figures (`.png`) for every experiment |
| `tests/` | Unit tests (`pytest`) |

## Experiments and how to reproduce them

All commands run from the repository root (Python 3.11). Cluster commands run on the master VM.

| Experiment | Command | Result |
| --- | --- | --- |
| SMD raw layer | `python src/detection/smd_to_parquet.py` | `data/raw` |
| Detectors | `python src/detection/autoencoder.py --inputs mean_max --recenter 24 --adaptive 288`; `python src/detection/baselines.py --recenter 24`; `python src/detection/tier1_zscore.py` | `models/` |
| E1 accuracy | `python src/detection/e1_report.py`; `python experiments/e1_robustness.py` | `e1.md`, `e1_robustness.md` |
| Parity | `python tools/check_stream_parity.py`; `python experiments/baselines_parity.py` | `tier1_stream.md` |
| Latency | `python experiments/e2e_latency_runs.py --master local[2] --trigger "5 seconds"` | `e2e_latency_linux.md` |
| Cluster | `bash infra/cluster/deploy.sh all` (guide: `infra/cluster/AZURE.md`) | |
| E2 scaling | `python experiments/e2_scaling_sweep.py --configs 1x2 2x2 3x2 4x2 5x2 --repeats 3` (`--weak`, `--max-offsets-per-trigger 800000` for the variants); `python experiments/e2_batch_model.py` | `e2_cluster_*.md` |
| E3 faults | `bash experiments/e3_faults.sh` | `e3_faults.md` |
| E4 windows | `python experiments/e4_windows.py` | `e4.md` |
| E5, E5b Alibaba | `python src/producer/inject_faults.py --machines 40 --per-type 10 --hours 48 --clean-hours 24 --seed 7 --threshold 20`; `python experiments/e5b_alibaba_ae.py` | `e5.md`, `e5b_alibaba_ae.md` |
| E6 Cassandra | `bash infra/cluster/deploy.sh cassandra`; `python experiments/e6_cap.py --hosts spark-w2,spark-w5` | `e6_cap.md` |
| E7 HDFS | `bash infra/cluster/deploy.sh hdfs`; `bash experiments/e7_hdfs.sh --run-id e7_run1` | `e7_hdfs.md` |
| Incidents | `python src/incidents/group.py`; `python src/incidents/attribution_eval.py` | `attribution.md` |
| E8 graphs | `python experiments/e8_graph.py pregel --source smd` (and `--source alibaba` after `e8_alibaba_input.py`); `docker compose -f infra/docker-compose.neo4j.yml up -d`; `python experiments/e8_neo4j.py --source smd` | `e8_graph.md` |
| E8c cascades | `python experiments/e8_cascade.py stream|pregel|neo4j`; `python experiments/e8_onset.py stream|pregel|neo4j`; `python experiments/e8_graph.py report` | `e8_graph.md` |

## Local quick start (Windows, PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1   # tools check, .venv, requirements, Spark helper files, PASS
.\.venv\Scripts\Activate.ps1         # every new PowerShell window
docker compose -f infra/docker-compose.yml up -d
python infra/create_topics.py
python tools/check_kafka.py          # must print PASS
```

Git Bash users: run `bash tools/setup_windows.sh` and `source .venv/Scripts/activate` instead of the first two lines.

## Data

- **SMD** (Server Machine Dataset, OmniAnomaly, KDD 2019): 28 machines, 38 metrics, 1,416,825 rows, official
  time-ordered 50/50 train/test split, 327 labelled fault segments with culprit metrics.
- **Alibaba cluster trace v2018** (`machine_usage`): 40 machines, 2,468,863 readings over 8 days, no labels;
  faults and cascades are injected by `src/producer/inject_faults.py`.

The data contracts are in `contracts/` and summarised in `CLAUDE.md`.
