# Team SPARK – rules for Claude Code

Project: scalable real-time anomaly detection on data-center telemetry (SMD + Alibaba v2018)
with Kafka and Spark Structured Streaming. Report and repo due 15 Nov 2026.

## Frozen contracts (revision B) – never change these without Syam's approval

All names, paths, columns and types live in `contracts/schemas.py` and `contracts/topics.yaml`.
**Always import from `contracts.schemas`; never retype a topic name, path or column name.**

- Versions: Python 3.11, Java 17, PySpark 3.5.1, Kafka 3.7.0 (KRaft, Docker image `apache/kafka:3.7.0`),
  connector `org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1`, confluent-kafka 2.4.0, pyarrow 16.1.0, streamlit 1.36.0.
  PyTorch 2.3.1 CPU build, only for autoencoder work: `pip install -r requirements-ml.txt`.
- Kafka on laptops: `localhost:9092`. Topics: `telemetry.smd`, `telemetry.alibaba`, `anomalies`.
  Message key = `entity_id` (UTF-8). Value = one telemetry record as single-line JSON.
- Telemetry record fields, in this order: `source` (`smd`|`alibaba`), `entity_id`, `event_time`
  (ISO 8601 UTC, ms, ends in `Z`), `metrics` (map string->double), `attrs` (map string->string), `label` (0, 1 or null).
- SMD metric keys `m00`..`m37`. Alibaba metric keys `cpu, mem, mem_gps, mkpi, net_in, net_out, disk_io`.
- SMD has no timestamps: row i of a machine (train rows first, then test rows) gets 2026-01-01T00:00:00Z + i minutes.
  `attrs.split` = `train` or `test`. Train rows have `label` null; test rows 0 or 1.
- Missing metric values are left out of the map, never written as 0.
- Parquet layers under `data/` (git-ignored), partitioned by `source=<source>`:
  `data/raw`, `data/features`, `data/scores`, `data/incidents`, plus `data/reference/smd_interpretation`.
  Column lists are in `contracts/schemas.py` (`LAYERS`).
- `incident_id` = `<entity_id>_<start_time as yyyyMMddHHmm>`.
- Duplicate anomalies are removed using (`entity_id`, `window_start`).
- Detector ladder (each writes its own column in the `scores` layer):
  Tier 1 robust z-score per machine-metric (`tier1_score`, `tier1_flag`);
  Tier 2 MLlib KMeans distance to centroid (`tier2_score`, `tier2_flag`);
  Tier 2b MLlib PCA reconstruction error, linear multi-metric (`pca_score`);
  Tier 3 dense autoencoder reconstruction error, non-linear multi-metric (`ae_score`, `ae_flag`).
  `is_anomaly = ae_flag OR tier1_flag`. `top_metrics` = 3 metrics with the largest autoencoder reconstruction error.
- Models are trained offline (Spark MLlib for KMeans and PCA, PyTorch on one node for the autoencoder) and
  saved to `models/kmeans_<source>/`, `models/pca_<source>/`, `models/ae_<source>/` (`model.pt` + `scaler.json`).
  The streaming job loads them once and scores with `mapInPandas`; models are never trained inside the stream.
  Train only on SMD `attrs.split = train` rows; evaluate only on `test` rows.
- Spark checkpoints go under `checkpoints/<job_name>/`. Trained models go under `models/<name>/` (e.g. `models/kmeans_smd/`); both are git-ignored.

## Team machines: Windows + Git Bash

Everyone works on Windows in Git Bash (not WSL). One-time setup: `bash tools/setup_windows.sh`.
- Activate the venv with `source .venv/Scripts/activate` (not `.venv/bin/activate`); create it with `py -3.11 -m venv .venv`.
- PySpark needs `HADOOP_HOME` with `winutils.exe` and `hadoop.dll` (the setup script installs them in `~/hadoop/bin`)
  and `PYSPARK_PYTHON=python`. If Spark fails with winutils, HADOOP_HOME or "Python worker failed to connect", check these first.
- Kafka runs in Docker Desktop: `docker compose -f infra/docker-compose.yml up -d`.
- Use forward slashes and `pathlib` in code; never hard-code `C:\\`. Keep `git config core.autocrlf input`.

## Coding rules

- Python only (PySpark). Run everything from the repo root.
- Spark sessions must set `spark.sql.session.timeZone=UTC` and
  `spark.sql.parquet.outputTimestampType=TIMESTAMP_MICROS`.
- Every new module gets a tiny test fixture (a few machines, tens of rows) and a pytest in `tests/`.
- Keep each work area in its folder: `src/producer`, `src/streaming`, `src/detection`, `src/incidents`, `src/dashboard`.
- Do not edit `contracts/`, `tools/check_*.py` or this file unless the user says the change is approved.
- Before saying a task is done, run its check script from `tools/` and show the PASS line.

## Checks (inspection gauges)

| Check | Proves |
| --- | --- |
| `python tools/check_contracts.py` | every sample and Parquet layer matches the contract |
| `python tools/check_kafka.py` | Kafka is up, topics are right, records round-trip |
| `python tools/check_spark_kafka.py` | PySpark reads `telemetry.smd` with the frozen schema |
| `python tools/check_smd.py` | SMD converted to the raw layer + interpretation reference |
| `python tools/check_eval.py` | `src/detection/evaluate.py` gives the known answers |
| `python tools/check_dashboard.py` | `src/dashboard/data.py` reads fake scores and incidents |
| `python tools/toy_pipeline.py` | not a check: a toy run of every stage, to compare your output against |
