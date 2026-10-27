# Scalable real-time anomaly detection on data-center telemetry

Team SPARK: Syamprasad P, Samritha D, Venkatesh P A, Raghav Mishra (IISc).

Trace replay → Kafka → Spark Structured Streaming (features → scoring → incidents) → Parquet → dashboard.

## Quick start (Windows, Git Bash)

```bash
bash tools/setup_windows.sh          # tools check, .venv, requirements, Spark helper files, PASS
source .venv/Scripts/activate        # every new Git Bash window
docker compose -f infra/docker-compose.yml up -d
python infra/create_topics.py
python tools/check_kafka.py          # must print PASS
```

The data contracts are in `contracts/` and summarised in `CLAUDE.md`. Do not change them without approval.
