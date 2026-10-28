# Scalable real-time anomaly detection on data-center telemetry

Team SPARK: Syamprasad P, Samritha D, Venkatesh P A, Raghav Mishra (IISc).

Trace replay → Kafka → Spark Structured Streaming (features → scoring → incidents) → Parquet → dashboard.

## Quick start (Windows, PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1   # tools check, .venv, requirements, Spark helper files, PASS
.\.venv\Scripts\Activate.ps1         # every new PowerShell window
docker compose -f infra/docker-compose.yml up -d
python infra/create_topics.py
python tools/check_kafka.py          # must print PASS
```

Git Bash users: run `bash tools/setup_windows.sh` and `source .venv/Scripts/activate` instead of the first two lines.

The data contracts are in `contracts/` and summarised in `CLAUDE.md`. Do not change them without approval.
