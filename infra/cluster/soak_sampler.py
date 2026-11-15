"""Per-minute throughput sampler for the cluster soak test (started by soak_test.sh).

Every --interval seconds (default 60) it appends one row to <out>/throughput.csv:

    minute, ts_utc, topic_end_offset_total, produced_rec_per_s, job_batches, job_input_rows, job_rows_per_s

* produced_rec_per_s = growth of the topic's end offsets (summed over partitions) since the last sample, i.e. what
  the broker accepted from the replay. Read straight from Kafka, so it does not depend on the replay's own output.
* job_* = score_job's own query progress: data/latency_stream/progress.jsonl, one JSON line per micro-batch
  (logged_ts_ms, numInputRows, ...). Rows are attributed to the minute in which the batch was logged.
  score_job reads telemetry.smd only, so with --source alibaba these stay 0 (see soak_test.sh).

At the end it prints min / mean / max of the produced and job rates (the first minute is reported but excluded from
"steady" figures because producer start-up and the first Kafka metadata fetch distort it).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HEADER = ["minute", "ts_utc", "topic_end_offset_total", "produced_rec_per_s",
          "job_batches", "job_input_rows", "job_rows_per_s"]


def end_offsets_total(consumer, topic: str) -> int:
    """Sum of the high watermarks of every partition of `topic` (0 if the topic does not exist yet)."""
    from confluent_kafka import TopicPartition

    meta = consumer.list_topics(topic, timeout=10).topics.get(topic)
    total = 0
    if meta is not None and meta.partitions:
        for p in meta.partitions:
            _low, high = consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=10)
            total += high
    return total


def progress_since(path: Path, since_ms: int, until_ms: int) -> tuple[int, int]:
    """(batches, input rows) of progress.jsonl lines logged in (since_ms, until_ms]."""
    batches = rows = 0
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue   # a line still being written
                if since_ms < int(rec.get("logged_ts_ms", 0)) <= until_ms:
                    batches += 1
                    rows += int(rec.get("numInputRows", 0))
    return batches, rows


def summarise(rows: list[dict]) -> list[str]:
    steady = rows[1:] or rows
    out = []
    for key, label in (("produced_rec_per_s", "produced"), ("job_rows_per_s", "score_job")):
        vals = [float(r[key]) for r in steady]
        if vals:
            out.append(f"{label:>10}: min {min(vals):,.0f}  mean {sum(vals) / len(vals):,.0f}  "
                       f"max {max(vals):,.0f} rec/s over {len(vals)} steady minutes")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--progress", type=Path, required=True, help="data/latency_stream/progress.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="directory for throughput.csv")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between samples (tests use less)")
    args = ap.parse_args(argv)

    from confluent_kafka import Consumer

    args.out.mkdir(parents=True, exist_ok=True)
    consumer = Consumer({"bootstrap.servers": args.bootstrap, "group.id": "soak-sampler",
                         "enable.auto.commit": False})
    csv_path = args.out / "throughput.csv"
    results: list[dict] = []
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
            t_prev = time.monotonic()
            ms_prev = int(time.time() * 1000)
            off_prev = end_offsets_total(consumer, args.topic)
            start = t_prev
            for minute in range(1, args.minutes + 1):
                target = start + minute * args.interval
                time.sleep(max(0.0, target - time.monotonic()))
                now, ms_now = time.monotonic(), int(time.time() * 1000)
                off = end_offsets_total(consumer, args.topic)
                dt = max(now - t_prev, 1e-9)
                batches, job_rows = progress_since(args.progress, ms_prev, ms_now)
                row = {"minute": minute,
                       "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "topic_end_offset_total": off,
                       "produced_rec_per_s": round((off - off_prev) / dt, 1),
                       "job_batches": batches, "job_input_rows": job_rows,
                       "job_rows_per_s": round(job_rows / dt, 1)}
                writer.writerow(row)
                f.flush()
                results.append(row)
                print(f"minute {minute:>2}: produced {row['produced_rec_per_s']:>10,.0f} rec/s, "
                      f"score_job {row['job_rows_per_s']:>8,.0f} rows/s ({batches} batches)", flush=True)
                t_prev, ms_prev, off_prev = now, ms_now, off
    finally:
        consumer.close()
    print(f"throughput written to {csv_path}", flush=True)
    for line in summarise(results):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
