"""Kafka / progress-log / Spark-UI side of the E2 backlog-drain test (used by drain_test.sh).

Two sub-commands:

    drain_monitor.py wait-written --bootstrap B --topic T --expect N [--timeout S]
        Wait until the topic's end offsets (summed over partitions) reach N, i.e. every replayed record is
        on the broker. Exit 0 when they do, 1 on timeout. (The topic is recreated first, so offsets start at 0.)

    drain_monitor.py watch --bootstrap B --topic T --progress data/latency_stream/progress.jsonl
                           --out data/scaling --run-id ID [--ui http://localhost:4040] [--job-pid PID]
                           [--since-ms T0] [--timeout S] [--meta key=value ...]
        Read the topic end once (the backlog), then poll score_job's progress log until the consumer offset
        (sum over partitions of the newest batch's sources[].endOffset for the topic) reaches it. Meanwhile
        it polls the driver's Spark UI REST API for the executors. When drained (or on timeout / job exit) it
        writes <out>/<run-id>.json (summary + every batch) and appends one row to <out>/drain_runs.csv.
        Exit 0 if drained, 1 otherwise.

What one run reports (summarise()):
    records          backlog drained (topic end offset total)
    drain_s          start of the first batch with input rows -> end of the last one (progress `timestamp` +
                     durationMs.triggerExecution). Job start-up before the first batch is NOT included; it is
                     in wall_s (job submitted -> drained, from --since-ms).
    records_per_s    records / drain_s
    steady_records_per_s  the same without the first data batch (it also pays the Python workers' first
                     torch import and model load)
    batches, batch_mean_s, batch_p95_s (nearest rank), first_batch_s   over the batches with input rows
    executors_seen   distinct executor ids the UI listed during the run (an executor lost and replaced counts
                     twice); executors_max_active / executor_cores_max = the largest simultaneous count / cores
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CSV_NAME = "drain_runs.csv"
CSV_FIELDS = ["run_id", "ts_utc", "status", "executors", "cores", "total_cores", "max_offsets_per_trigger",
              "replicate", "shuffle_partitions", "records", "consumed", "drain_s", "records_per_s",
              "steady_records_per_s", "batches", "batch_mean_s", "batch_p95_s", "first_batch_s", "wall_s",
              "executors_seen", "executors_max_active", "executor_cores_max"]
POLL_S = 2.0


# ---- pure parsing (tested) ------------------------------------------------------------------------------

def parse_iso_ms(ts: str) -> int:
    """Spark progress timestamp '2026-11-15T10:00:00.123Z' -> epoch ms."""
    dt = datetime.strptime(ts.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")
    return int(round(dt.timestamp() * 1000))


def read_progress(path: Path, since_ms: int = 0) -> list[dict]:
    """progress.jsonl lines logged at or after since_ms, in batchId order (a half-written line is skipped;
    the newest line per batchId wins)."""
    by_id: dict[int, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if "batchId" in rec and int(rec.get("logged_ts_ms", 0)) >= since_ms:
                    by_id[int(rec["batchId"])] = rec
    return [by_id[k] for k in sorted(by_id)]


def topic_offset(batch: dict, topic: str) -> int | None:
    """Sum over partitions of the batch's endOffset for `topic` (None if the batch has no offset for it).
    Spark writes endOffset as a JSON object {topic: {partition: offset}}; a JSON string is accepted too."""
    total, found = 0, False
    for src in batch.get("sources", []):
        end = src.get("endOffset")
        if isinstance(end, str):
            try:
                end = json.loads(end)
            except ValueError:
                continue
        parts = end.get(topic) if isinstance(end, dict) else None
        if isinstance(parts, dict):
            total += sum(int(v) for v in parts.values())
            found = True
    return total if found else None


def consumed_offset(batches: list[dict], topic: str) -> int:
    """Consumer position = the largest topic_offset over the batches (0 if none has one yet)."""
    return max((o for o in (topic_offset(b, topic) for b in batches) if o is not None), default=0)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); NaN for no values."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def batch_duration_s(batch: dict) -> float:
    return float(batch.get("durationMs", {}).get("triggerExecution", 0)) / 1000


def summarise(batches: list[dict], records: int, *, wall_s: float | None = None) -> dict:
    """Drain figures over the batches that had input rows (see the module docstring)."""
    data = [b for b in batches if int(b.get("numInputRows", 0)) > 0]
    out = {"batches": len(data), "drain_s": float("nan"), "records_per_s": float("nan"),
           "steady_records_per_s": float("nan"), "batch_mean_s": float("nan"), "batch_p95_s": float("nan"),
           "first_batch_s": float("nan"), "wall_s": wall_s if wall_s is not None else float("nan"),
           "input_rows": sum(int(b["numInputRows"]) for b in data)}
    if not data:
        return out
    durs = [batch_duration_s(b) for b in data]
    start_ms = parse_iso_ms(data[0]["timestamp"])
    end_ms = max(parse_iso_ms(b["timestamp"]) + batch_duration_s(b) * 1000 for b in data)
    drain_s = (end_ms - start_ms) / 1000
    out.update(drain_s=drain_s, batch_mean_s=sum(durs) / len(durs), batch_p95_s=percentile(durs, 95),
               first_batch_s=durs[0], records_per_s=records / drain_s if drain_s > 0 else float("nan"))
    if len(data) > 1:
        rest_s = drain_s - durs[0]
        rest_rows = sum(int(b["numInputRows"]) for b in data[1:])
        out["steady_records_per_s"] = rest_rows / rest_s if rest_s > 0 else float("nan")
    return out


def parse_executors(executors: list[dict]) -> tuple[set[str], int, int]:
    """Spark REST /applications/<id>/executors -> (ids of the active non-driver executors, how many, cores)."""
    active = [e for e in executors if e.get("id") != "driver" and e.get("isActive", True)]
    return {str(e["id"]) for e in active}, len(active), sum(int(e.get("totalCores", 0)) for e in active)


class ExecutorTracker:
    """Accumulates parse_executors() results over the run."""

    def __init__(self):
        self.ids: set[str] = set()
        self.max_active = 0
        self.max_cores = 0
        self.polled = False

    def add(self, executors: list[dict]) -> None:
        ids, n, cores = parse_executors(executors)
        self.polled = True
        self.ids |= ids
        self.max_active = max(self.max_active, n)
        self.max_cores = max(self.max_cores, cores)

    def result(self) -> dict:
        if not self.polled:
            return {"executors_seen": None, "executors_max_active": None, "executor_cores_max": None}
        return {"executors_seen": len(self.ids), "executors_max_active": self.max_active,
                "executor_cores_max": self.max_cores}


def parse_meta(pairs: list[str]) -> dict:
    """['executors=2', 'cores=4', 'note=x'] -> {'executors': 2, 'cores': 4, 'note': 'x'} (ints where they parse)."""
    meta = {}
    for p in pairs:
        key, sep, value = p.partition("=")
        if not sep or not key:
            raise ValueError(f"--meta wants key=value, got {p!r}")
        try:
            meta[key] = int(value)
        except ValueError:
            meta[key] = None if value == "" else value
    return meta


def csv_row(result: dict) -> dict:
    """The CSV_FIELDS of a run result (floats rounded; NaN / None -> empty cell)."""
    row = {}
    for k in CSV_FIELDS:
        v = result.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = ""
        elif isinstance(v, float):
            v = round(v, 3)
        row[k] = v
    return row


def _no_nan(v):
    """NaN floats -> None, recursively (JSON has no NaN)."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: _no_nan(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_no_nan(x) for x in v]
    return v


def write_result(result: dict, out_dir: Path) -> tuple[Path, Path]:
    """<out>/<run_id>.json (NaN -> null) and one appended row of <out>/drain_runs.csv (header if new)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"{result['run_id']}.json"
    js.write_text(json.dumps(_no_nan(result), indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    cs = out_dir / CSV_NAME
    new = not cs.exists() or cs.stat().st_size == 0
    with cs.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(csv_row(result))
    return js, cs


def build_result(run_id: str, status: str, meta: dict, records: int, batches: list[dict], topic: str,
                 tracker: ExecutorTracker, wall_s: float | None) -> dict:
    total_cores = (meta.get("executors") or 0) * (meta.get("cores") or 0) or None
    detail = [{"batch": b.get("batchId"), "timestamp": b.get("timestamp"), "rows": int(b.get("numInputRows", 0)),
               "duration_s": batch_duration_s(b), "offset": topic_offset(b, topic)} for b in batches]
    return {"run_id": run_id, "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status, "topic": topic, "total_cores": total_cores, **meta, "records": records,
            "consumed": consumed_offset(batches, topic), **summarise(batches, records, wall_s=wall_s),
            **tracker.result(), "batches_detail": detail}


# ---- I/O (cluster only) ---------------------------------------------------------------------------------

def end_offsets_total(consumer, topic: str) -> int:
    """Sum of the high watermarks of every partition of `topic` (0 if it does not exist yet)."""
    from confluent_kafka import TopicPartition

    meta = consumer.list_topics(topic, timeout=10).topics.get(topic)
    total = 0
    if meta is not None and meta.partitions:
        for p in meta.partitions:
            _low, high = consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=10)
            total += high
    return total


def make_consumer(bootstrap: str):
    from confluent_kafka import Consumer

    return Consumer({"bootstrap.servers": bootstrap, "group.id": "drain-monitor", "enable.auto.commit": False})


def fetch_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (driver UI on the master)
        return json.loads(r.read().decode("utf-8"))


def poll_executors(ui: str) -> list[dict] | None:
    """Executors of the driver's (only) application, or None when the UI is not reachable yet."""
    try:
        apps = fetch_json(f"{ui.rstrip('/')}/api/v1/applications")
        if not apps:
            return None
        return fetch_json(f"{ui.rstrip('/')}/api/v1/applications/{apps[0]['id']}/executors")
    except (OSError, ValueError, KeyError):
        return None


def pid_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def cmd_wait_written(args) -> int:
    consumer = make_consumer(args.bootstrap)
    deadline = time.monotonic() + args.timeout
    try:
        while True:
            total = end_offsets_total(consumer, args.topic)
            if total >= args.expect:
                print(f"{args.topic}: {total:,} records on the broker (expected {args.expect:,})", flush=True)
                return 0
            if time.monotonic() > deadline:
                print(f"FAIL {args.topic}: only {total:,} of {args.expect:,} records after {args.timeout:g} s",
                      flush=True)
                return 1
            time.sleep(1)
    finally:
        consumer.close()


def cmd_watch(args) -> int:
    meta = parse_meta(args.meta)
    consumer = make_consumer(args.bootstrap)
    try:
        records = end_offsets_total(consumer, args.topic)
    finally:
        consumer.close()
    print(f"backlog: {records:,} records in {args.topic}", flush=True)
    tracker = ExecutorTracker()
    since_ms = args.since_ms or 0
    deadline = time.monotonic() + args.timeout
    last_print = 0.0
    status = "timeout"
    batches: list[dict] = []
    while True:
        execs = poll_executors(args.ui)
        if execs is not None:
            tracker.add(execs)
        batches = read_progress(args.progress, since_ms)
        consumed = consumed_offset(batches, args.topic)
        if records and consumed >= records:
            status = "drained"
            break
        if not pid_alive(args.job_pid):
            status = "job_exited"
            break
        if time.monotonic() > deadline:
            break
        if time.monotonic() - last_print >= 15:
            pct = 100 * consumed / records if records else 0
            print(f"  consumed {consumed:,} / {records:,} ({pct:.0f}%), {len(batches)} batches, "
                  f"executors {tracker.max_active}", flush=True)
            last_print = time.monotonic()
        time.sleep(POLL_S)
    wall_s = (time.time() * 1000 - since_ms) / 1000 if since_ms else None
    result = build_result(args.run_id, status, meta, records, batches, args.topic, tracker, wall_s)
    js, cs = write_result(result, args.out)
    print(f"{status}: {result['consumed']:,} / {records:,} records, drain {result['drain_s']:.1f} s, "
          f"{result['records_per_s']:,.0f} records/s, {result['batches']} batches "
          f"(mean {result['batch_mean_s']:.2f} s, p95 {result['batch_p95_s']:.2f} s), "
          f"executors seen {result['executors_seen']}", flush=True)
    print(f"written {js} and a row of {cs}", flush=True)
    return 0 if status == "drained" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wait-written", help="wait until the topic holds --expect records")
    w.add_argument("--bootstrap", required=True)
    w.add_argument("--topic", required=True)
    w.add_argument("--expect", type=int, required=True)
    w.add_argument("--timeout", type=float, default=300.0)
    v = sub.add_parser("watch", help="poll the progress log until the backlog is drained, then write the result")
    v.add_argument("--bootstrap", required=True)
    v.add_argument("--topic", required=True)
    v.add_argument("--progress", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--run-id", required=True)
    v.add_argument("--ui", default="http://localhost:4040", help="driver Spark UI (REST API for the executors)")
    v.add_argument("--job-pid", type=int, default=None, help="stop (status job_exited) if this process ends")
    v.add_argument("--since-ms", type=int, default=None, help="job submit time; older progress lines are ignored")
    v.add_argument("--timeout", type=float, default=3600.0)
    v.add_argument("--meta", nargs="*", default=[], help="key=value pairs stored in the result")
    args = ap.parse_args(argv)
    return cmd_wait_written(args) if args.cmd == "wait-written" else cmd_watch(args)


if __name__ == "__main__":
    sys.exit(main())
