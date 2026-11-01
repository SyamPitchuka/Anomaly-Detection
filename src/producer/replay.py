"""Replay the raw layer into Kafka in event_time order.

Run from the repo root:
    python src/producer/replay.py --source smd [--speedup 100] [--limit N]

Reads data/raw/source=<source>/*.parquet, sorts all machines together by event_time and sends every
record to the contract topic of that source (key = entity_id, value = single-line JSON with the fields
source, entity_id, event_time, metrics, attrs, label). attrs.ingest_ts = send time in epoch ms (string),
used downstream to measure end-to-end latency.

--speedup S  event time runs S times faster than wall time (default 100); 0 = as fast as possible.
--limit N    send only the first N records in event_time order.
Prints records/s every 5 s.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402

TOPICS = {"smd": S.TOPIC_SMD, "alibaba": S.TOPIC_ALIBABA}
COLUMNS = ["entity_id", "event_time", "metrics", "attrs", "label"]
REPORT_INTERVAL_S = 5.0
CHUNK_ROWS = 20000
POLL_EVERY = 100
MAX_SLEEP_S = 0.5

PRODUCER_CONF = {
    "bootstrap.servers": S.KAFKA_BOOTSTRAP,
    "linger.ms": 20,
    "batch.num.messages": 10000,
    "queue.buffering.max.messages": 200000,
    "compression.type": "lz4",
}


@lru_cache(maxsize=4096)
def format_event_time(event_us: int) -> str:
    """Epoch microseconds -> ISO 8601 UTC with ms precision, ending in Z."""
    dt = datetime.fromtimestamp(event_us // 1_000_000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{(event_us // 1000) % 1000:03d}Z"


def build_value(source: str, entity_id: str, event_us: int, metrics, attrs, label, ingest_ts_ms: int) -> bytes:
    """One telemetry record as single-line UTF-8 JSON in the contract field order."""
    metric_map = {k: v for k, v in (metrics or []) if v is not None and math.isfinite(v)}
    attr_map = {k: v for k, v in (attrs or [])}
    attr_map["ingest_ts"] = str(ingest_ts_ms)
    record = {
        "source": source,
        "entity_id": entity_id,
        "event_time": format_event_time(event_us),
        "metrics": metric_map,
        "attrs": attr_map,
        "label": None if label is None else int(label),
    }
    return json.dumps(record, separators=(",", ":")).encode("utf-8")


def load_raw(root: Path, source: str) -> pa.Table:
    """Read all Parquet files of data/raw/source=<source> into one table."""
    folder = Path(root) / S.PATHS["raw"] / f"source={source}"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files in {folder}")
    return pa.concat_tables([pq.read_table(f, columns=COLUMNS) for f in files])


def map_rows(column) -> list:
    """Map column -> list of [(key, value), ...] per row; much faster than MapArray.to_pylist()."""
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    offsets = arr.offsets.to_pylist()
    keys = arr.keys.to_pylist()
    items = arr.items.to_pylist()
    valid = arr.is_valid().to_pylist() if arr.null_count else None
    rows = []
    for i in range(len(arr)):
        if valid is not None and not valid[i]:
            rows.append([])
        else:
            lo, hi = offsets[i], offsets[i + 1]
            rows.append(list(zip(keys[lo:hi], items[lo:hi])))
    return rows


def event_order(table: pa.Table, limit):
    """Row indices sorted by (event_time, entity_id), cut to the first `limit` rows."""
    idx = pc.sort_indices(table, sort_keys=[("event_time", "ascending"), ("entity_id", "ascending")])
    if limit is not None:
        idx = idx.slice(0, max(limit, 0))
    return idx


class Reporter:
    """Prints records/s every REPORT_INTERVAL_S seconds."""

    def __init__(self, monotonic, out=print):
        self.monotonic = monotonic
        self.out = out
        self.count = 0
        self.last_count = 0
        self.last_time = monotonic()

    def tick(self) -> None:
        now = self.monotonic()
        elapsed = now - self.last_time
        if elapsed >= REPORT_INTERVAL_S:
            rate = (self.count - self.last_count) / elapsed
            self.out(f"{rate:,.0f} records/s (total {self.count:,})", flush=True)
            self.last_count = self.count
            self.last_time = now


def _produce(producer, topic: str, key: bytes, value: bytes, on_delivery) -> None:
    """produce() with back-pressure: when the local queue is full, serve callbacks and retry."""
    sent = False
    while not sent:
        try:
            producer.produce(topic, key=key, value=value, on_delivery=on_delivery)
            sent = True
        except BufferError:
            producer.poll(0.1)


def _wait_until(target, producer, reporter, monotonic, sleep) -> None:
    """Sleep (in short slices, keeping callbacks and the rate report alive) until monotonic() >= target."""
    remaining = target - monotonic()
    while remaining > 0:
        producer.poll(0)
        reporter.tick()
        sleep(min(remaining, MAX_SLEEP_S))
        remaining = target - monotonic()


def replay(producer, table: pa.Table, source: str, speedup: float = 100.0, limit=None, *,
           now=time.time, monotonic=time.monotonic, sleep=time.sleep, out=print) -> int:
    """Send the table's rows to the source topic in event_time order; returns the number of records sent."""
    topic = TOPICS[source]
    idx = event_order(table, limit)
    total = len(idx)
    stats = {"errors": 0}

    def on_delivery(err, _msg):
        if err is not None:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                out(f"delivery failed: {err}", flush=True)

    table = table.combine_chunks()
    reporter = Reporter(monotonic, out)
    start = monotonic()
    first_us = None
    sent = 0
    for begin in range(0, total, CHUNK_ROWS):
        chunk = table.take(idx.slice(begin, CHUNK_ROWS))
        times = chunk["event_time"].cast(pa.timestamp("us", tz="UTC"), safe=False).cast(pa.int64()).to_pylist()
        entities = chunk["entity_id"].to_pylist()
        metrics = map_rows(chunk["metrics"])
        attrs = map_rows(chunk["attrs"])
        labels = chunk["label"].to_pylist()
        for i, event_us in enumerate(times):
            if first_us is None:
                first_us = event_us
            if speedup > 0:
                target = start + (event_us - first_us) / 1_000_000 / speedup
                _wait_until(target, producer, reporter, monotonic, sleep)
            value = build_value(source, entities[i], event_us, metrics[i], attrs[i], labels[i], int(now() * 1000))
            _produce(producer, topic, entities[i].encode("utf-8"), value, on_delivery)
            sent += 1
            reporter.count = sent
            if sent % POLL_EVERY == 0:
                producer.poll(0)
            reporter.tick()

    producer.flush()
    elapsed = max(monotonic() - start, 1e-9)
    out(f"done: {sent:,} records to {topic} in {elapsed:.1f} s ({sent / elapsed:,.0f} records/s), "
        f"{stats['errors']} delivery errors", flush=True)
    return sent


def make_producer(bootstrap: str):
    from confluent_kafka import Producer

    return Producer({**PRODUCER_CONF, "bootstrap.servers": bootstrap})


def main(argv=None, producer=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=S.SOURCES, default="smd")
    ap.add_argument("--speedup", type=float, default=100.0, help="event-time speedup; 0 = as fast as possible")
    ap.add_argument("--limit", type=int, default=None, help="send only the first N records")
    ap.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/raw")
    ap.add_argument("--bootstrap", default=S.KAFKA_BOOTSTRAP)
    args = ap.parse_args(argv)

    code = 0
    try:
        table = load_raw(args.root, args.source)
    except FileNotFoundError as e:
        print(f"FAIL {e}", file=sys.stderr)
        code = 1
    else:
        if producer is None:
            producer = make_producer(args.bootstrap)
        try:
            replay(producer, table, args.source, args.speedup, args.limit)
        except KeyboardInterrupt:
            producer.flush(5)
            print("interrupted", flush=True)
            code = 130
    return code


if __name__ == "__main__":
    sys.exit(main())
