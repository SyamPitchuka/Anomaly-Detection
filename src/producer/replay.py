"""Replay the raw layer into Kafka in event_time order.

Run from the repo root:
    python src/producer/replay.py --source smd [--speedup 100] [--limit N]

Reads data/raw/source=<source>/*.parquet (fake.parquet skipped next to real files), merges all machines
by (event_time, entity_id) and sends every record to the contract topic of that source (key = entity_id, value = single-line JSON with the fields
source, entity_id, event_time, metrics, attrs, label). attrs.ingest_ts = send time in epoch ms (string),
used downstream to measure end-to-end latency.

--speedup S  event time runs S times faster than wall time (default 100); 0 = as fast as possible.
--limit N    send only the first N records in event_time order.
Prints records/s every 5 s.

Memory: the files are not loaded. Each sorted file (every SMD machine file) is streamed in batches of 256
rows and the streams are k-way merged, so memory is one small batch per file, and with --limit nothing
past the first N records is read. (Loading the whole SMD raw layer took about 1.2 GB.) A file that is not
in (event_time, entity_id) order is loaded and sorted on its own.
"""
import argparse
import heapq
import itertools
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
SKIP_FILES = {"fake.parquet"}   # same rule as src/detection/autoencoder.py (not imported: it pulls in torch)
REPORT_INTERVAL_S = 5.0
BATCH_ROWS = 256          # rows read per file at a time (kept small: one batch per file is held during the merge)
READ_BUFFER = 1 << 16     # Parquet buffered-stream size
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


def raw_files(root: Path, source: str) -> list[Path]:
    """The Parquet files of data/raw/source=<source> to replay.

    fake.parquet (tools/fake_data.py) is skipped when real files are there, as the batch jobs and
    score_job --input files do: its rows reuse real machine ids and timestamps, so replaying it next to the
    real data put test rows inside real train windows. It is used only when it is the only file."""
    folder = Path(root) / S.PATHS["raw"] / f"source={source}"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files in {folder}")
    return [f for f in files if f.name not in SKIP_FILES] or files


def _event_us(column) -> list:
    return column.cast(pa.timestamp("us", tz="UTC"), safe=False).cast(pa.int64()).to_pylist()


def batch_records(batch):
    """Record batch / table -> (event_us, entity_id, metric pairs, attr pairs, label) per row."""
    yield from zip(_event_us(batch.column("event_time")), batch.column("entity_id").to_pylist(),
                   map_rows(batch.column("metrics")), map_rows(batch.column("attrs")),
                   batch.column("label").to_pylist())


def is_event_sorted(path: Path) -> bool:
    """True if the file's rows are already in (event_time, entity_id) order (reads only those two columns)."""
    t = pq.read_table(path, columns=["event_time", "entity_id"])
    if t.num_rows < 2:
        return True
    et = pa.array(_event_us(t.column("event_time")))
    ent = t.column("entity_id").combine_chunks()
    a, b = et.slice(0, len(et) - 1), et.slice(1)
    later = pc.greater(b, a)
    tie_ok = pc.and_(pc.equal(b, a), pc.greater_equal(ent.slice(1), ent.slice(0, len(ent) - 1)))
    return bool(pc.all(pc.or_(later, tie_ok)).as_py())


def _iter_batches(path: Path, batch_rows: int):
    """Stream a file in small record batches (buffered reads: a column chunk is not loaded whole)."""
    return pq.ParquetFile(path, pre_buffer=False, buffer_size=READ_BUFFER).iter_batches(
        batch_size=batch_rows, columns=COLUMNS)


def file_records(path: Path, batch_rows: int = BATCH_ROWS):
    """One file's records in (event_time, entity_id) order. A sorted file (every SMD machine file) is
    streamed batch by batch, so only what is consumed is read; an unsorted one is loaded and sorted."""
    if is_event_sorted(path):
        for batch in _iter_batches(path, batch_rows):
            yield from batch_records(batch)
    else:
        t = pq.read_table(path, columns=COLUMNS)
        yield from table_records(t, batch_rows)


def table_records(table: pa.Table, batch_rows: int = BATCH_ROWS):
    """An in-memory table's records in (event_time, entity_id) order."""
    t = table.take(event_order(table, None))
    for batch in t.to_batches(max_chunksize=batch_rows):
        yield from batch_records(batch)


def merged_records(files: list[Path], batch_rows: int = BATCH_ROWS):
    """All files' records in global (event_time, entity_id) order: a k-way merge of the per-file streams.
    heapq.merge is lazy and stable (ties keep file order), so this matches sorting the concatenated table,
    and memory stays at one small batch per file however many records are sent."""
    return heapq.merge(*(file_records(f, batch_rows) for f in files), key=lambda r: (r[0], r[1]))


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


def replay(producer, records, source: str, speedup: float = 100.0, limit=None, *,
           now=time.time, monotonic=time.monotonic, sleep=time.sleep, out=print) -> int:
    """Send records to the source topic in event_time order; returns the number of records sent.

    records: an iterator of (event_us, entity_id, metric pairs, attr pairs, label) already in
    (event_time, entity_id) order, e.g. merged_records(raw_files(...)), or a pyarrow Table (sorted here).
    Only the first `limit` records are pulled from the iterator, so nothing beyond them is read."""
    topic = TOPICS[source]
    if isinstance(records, pa.Table):
        records = table_records(records)
    if limit is not None:
        records = itertools.islice(records, max(limit, 0))
    stats = {"errors": 0}

    def on_delivery(err, _msg):
        if err is not None:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                out(f"delivery failed: {err}", flush=True)

    reporter = Reporter(monotonic, out)
    start = monotonic()
    first_us = None
    sent = 0
    for event_us, entity, metrics, attrs, label in records:
        if first_us is None:
            first_us = event_us
        if speedup > 0:
            target = start + (event_us - first_us) / 1_000_000 / speedup
            _wait_until(target, producer, reporter, monotonic, sleep)
        value = build_value(source, entity, event_us, metrics, attrs, label, int(now() * 1000))
        _produce(producer, topic, entity.encode("utf-8"), value, on_delivery)
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
        files = raw_files(args.root, args.source)
    except FileNotFoundError as e:
        print(f"FAIL {e}", file=sys.stderr)
        code = 1
    else:
        if producer is None:
            producer = make_producer(args.bootstrap)
        try:
            replay(producer, merged_records(files), args.source, args.speedup, args.limit)
        except KeyboardInterrupt:
            producer.flush(5)
            print("interrupted", flush=True)
            code = 130
    return code


if __name__ == "__main__":
    sys.exit(main())
