import json
import time
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from src.producer import replay as R

BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
MINUTE_US = 60_000_000
FIELDS = ["source", "entity_id", "event_time", "metrics", "attrs", "label"]


class MockProducer:
    def __init__(self, buffer_errors=0):
        self.messages = []
        self.flushed = False
        self.buffer_errors = buffer_errors
        self.polls = 0

    def produce(self, topic, key=None, value=None, on_delivery=None):
        if self.buffer_errors > 0:
            self.buffer_errors -= 1
            raise BufferError("queue full")
        self.messages.append((topic, key, value))
        if on_delivery is not None:
            on_delivery(None, None)

    def poll(self, timeout=0):
        self.polls += 1
        return 0

    def flush(self, timeout=None):
        self.flushed = True
        return 0


def _raw_table(n=10) -> pa.Table:
    """n fake rows over two machines, deliberately stored out of event_time order."""
    order = list(reversed(range(n)))
    entities = [f"machine-1-{i % 2 + 1}" for i in order]
    times = pa.array([BASE_US + i * MINUTE_US for i in order], type=pa.int64()).cast(pa.timestamp("us", tz="UTC"))
    metrics = [[("m00", 0.5 + i), ("m01", 1.5)] if i != 3 else [("m01", 1.5)] for i in order]
    attrs = [[("split", "train" if i < 4 else "test")] for i in order]
    labels = [None if i < 4 else i % 2 for i in order]
    return pa.table(
        {
            "entity_id": pa.array(entities, pa.string()),
            "event_time": times,
            "metrics": pa.array(metrics, pa.map_(pa.string(), pa.float64())),
            "attrs": pa.array(attrs, pa.map_(pa.string(), pa.string())),
            "label": pa.array(labels, pa.int32()),
        },
        schema=S.arrow_schema("raw"),
    )


def test_replay_ten_records_into_mock_producer():
    producer = MockProducer()
    t0 = int(time.time() * 1000)
    sent = R.replay(producer, _raw_table(10), "smd", speedup=0, out=lambda *a, **k: None)
    t1 = int(time.time() * 1000)

    assert sent == 10
    assert len(producer.messages) == 10
    assert producer.flushed

    values = []
    for topic, key, raw in producer.messages:
        assert topic == S.TOPIC_SMD
        assert b"\n" not in raw
        value = json.loads(raw)
        assert list(value) == FIELDS
        assert key == value["entity_id"].encode("utf-8")
        assert value["source"] == "smd"
        assert t0 <= int(value["attrs"]["ingest_ts"]) <= t1
        values.append(value)

    # event_time order, ISO ms with trailing Z
    times = [v["event_time"] for v in values]
    assert times == sorted(times)
    assert times[0] == "2026-01-01T00:00:00.000Z"
    assert times[-1] == "2026-01-01T00:09:00.000Z"

    # a missing metric is left out, labels and split carried over
    assert "m00" not in values[3]["metrics"] and values[3]["metrics"] == {"m01": 1.5}
    assert values[0]["label"] is None and values[0]["attrs"]["split"] == "train"
    assert values[9]["label"] == 1 and values[9]["attrs"]["split"] == "test"
    assert values[9]["metrics"]["m00"] == pytest.approx(9.5)


def test_limit_takes_first_records_in_event_order():
    producer = MockProducer()
    sent = R.replay(producer, _raw_table(10), "smd", speedup=0, limit=3, out=lambda *a, **k: None)
    assert sent == 3
    times = [json.loads(m[2])["event_time"] for m in producer.messages]
    assert times == ["2026-01-01T00:00:00.000Z", "2026-01-01T00:01:00.000Z", "2026-01-01T00:02:00.000Z"]


def test_alibaba_goes_to_alibaba_topic():
    producer = MockProducer()
    R.replay(producer, _raw_table(2), "alibaba", speedup=0, out=lambda *a, **k: None)
    assert {m[0] for m in producer.messages} == {S.TOPIC_ALIBABA}
    assert json.loads(producer.messages[0][2])["source"] == "alibaba"


def test_buffer_error_is_retried():
    producer = MockProducer(buffer_errors=3)
    sent = R.replay(producer, _raw_table(4), "smd", speedup=0, out=lambda *a, **k: None)
    assert sent == 4 and len(producer.messages) == 4


def test_speedup_paces_by_event_time():
    clock = {"t": 0.0}

    def monotonic():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    producer = MockProducer()
    # 10 records, 1 event-minute apart, speedup 60 -> 1 s apart -> 9 s in total
    R.replay(producer, _raw_table(10), "smd", speedup=60, monotonic=monotonic, sleep=sleep,
             out=lambda *a, **k: None)
    assert len(producer.messages) == 10
    assert clock["t"] == pytest.approx(9.0)


def test_reporter_prints_rate_every_5_seconds():
    clock = {"t": 0.0}
    lines = []
    rep = R.Reporter(lambda: clock["t"], lambda s, **k: lines.append(s))
    rep.count = 100
    clock["t"] = 4.0
    rep.tick()
    assert lines == []
    clock["t"] = 5.0
    rep.tick()
    assert len(lines) == 1 and lines[0].startswith("20 records/s")


def test_main_reads_raw_layer_from_disk(tmp_path):
    folder = tmp_path / "data/raw/source=smd"
    folder.mkdir(parents=True)
    pq.write_table(_raw_table(10), folder / "fake.parquet")
    producer = MockProducer()
    code = R.main(["--source", "smd", "--speedup", "0", "--limit", "4", "--root", str(tmp_path)],
                  producer=producer)
    assert code == 0
    assert len(producer.messages) == 4


def test_main_fails_without_raw_data(tmp_path):
    assert R.main(["--speedup", "0", "--root", str(tmp_path)], producer=MockProducer()) == 1
