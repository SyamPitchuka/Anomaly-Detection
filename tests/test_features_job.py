import math
from datetime import datetime, timedelta

import pytest

from contracts import schemas as S
from src.streaming import features_job as M

BASE = datetime(2026, 1, 1)  # session time zone is UTC
RAW_DDL = "entity_id STRING, event_time TIMESTAMP, metrics MAP<STRING,DOUBLE>"


@pytest.fixture(scope="module")
def spark():
    session = M.build_spark("test_features_job", with_kafka=False)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _rows():
    rows = []
    # machine-a, window 00:00-00:05: m00 = 2 + 3 t (t = 0..4 minutes); m01 only at t = 0 (missing afterwards)
    for t in range(5):
        metrics = {"m00": 2.0 + 3.0 * t}
        if t == 0:
            metrics["m01"] = 7.0
        rows.append(("machine-a", BASE + timedelta(minutes=t), metrics))
    # machine-a, window 00:05-00:10: m00 = 100 - 2 t at t = 0.5, 1.5, 2.5 (30 s offset, not on the minute)
    for t in (0.5, 1.5, 2.5):
        rows.append(("machine-a", BASE + timedelta(minutes=5 + t), {"m00": 100.0 - 2.0 * t}))
    # machine-b, window 00:00-00:05: constant series
    for t in range(3):
        rows.append(("machine-b", BASE + timedelta(minutes=t), {"m00": 4.0}))
    return rows


def _by_key(df):
    return {(r["entity_id"], r["window_start"], r["metric"]): r for r in df.collect()}


def test_build_features_batch(spark):
    out = M.build_features(spark.createDataFrame(_rows(), RAW_DDL))
    assert out.columns == list(S.LAYERS["features"])
    assert dict(out.dtypes) == {"entity_id": "string", "window_start": "timestamp", "window_end": "timestamp",
                                "metric": "string", "mean": "double", "max": "double", "std": "double",
                                "slope": "double", "n": "bigint"}
    rows = _by_key(out)
    assert len(rows) == 4

    w0, w1 = BASE, BASE + timedelta(minutes=5)
    a0 = rows[("machine-a", w0, "m00")]
    assert a0["window_end"] == w1
    assert a0["n"] == 5
    assert a0["mean"] == pytest.approx(8.0)
    assert a0["max"] == pytest.approx(14.0)
    assert a0["std"] == pytest.approx(3.0 * math.sqrt(2.5))  # sample std of 0..4 is sqrt(2.5)
    assert a0["slope"] == pytest.approx(3.0)

    single = rows[("machine-a", w0, "m01")]
    assert single["n"] == 1 and single["mean"] == pytest.approx(7.0)
    assert single["slope"] == 0.0  # n < 2

    a1 = rows[("machine-a", w1, "m00")]
    assert a1["n"] == 3
    assert a1["mean"] == pytest.approx(97.0)
    assert a1["max"] == pytest.approx(99.0)
    assert a1["std"] == pytest.approx(2.0)
    assert a1["slope"] == pytest.approx(-2.0)

    b0 = rows[("machine-b", w0, "m00")]
    assert b0["n"] == 3 and b0["std"] == pytest.approx(0.0) and b0["slope"] == pytest.approx(0.0)


def test_build_features_streaming(spark, tmp_path):
    """Same function as a streaming aggregation with a watermark, in append mode."""
    src = tmp_path / "raw"
    late = ("machine-a", BASE + timedelta(hours=1), {"m00": 0.0})  # moves the watermark past both windows
    spark.createDataFrame(_rows() + [late], RAW_DDL).write.parquet(src.as_posix())

    stream = spark.readStream.schema(RAW_DDL).parquet(src.as_posix()).withWatermark("event_time", M.WATERMARK)
    query = (M.build_features(stream).writeStream.format("memory").queryName("features_test")
             .outputMode("append").option("checkpointLocation", (tmp_path / "ckpt").as_posix())
             .trigger(availableNow=True).start())
    query.awaitTermination()

    rows = _by_key(spark.table("features_test"))
    assert ("machine-a", BASE + timedelta(hours=1), "m00") not in rows  # still open in state
    a0 = rows[("machine-a", BASE, "m00")]
    assert a0["n"] == 5 and a0["slope"] == pytest.approx(3.0)
    assert len(rows) == 4
