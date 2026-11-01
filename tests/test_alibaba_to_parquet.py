from datetime import datetime, timedelta, timezone

import pyarrow.dataset as ds
import pytest

from contracts import schemas as S
from src.producer import alibaba_to_parquet as A

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)

# machine_id, time_stamp, cpu, mem, mem_gps, mkpi, net_in, net_out, disk_io (no header; empty = missing).
# Rows are deliberately out of time order for m_2.
CSV = """\
m_1,0,10.5,20.5,30.5,40.5,50.5,60.5,70.5
m_1,10,,21.5,31.5,41.5,51.5,61.5,71.5
m_1,20,12.5,22.5,,,,,
m_1,30,,,,,,,
m_1,40,14.5,24.5,34.5,44.5,54.5,64.5,74.5
m_2,20,1.0,2.0,3.0,4.0,5.0,6.0,7.0
m_2,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
m_2,10,1.5,2.5,3.5,4.5,5.5,6.5,7.5
m_3,0,9.0,9.0,9.0,9.0,9.0,9.0,9.0
m_3,10,9.5,9.5,9.5,9.5,9.5,9.5,9.5
,10,1.0,1.0,1.0,1.0,1.0,1.0,1.0
"""


@pytest.fixture(scope="module")
def spark():
    session = A.build_spark("test_alibaba_to_parquet", driver_memory="1g")
    session.conf.set("spark.sql.shuffle.partitions", "4")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def csv_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("alibaba_in") / "machine_usage.csv"
    path.write_text(CSV)
    return path


def _read(root) -> list[dict]:
    dset = ds.dataset(root / S.PATHS["raw"], format="parquet", partitioning="hive")
    rows = dset.to_table().to_pylist()
    return sorted(rows, key=lambda r: (r["entity_id"], r["event_time"]))


@pytest.fixture(scope="module")
def full(spark, csv_file, tmp_path_factory):
    root = tmp_path_factory.mktemp("repo_full")
    summary = A.convert(csv_file, root, spark=spark, partitions=2)
    return root, summary


def test_summary_and_contract_schema(full):
    root, summary = full
    assert summary["rows"] == 10 and summary["machines"] == 3      # the row without machine_id is dropped
    assert summary["empty_metrics"] == 1                            # m_1 at time_stamp 30
    dset = ds.dataset(root / S.PATHS["raw"], format="parquet", partitioning="hive")
    got = {f.name: f.type for f in dset.schema}
    assert list(got)[:len(S.LAYERS["raw"])] == list(S.LAYERS["raw"])
    assert set(got) - set(S.LAYERS["raw"]) <= {"source"}
    for name, code in S.LAYERS["raw"].items():
        assert S.type_matches(code, got[name]), name
    assert (root / S.PATHS["raw"] / "source=alibaba").is_dir()


def test_records_follow_the_contract(full):
    rows = _read(full[0])
    first = rows[0]
    assert first["entity_id"] == "m_1"
    assert first["event_time"] == BASE
    assert dict(first["metrics"]) == {"cpu": 10.5, "mem": 20.5, "mem_gps": 30.5, "mkpi": 40.5,
                                      "net_in": 50.5, "net_out": 60.5, "disk_io": 70.5}
    assert first["attrs"] == [] and first["label"] is None
    for r in rows:
        assert set(dict(r["metrics"])) <= set(S.ALIBABA_METRICS)
        assert r["attrs"] == [] and r["label"] is None


def test_event_time_is_base_plus_seconds(full):
    rows = [r for r in _read(full[0]) if r["entity_id"] == "m_2"]
    assert [r["event_time"] for r in rows] == [BASE + timedelta(seconds=s) for s in (0, 10, 20)]
    assert dict(rows[0]["metrics"])["cpu"] == 0.0      # a real zero is kept


def test_missing_values_are_left_out_not_zero(full):
    by_time = {r["event_time"]: dict(r["metrics"]) for r in _read(full[0]) if r["entity_id"] == "m_1"}
    assert "cpu" not in by_time[BASE + timedelta(seconds=10)]
    assert by_time[BASE + timedelta(seconds=10)]["mem"] == 21.5
    assert set(by_time[BASE + timedelta(seconds=20)]) == {"cpu", "mem"}
    assert by_time[BASE + timedelta(seconds=30)] == {}


def test_machines_option_keeps_first_n_sorted(spark, csv_file, tmp_path):
    summary = A.convert(csv_file, tmp_path, machines=2, spark=spark, partitions=2)
    assert summary["machines"] == 2 and summary["rows"] == 8
    assert {r["entity_id"] for r in _read(tmp_path)} == {"m_1", "m_2"}


def test_rerun_overwrites(spark, csv_file, tmp_path):
    A.convert(csv_file, tmp_path, spark=spark, partitions=2)
    summary = A.convert(csv_file, tmp_path, machines=1, spark=spark, partitions=2)
    assert summary["rows"] == 5 and {r["entity_id"] for r in _read(tmp_path)} == {"m_1"}
