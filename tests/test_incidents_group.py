import importlib.util
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from src.incidents import group as G

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

_spec = importlib.util.spec_from_file_location("fake_data", ROOT / "tools/fake_data.py")
fake_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake_data)


def fake_scores() -> list[dict]:
    """The scores layer of tools/fake_data.py: machine-1-2 is anomalous in windows 00:20, 00:25, 00:30."""
    return fake_data.scores(fake_data.smd_rows(random.Random(42)))


def alert(entity, minute, ae, metrics=("m00", "m01", "m02"), anomaly=True):
    start = T0 + timedelta(minutes=minute)
    return {"entity_id": entity, "window_start": start, "window_end": start + timedelta(minutes=5),
            "tier1_score": None, "tier1_flag": None, "tier2_score": None, "tier2_flag": None, "pca_score": None,
            "ae_score": ae, "ae_flag": anomaly, "is_anomaly": anomaly, "top_metrics": list(metrics), "label": None}


def write_scores(root: Path, rows: list[dict], name="part.parquet") -> None:
    schema = S.arrow_schema("scores")
    table = pa.table({n: [r[n] for r in rows] for n in schema.names}, schema=schema)
    path = root / S.PATHS["scores"] / "source=smd" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture(scope="module")
def spark():
    session = G.build_spark("test_incidents_group", driver_memory="1g")
    session.conf.set("spark.sql.shuffle.partitions", "2")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def batch(spark, tmp_path, rows) -> list[dict]:
    write_scores(tmp_path, rows)
    G.run_batch("smd", tmp_path, spark)
    out = spark.read.parquet(G.incidents_dir(tmp_path, "smd").as_posix()).orderBy("incident_id").collect()
    return [r.asDict() for r in out]


def test_fake_data_gives_one_incident(spark, tmp_path):
    rows = batch(spark, tmp_path, fake_scores())
    assert [r["incident_id"] for r in rows] == ["machine-1-2_202601010020"]
    inc = rows[0]
    assert inc["entity_id"] == "machine-1-2" and inc["n_windows"] == 3
    assert inc["start_time"].astimezone(timezone.utc) == datetime(2026, 1, 1, 0, 20, tzinfo=timezone.utc)   # collect() gives local naive time
    assert inc["end_time"] - inc["start_time"] == timedelta(minutes=15)
    assert inc["max_score"] == 5.2 and inc["top_metrics"] == ["m00", "m01", "m02"]
    assert inc["created_at"] is not None


def test_output_follows_the_contract(spark, tmp_path):
    batch(spark, tmp_path, fake_scores())
    out = G.incidents_dir(tmp_path, "smd")
    got = pq.read_schema(next(out.glob("*.parquet")))
    assert got.names == list(S.LAYERS["incidents"])
    for name, code in S.LAYERS["incidents"].items():
        assert S.type_matches(code, got.field(name).type), name


def test_gap_of_one_normal_window_is_bridged_two_are_not(spark, tmp_path):
    rows = [alert("machine-1-1", m, 1.0) for m in (0, 10, 25)] + \
           [alert("machine-1-1", m, 0.1, anomaly=False) for m in (5, 15, 20)]
    got = batch(spark, tmp_path, rows)
    assert [(r["incident_id"], r["n_windows"]) for r in got] == [("machine-1-1_202601011000", 2),
                                                                 ("machine-1-1_202601011025", 1)]
    assert got[0]["end_time"] - got[0]["start_time"] == timedelta(minutes=15)


def test_machines_are_separate_and_duplicates_count_once(spark, tmp_path):
    rows = [alert("machine-1-2", 40, 0.26), alert("machine-2-1", 40, 97.47), alert("machine-2-1", 40, 50.0),
            alert("machine-2-1", 40, None)]
    got = batch(spark, tmp_path, rows)
    assert [(r["incident_id"], r["n_windows"], r["max_score"]) for r in got] == [
        ("machine-1-2_202601011040", 1, 0.26), ("machine-2-1_202601011040", 1, 97.47)]


def test_max_score_null_when_no_ae_score_and_nulls_ignored(spark, tmp_path):
    rows = [alert("machine-1-1", 0, None), alert("machine-1-1", 5, None),
            alert("machine-1-2", 0, None), alert("machine-1-2", 5, 2.5)]
    got = {r["entity_id"]: r["max_score"] for r in batch(spark, tmp_path, rows)}
    assert got == {"machine-1-1": None, "machine-1-2": 2.5}


def test_top_metrics_most_frequent_ties_by_name(spark, tmp_path):
    rows = [alert("machine-1-2", 20, 0.15, ("m00", "m02", "m01")), alert("machine-1-2", 25, 0.26, ("m00", "m02", "m01")),
            alert("machine-1-2", 30, 0.26, ("m00", "m02")),
            alert("machine-1-1", 0, 1.0, ("m09", "m03", "m05")), alert("machine-1-1", 5, 1.0, ("m03",)),
            alert("machine-1-1", 10, 1.0, []),
            alert("machine-2-1", 0, 1.0, ("m07",))]
    got = {r["entity_id"]: r["top_metrics"] for r in batch(spark, tmp_path, rows)}
    assert got["machine-1-2"] == ["m00", "m02", "m01"]
    assert got["machine-1-1"] == ["m03", "m05", "m09"]       # m03 twice, then the tie by name
    assert got["machine-2-1"] == ["m07"]                      # fewer than 3 metrics: shorter list


def test_no_alerts_gives_no_incidents(spark, tmp_path):
    assert batch(spark, tmp_path, [alert("machine-1-1", 0, 0.1, anomaly=False)]) == []


def test_stream_matches_batch_on_fake_data(spark, tmp_path):
    write_scores(tmp_path, fake_scores())
    G.start_stream("smd", tmp_path, spark, once=True).awaitTermination(120)
    # An incident is written when the watermark passes its end, which needs later windows to arrive.
    write_scores(tmp_path, [alert("machine-2-1", 240, 0.1, anomaly=False)], name="later.parquet")
    G.start_stream("smd", tmp_path, spark, once=True).awaitTermination(120)
    out = spark.read.parquet(G.incidents_dir(tmp_path, "smd").as_posix())
    got = [r.asDict() for r in out.collect()]
    assert [r["incident_id"] for r in got] == ["machine-1-2_202601010020"]
    assert got[0]["n_windows"] == 3 and got[0]["max_score"] == 5.2
    assert got[0]["top_metrics"] == ["m00", "m01", "m02"]
    assert out.columns == list(S.LAYERS["incidents"])
