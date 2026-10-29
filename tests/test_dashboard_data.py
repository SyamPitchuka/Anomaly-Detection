from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from src.dashboard.data import load_incidents, load_scores

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCORE_COLS = list(S.LAYERS["scores"])
INCIDENT_COLS = list(S.LAYERS["incidents"])


def score_row(entity, i, bad):
    w = BASE + timedelta(minutes=5 * i)
    return {"entity_id": entity, "window_start": w, "window_end": w + timedelta(minutes=5),
            "tier1_score": 4.0 if bad else 0.5, "tier1_flag": bad, "tier2_score": 0.5, "tier2_flag": False,
            "pca_score": 0.5, "ae_score": 4.0 if bad else 0.5, "ae_flag": bad, "is_anomaly": bad,
            "top_metrics": ["m00", "m01", "m02"] if bad else [], "label": int(bad)}


def incident_row(entity):
    return {"incident_id": f"{entity}_202601010000", "entity_id": entity, "start_time": BASE,
            "end_time": BASE + timedelta(minutes=10), "n_windows": 2, "max_score": 4.0,
            "top_metrics": ["m00", "m01", "m02"], "created_at": BASE + timedelta(days=1)}


def write(root, layer, source, name, rows, schema=None):
    schema = schema or S.arrow_schema(layer)
    table = pa.table({c: [r[c] for r in rows] for c in schema.names}, schema=schema)
    folder = root / S.PATHS[layer] / f"source={source}"
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, folder / name)


def test_reads_every_file_for_the_source(tmp_path):
    write(tmp_path, "scores", "smd", "part-0.parquet", [score_row("machine-1-1", i, i == 1) for i in range(3)])
    write(tmp_path, "scores", "smd", "part-1.parquet", [score_row("machine-1-2", i, False) for i in range(2)])
    write(tmp_path, "scores", "alibaba", "part-0.parquet", [score_row("m_1932", 0, True)])
    df = load_scores("smd", root=tmp_path)
    assert list(df.columns) == SCORE_COLS
    assert len(df) == 5
    assert int(df["is_anomaly"].sum()) == 1
    assert set(df["entity_id"]) == {"machine-1-1", "machine-1-2"}
    assert len(load_scores("alibaba", root=tmp_path)) == 1


def test_incidents(tmp_path):
    write(tmp_path, "incidents", "smd", "part-0.parquet", [incident_row("machine-1-2")])
    df = load_incidents("smd", root=tmp_path)
    assert list(df.columns) == INCIDENT_COLS
    assert df.iloc[0]["incident_id"] == "machine-1-2_202601010000"
    assert list(df.iloc[0]["top_metrics"]) == ["m00", "m01", "m02"]
    assert str(df["start_time"].dt.tz) == "UTC"


@pytest.mark.parametrize("make_folder", [False, True])
def test_missing_or_empty_folder_gives_empty_frame(tmp_path, make_folder):
    if make_folder:
        (tmp_path / S.PATHS["scores"] / "source=smd").mkdir(parents=True)
        (tmp_path / S.PATHS["incidents"] / "source=smd").mkdir(parents=True)
    scores, incidents = load_scores("smd", root=tmp_path), load_incidents("smd", root=tmp_path)
    assert scores.empty and list(scores.columns) == SCORE_COLS
    assert incidents.empty and list(incidents.columns) == INCIDENT_COLS


def test_skips_spark_side_files(tmp_path):
    write(tmp_path, "incidents", "smd", "part-00000.snappy.parquet", [incident_row("machine-1-1")])
    folder = tmp_path / S.PATHS["incidents"] / "source=smd"
    (folder / "_SUCCESS").write_bytes(b"")
    (folder / ".part-00000.snappy.parquet.crc").write_bytes(b"not parquet")
    assert len(load_incidents("smd", root=tmp_path)) == 1


def test_spark_style_types_and_extra_columns(tmp_path):
    # Spark may write tz-naive timestamps and int64 labels; extra columns are dropped.
    fields = [pa.field(f.name, pa.timestamp("us") if pa.types.is_timestamp(f.type) else f.type)
              for f in S.arrow_schema("scores")]
    fields = [pa.field("label", pa.int64()) if f.name == "label" else f for f in fields]
    schema = pa.schema(fields + [pa.field("debug", pa.string())])
    rows = [{**score_row("machine-1-1", 0, True), "window_start": BASE.replace(tzinfo=None),
             "window_end": (BASE + timedelta(minutes=5)).replace(tzinfo=None), "debug": "x"}]
    write(tmp_path, "scores", "smd", "part-0.parquet", rows, schema=schema)
    df = load_scores("smd", root=tmp_path)
    assert list(df.columns) == SCORE_COLS
    assert df.iloc[0]["window_start"] == BASE


def test_missing_contract_column_raises(tmp_path):
    schema = S.arrow_schema("incidents").remove(S.arrow_schema("incidents").get_field_index("max_score"))
    write(tmp_path, "incidents", "smd", "part-0.parquet", [incident_row("machine-1-1")], schema=schema)
    with pytest.raises(ValueError, match="max_score"):
        load_incidents("smd", root=tmp_path)


def test_unknown_source_raises(tmp_path):
    with pytest.raises(ValueError):
        load_scores("google", root=tmp_path)
