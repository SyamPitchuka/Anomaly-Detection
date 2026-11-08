from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from contracts import schemas as S
from src.incidents import attribution_eval as A

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
N_TRAIN = {"machine-1-1": 100, "machine-1-2": 50}


def write_raw(root, machine, n_train, n_test=100):
    labels = pa.array([None] * n_train + [0] * n_test, type=pa.int32())
    pq.write_table(pa.table({"label": labels}), _dir(root, "raw") / f"{machine}.parquet")


def _dir(root, layer):
    path = root / S.PATHS[layer] / ("source=smd" if layer != "smd_interpretation" else "")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_interpretation(root, rows):
    schema = S.arrow_schema("smd_interpretation")
    table = pa.table({n: [r[i] for r in rows] for i, n in enumerate(schema.names)}, schema=schema)
    pq.write_table(table, _dir(root, "smd_interpretation") / "interpretation.parquet")


def write_incidents(root, rows):
    """rows: (entity_id, minute_start, minute_end, top_metrics); minutes since TIME_BASE_UTC."""
    schema = S.arrow_schema("incidents")
    cols = {
        "incident_id": [f"{e}_{m0}" for e, m0, _, _ in rows], "entity_id": [r[0] for r in rows],
        "start_time": [BASE + timedelta(minutes=r[1]) for r in rows],
        "end_time": [BASE + timedelta(minutes=r[2]) for r in rows],
        "n_windows": [1] * len(rows), "max_score": [1.0] * len(rows),
        "top_metrics": [r[3] for r in rows], "created_at": [BASE] * len(rows),
    }
    pq.write_table(pa.table(cols, schema=schema), _dir(root, "incidents") / "part.parquet")


def build_fixture(root):
    for machine, n_train in N_TRAIN.items():
        write_raw(root, machine, n_train)
    # segment minutes (global) = n_train + idx: machine-1-1 -> [110, 121), [140, 146); machine-1-2 -> [60, 66)
    write_interpretation(root, [
        ("machine-1-1", 10, 20, ["m00", "m05"]),     # overlaps incident A (m05 in top 3)          -> hit
        ("machine-1-1", 40, 45, ["m07"]),            # overlaps incident B, m07 only 4th           -> miss
        ("machine-1-2", 10, 15, ["m01"]),            # no incident of machine-1-2 overlaps         -> no incident
        ("machine-1-2", 30, 35, ["m02"]),            # incident of machine-1-1 at same time ignored -> no incident
    ])
    write_incidents(root, [
        ("machine-1-1", 115, 125, ["m09", "m05", "m02"]),   # A: overlaps segment 1 only
        ("machine-1-1", 140, 150, ["m01", "m02", "m03", "m07"]),   # B: m07 is beyond the first 3
        ("machine-1-1", 200, 210, ["m00"]),                  # far away, overlaps nothing
        ("machine-1-2", 100, 110, ["m01"]),                  # machine-1-2 incident that overlaps no segment
        ("machine-1-1", 80, 90, ["m02"]),                    # machine-1-1 incident in machine-1-2 segment 4's time: ignored
    ])


def test_segment_bounds_use_train_offset_and_inclusive_end():
    start, end = A.segment_bounds_us(100, 10, 20)
    assert start == int((BASE + timedelta(minutes=110)).timestamp()) * 1_000_000
    assert end == int((BASE + timedelta(minutes=121)).timestamp()) * 1_000_000


def test_train_row_counts_count_null_labels(tmp_path):
    build_fixture(tmp_path)
    got = A.train_row_counts(tmp_path / S.PATHS["raw"] / "source=smd", ["machine-1-1", "machine-1-2"])
    assert got == N_TRAIN


def test_evaluate_and_summary(tmp_path):
    build_fixture(tmp_path)
    summary = A.run(tmp_path, tmp_path / "out" / "attribution.md")
    by_label = {r["label"]: r for r in summary}
    assert by_label["**All machines**"] == {"label": "**All machines**", "segments": 4, "with_incident": 2, "hits": 1}
    assert by_label["machine-1-1"]["segments"] == 2 and by_label["machine-1-1"]["with_incident"] == 2
    assert by_label["machine-1-1"]["hits"] == 1
    assert by_label["machine-1-2"] == {"label": "machine-1-2", "segments": 2, "with_incident": 0, "hits": 0}


def test_report_is_a_markdown_table(tmp_path):
    build_fixture(tmp_path)
    out = tmp_path / "out" / "attribution.md"
    A.run(tmp_path, out)
    text = out.read_text(encoding="utf-8")
    assert "| Machine | Segments | With incident | Hits@3 | Hit-rate@3 | Hit-rate if covered |" in text
    assert "| **All machines** | 4 | 2 | 1 | 25.0 % | 50.0 % |" in text
    assert "| machine-1-2 | 2 | 0 | 0 | 0.0 % | n/a |" in text


def test_machines_sorted_naturally():
    rows = [{"entity_id": m, "has_incident": False, "hit": False} for m in ("machine-1-10", "machine-1-2")]
    assert [r["label"] for r in A.summarize(rows)][1:] == ["machine-1-2", "machine-1-10"]
