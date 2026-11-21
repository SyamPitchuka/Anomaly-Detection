"""experiments/e8_alibaba_input.py: the hours-0-72 raw slice (one file per machine) and the scores file."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from contracts import schemas as S
from experiments import e8_alibaba_input as A

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def write_raw(folder, hours=(0, 1, 71.9, 72, 100)):
    """Two machines in one Spark-like part file, rows at the given hours after T0 (+ one row in each 5-min step)."""
    rows = []
    for m in ("m_2", "m_1"):
        for h in hours:
            t = T0 + timedelta(hours=h)
            rows.append({"entity_id": m, "event_time": t, "metrics": [("cpu", 1.0), ("mem", 2.0)],
                         "attrs": [], "label": None})
    schema = S.arrow_schema("raw")
    folder.mkdir(parents=True)
    pq.write_table(pa.table({n: [r[n] for r in rows] for n in schema.names}, schema=schema),
                   folder / "part-00000.parquet")


def test_slice_raw_keeps_hours_0_to_72_one_file_per_machine(tmp_path):
    write_raw(tmp_path / "raw")
    info = A.slice_raw(tmp_path / "raw", tmp_path / "out", hours=72)
    assert info["machines"] == 2 and info["rows"] == 6  # hours 0, 1, 71.9 per machine; 72 and 100 dropped
    files = sorted(p.name for p in (tmp_path / "out").glob("*.parquet"))
    assert files == ["m_1.parquet", "m_2.parquet"]
    t = pq.read_table(tmp_path / "out" / "m_1.parquet")
    assert t.schema.names == list(S.LAYERS["raw"]) and set(t["entity_id"].to_pylist()) == {"m_1"}
    times = t["event_time"].to_pylist()
    assert times == sorted(times) and max(times) < T0 + timedelta(hours=72)
    # a rerun replaces old files
    (tmp_path / "out" / "stale.parquet").write_bytes(b"x")
    A.slice_raw(tmp_path / "raw", tmp_path / "out", hours=1)
    assert sorted(p.name for p in (tmp_path / "out").glob("*.parquet")) == ["m_1.parquet", "m_2.parquet"]


def test_score_writes_scores_layer_with_is_anomaly_equal_ae_flag(tmp_path, monkeypatch):
    from src.detection import autoencoder as M
    ws = np.array([0, 300, 600], dtype=np.int64) * 1_000_000 + int(T0.timestamp() * 1e6)
    meta = pd.DataFrame({"entity_id": ["m_1"] * 3, "window_start": ws, "split": "test", "label": [0.0] * 3})
    seen = {}

    def fake_score_layer(model_dir, raw_dir):
        seen["args"] = (model_dir, raw_dir)
        return meta, np.array([0.1, 0.9, 0.2]), np.array([False, True, False]), [["cpu"], ["mem", "cpu"], []]

    monkeypatch.setattr(M, "score_layer", fake_score_layer)
    info = A.score(tmp_path / "model", tmp_path / "raw", tmp_path / "scores")
    assert seen["args"] == (tmp_path / "model", tmp_path / "raw")
    assert info["windows"] == 3 and info["flagged"] == 1 and info["machines"] == 1
    t = pq.read_table(tmp_path / "scores" / "scores.parquet")
    assert t.schema.names == list(S.LAYERS["scores"])
    assert t["is_anomaly"].to_pylist() == t["ae_flag"].to_pylist() == [False, True, False]
