"""Latency side output of src/streaming/score_job.py and the summary in src/streaming/latency_report.py."""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.streaming import latency_report as LR
from src.streaming import score_job as M
from tests.test_score_job import MACHINES, setup  # noqa: F401  (module fixture: tiny AE + 2 machines)

WINDOW_MIN = 5


def _with_ingest(rows: pd.DataFrame, t0: int = 1_000_000) -> pd.DataFrame:
    """ingest_ms = t0 + 10 ms per minute of event time (records arrive in event-time order)."""
    minutes = (rows["event_us"] - rows["event_us"].min()) // 60_000_000
    return rows.assign(ingest_ms=(t0 + 10 * minutes).astype("float64"))


def test_window_carries_newest_ingest(setup):  # noqa: F811
    raw, _, models = setup
    n_in = len(models["ae"]["feature_metrics"])
    rows = _with_ingest(raw[raw["entity_id"] == MACHINES[0]].drop(columns="entity_id"))
    # two micro-batches, with the state round-tripped as GroupState stores it; window [25, 30) straddles them
    st = M.new_state(n_in)
    out1, _ = M.step(MACHINES[0], rows.iloc[:27].sample(frac=1.0, random_state=0), st, models)
    st = M.decode_state(M.encode_state(st), n_in)
    out2, _ = M.step(MACHINES[0], rows.iloc[27:60], st, models)
    out = pd.concat([out1, out2], ignore_index=True)
    assert list(out.columns) == M.OUTPUT_COLUMNS
    first_minute = (out["ws_us"] - rows["event_us"].min()) // 60_000_000
    # newest record of window [s, s+5) is minute s+4
    np.testing.assert_array_equal(out["max_ingest_ms"], 1_000_000 + 10 * (first_minute + WINDOW_MIN - 1))
    # each window's batch value is the newest ingest of the micro-batch that closed it
    assert set(out1["batch_ingest_ms"]) == {1_000_000 + 10 * 26}
    assert set(out2["batch_ingest_ms"]) == {1_000_000 + 10 * 59}
    assert (out["batch_ingest_ms"] >= out["max_ingest_ms"]).all()


def test_no_ingest_column_gives_null(setup):  # noqa: F811
    raw, _, models = setup
    rows = raw[raw["entity_id"] == MACHINES[0]].drop(columns="entity_id").iloc[:60]
    out, _ = M.step(MACHINES[0], rows, M.new_state(len(models["ae"]["feature_metrics"])), models)
    assert len(out) and out["max_ingest_ms"].isna().all() and out["batch_ingest_ms"].isna().all()


def test_write_latency_file(tmp_path):
    pdf = pd.DataFrame({"entity_id": ["a", "b"], "ws_us": [1_767_225_600_000_000, 1_767_225_900_000_000],
                        "max_ingest_ms": [1000.0, np.nan], "batch_ingest_ms": [1500.0, 1500.0]})
    path = M.write_latency_file(pdf, 7, 2000, tmp_path)
    table = pq.read_table(path)
    assert table.schema.equals(M.LATENCY_SCHEMA)
    got = table.to_pydict()
    assert got["max_ingest_ts_ms"] == [1000, None] and got["emit_ts_ms"] == [2000, 2000]
    assert got["batch_id"] == [7, 7] and path.name == "batch-000007.parquet"
    assert str(table["window_start"][0].as_py()) == "2026-01-01 00:00:00+00:00"


def test_latency_summary(tmp_path):
    # 100 windows: total latency 1..100 s, processing 0.5 s each
    ingest = np.arange(100, dtype=np.int64) * 1000
    emit = ingest + (np.arange(100) + 1) * 1000
    pdf = pd.DataFrame({"entity_id": ["m"] * 100, "ws_us": np.arange(100, dtype=np.int64) * 300_000_000,
                        "max_ingest_ms": ingest.astype(float), "batch_ingest_ms": (emit - 500).astype(float)})
    for b in (0, 1):
        part = pdf.iloc[b * 50:(b + 1) * 50]
        M.write_latency_file(part, b, 0, tmp_path / "latency")
    # write_latency_file stamps one emit per batch; patch the files with per-row emit times for the test
    df = LR.load_latency(tmp_path / "latency")
    df["emit_ts_ms"] = emit
    s = LR.summarise(df, speedup=100)
    assert s["windows"] == 100
    assert s["total_p50_s"] == pytest.approx(50.5) and s["total_max_s"] == pytest.approx(100)
    assert s["total_p95_s"] == pytest.approx(95.05)
    assert s["proc_p95_s"] == pytest.approx(0.5)
    assert s["wait_theory_s"] == pytest.approx(11 * 60 / 100)


def test_lag_from_checkpoint(tmp_path):
    ck = tmp_path / "ck"
    (ck / "offsets").mkdir(parents=True)
    (ck / "commits").mkdir()
    for i, off in enumerate([{"0": 5, "1": 7}, {"0": 50, "1": 70}]):
        (ck / "offsets" / str(i)).write_text("v1\n{}\n" + json.dumps({"telemetry.smd": off}))
    (ck / "commits" / "0").write_text("v1\n{}")
    assert LR.committed_offsets(ck, "telemetry.smd") == {0: 5, 1: 7}   # batch 1 not committed yet
    assert LR.committed_offsets(tmp_path / "missing", "telemetry.smd") == {}
    assert LR.lag({0: 100, 1: 100}, {0: 5, 1: 7}) == 188


def test_worker_log(tmp_path, monkeypatch):
    path = tmp_path / "w.jsonl"
    monkeypatch.delenv(M.WORKER_LOG_ENV, raising=False)
    M.worker_log("load_models", seconds=1.0)          # off by default: nothing written
    assert not path.exists()
    monkeypatch.setenv(M.WORKER_LOG_ENV, str(path))
    M.worker_log("load_models", seconds=1.0)
    rec = json.loads(path.read_text().splitlines()[-1])
    assert rec["event"] == "load_models" and rec["seconds"] == 1.0 and isinstance(rec["pid"], int)
