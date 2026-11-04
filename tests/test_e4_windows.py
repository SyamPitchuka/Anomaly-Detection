import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from experiments import e4_windows as E
from src.detection import autoencoder as A
from src.detection import baselines as B
from src.detection import e1_report as R
from src.detection import tier1_zscore as T

N_TRAIN, N_TEST = 300, 150                # rows (minutes) per machine
ANOMALY_ROWS = range(60, 75)              # test rows 60..74 -> 15 anomalous minutes
BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
MACHINES = ["machine-1-1", "machine-1-2"]


def _machine_table(machine: str, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    t = np.arange(n, dtype=np.int64)
    latent = np.stack([np.sin(t / 30), np.cos(t / 50)], axis=1)
    values = latent @ rng.normal(size=(2, 38)) + 0.05 * rng.normal(size=(n, 38)) + 10 * seed
    label = np.zeros(N_TEST, dtype=np.int32)
    for i in ANOMALY_ROWS:
        values[N_TRAIN + i, :6] += 8.0
        label[i] = 1
    split = ["train"] * N_TRAIN + ["test"] * N_TEST
    return pa.Table.from_arrays(
        [pa.array([machine] * n, type=pa.string()),
         pa.array(BASE_US + t * 60_000_000, type=pa.int64()).cast(pa.timestamp("us", tz="UTC")),
         pa.array([list(zip(S.SMD_METRICS, v)) for v in values.tolist()], type=pa.map_(pa.string(), pa.float64())),
         pa.array([[("split", s)] for s in split], type=pa.map_(pa.string(), pa.string())),
         pa.concat_arrays([pa.nulls(N_TRAIN, pa.int32()), pa.array(label, type=pa.int32())])],
        schema=S.arrow_schema("raw"))


def _windows(rows: int, w: int) -> int:
    return -(-rows // w)                  # BASE_US is aligned to 15 minutes, so ceil(rows / w)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    raw = root / "data/raw/source=smd"
    raw.mkdir(parents=True)
    for seed, machine in enumerate(MACHINES, start=1):
        pq.write_table(_machine_table(machine, seed), raw / f"{machine}.parquet")
    pq.write_table(_machine_table("fake", 9).slice(0, 10), raw / "fake.parquet")   # must be skipped
    return root


@pytest.fixture(scope="module")
def spark():
    session = E.build_spark()
    session.sparkContext.setLogLevel("ERROR")
    yield session


def test_spans_keep_time_fixed():
    assert E.spans(5) == {"recenter": 24, "adaptive": 288}
    assert E.spans(1) == {"recenter": 120, "adaptive": 1440}
    assert E.spans(15) == {"recenter": 8, "adaptive": 96}


@pytest.mark.parametrize("w", [1, 5, 15])
def test_ae_windows_count(w):
    table = _machine_table("machine-1-1", 1)
    meta, X = A.build_windows(table, S.SMD_METRICS, ("mean", "max"), w * 60_000_000)
    assert len(meta) == _windows(N_TRAIN + N_TEST, w)
    assert (meta["split"] == "train").sum() == N_TRAIN // w      # the window at the boundary mixes -> test
    assert np.diff(meta["window_start"]).tolist() == [w * 60_000_000] * (len(meta) - 1)
    assert X.shape == (len(meta), 76)


def test_ae_default_is_5_minutes():
    table = _machine_table("machine-1-1", 1)
    m0, x0 = A.build_windows(table, S.SMD_METRICS, ("mean", "max"))
    m5, x5 = A.build_windows(table, S.SMD_METRICS, ("mean", "max"), 5 * 60_000_000)
    pd.testing.assert_frame_equal(m0, m5)
    np.testing.assert_array_equal(x0, x5)


@pytest.mark.parametrize("w", [1, 5, 15])
def test_spark_windows_count(spark, repo, w):
    raw = spark.read.parquet((repo / "data/raw/source=smd/machine-1-1.parquet").as_posix())
    b = B.build_windows(raw, S.SMD_METRICS, E.window_str(w))
    _, meta = T.build_windows(raw, S.SMD_METRICS, E.window_str(w))
    for df in (b, meta):
        assert df.count() == _windows(N_TRAIN + N_TEST, w)
        assert df.filter("split = 'train'").count() == N_TRAIN // w


def test_spark_default_is_5_minutes(spark, repo):
    raw = spark.read.parquet((repo / "data/raw/source=smd/machine-1-1.parquet").as_posix())
    cols = ["entity_id", "ws_us", "split", "label", "m00", "m37"]
    a = B.build_windows(raw, S.SMD_METRICS).select(*cols).orderBy("ws_us").toPandas()
    b = B.build_windows(raw, S.SMD_METRICS, "5 minutes").select(*cols).orderBy("ws_us").toPandas()
    pd.testing.assert_frame_equal(a, b)
    ta = T.build_windows(raw, S.SMD_METRICS)[1].orderBy("window_start").toPandas()
    tb = T.build_windows(raw, S.SMD_METRICS, "5 minutes")[1].orderBy("window_start").toPandas()
    pd.testing.assert_frame_equal(ta, tb)


def _fake_scores(n_per_machine: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for m in MACHINES:
        label = np.zeros(n_per_machine, dtype=int)
        label[10:14] = 1
        label[25:27] = 1
        s = rng.random(n_per_machine) + label
        parts.append(pd.DataFrame({
            "entity_id": m, "ws_us": BASE_US + np.arange(n_per_machine) * 300_000_000, "label": label,
            "tier1_score": s * 4, "tier1_flag": s * 4 > 3, "tier2_score": s, "tier2_flag": s > 1.1,
            "pca_score": s, "ae_score": s, "ae_flag": s > 1.3}))
    df = pd.concat(parts, ignore_index=True)
    df["is_anomaly"] = df["ae_flag"]
    return df


def test_5_minutes_matches_e1_report():
    df = _fake_scores()
    ours, pm = E.evaluate_window(df, 1.2, 5)
    e1_df = df.assign(window_start=df["ws_us"])
    theirs, pm1 = R.evaluate_all(e1_df, 1.2)
    for a, b in zip(ours, theirs):
        assert a["key"] == b["key"]
        for k in E.CHECK_KEYS:
            assert a[k] == pytest.approx(b[k], nan_ok=True), (a["key"], k)
    pd.testing.assert_frame_equal(pm, pm1)


def test_window_scales_minutes():
    df = _fake_scores()
    r5 = {r["key"]: r for r in E.evaluate_window(df, 1.2, 5)[0]}
    r15 = {r["key"]: r for r in E.evaluate_window(df, 1.2, 15)[0]}
    for k in r5:
        assert r15[k]["composite_f1"] == pytest.approx(r5[k]["composite_f1"])
        assert r15[k]["false_alarms_per_machine_day"] == pytest.approx(r5[k]["false_alarms_per_machine_day"] / 3)
        # same flagged windows, so the same false-alarm time whatever w is
        assert r15[k]["false_alarm_minutes_per_machine_day"] == pytest.approx(
            r5[k]["false_alarm_minutes_per_machine_day"])
        assert r5[k]["false_alarm_minutes_per_machine_day"] == pytest.approx(
            r5[k]["false_alarms_per_machine_day"] * 5)
        assert r15[k]["mean_delay_min"] == pytest.approx(r5[k]["mean_delay_min"] * 3, nan_ok=True)


def test_reference_faults_and_overlap(repo):
    faults = E.reference_faults(repo / "data/raw/source=smd", None, 5)
    assert set(faults) == set(MACHINES)
    start = BASE_US + (N_TRAIN + 60) * 60_000_000
    assert faults["machine-1-1"] == [(start, start + 15 * 60_000_000)]
    one = {"machine-1-1": faults["machine-1-1"]}
    df = pd.DataFrame({"entity_id": "machine-1-1",
                       "ws_us": [start - 15 * 60_000_000, start + 14 * 60_000_000, start + 15 * 60_000_000]})
    assert E.reference_caught(df, np.array([True, False, False]), 15, one) == 0   # ends where the fault starts
    assert E.reference_caught(df, np.array([False, True, False]), 1, one) == 1    # last minute of the fault
    assert E.reference_caught(df, np.array([False, False, True]), 1, one) == 0    # just after it


@pytest.fixture(scope="module")
def e4_run(repo, spark):
    return E.run([5, 15], root=repo, spark=spark, epochs=3, out_dir=repo / "out")


def test_run_writes_only_e4_folders(e4_run, repo):
    for w in (5, 15):
        scores = pq.read_table(repo / f"data/e4/w{w}/scores/source=smd/scores.parquet")
        assert scores.schema.equals(S.arrow_schema("scores"))
        assert scores.num_rows == len(MACHINES) * (_windows(N_TRAIN + N_TEST, w) - N_TRAIN // w)
        for name in ("tier1_smd", "kmeans_smd", "pca_smd", "ae_smd"):
            assert (repo / f"models/e4/w{w}/{name}").is_dir()
    assert not (repo / "data/scores").exists() and not (repo / "data/scores_ae").exists()
    assert not (repo / "data/scores_tier1").exists() and not (repo / "data/features").exists()
    assert not any((repo / "models").glob("*_smd"))
    assert {r["window_min"] for r in e4_run["rows"]} == {5, 15}
    assert e4_run["runs"][1]["info"]["recenter"] == 8 and e4_run["runs"][1]["info"]["adaptive"] == 96
    final = [r for r in e4_run["rows"] if r["key"] == "is_anomaly"]
    ae = [r for r in e4_run["rows"] if r["key"] == "ae_v2"]
    assert [r["composite_f1"] for r in final] == [r["composite_f1"] for r in ae]
    doc = json.loads((repo / "out/e4.json").read_text())
    assert doc["windows_min"] == [5, 15]
    assert b"\r\n" not in (repo / "out/e4.md").read_bytes()
    assert (repo / "out/e4.png").stat().st_size > 0
    assert "false-alarm minutes / machine / day" in (repo / "out/e4.md").read_text(encoding="utf-8")


def test_report_only_matches_run(e4_run, repo):
    rebuilt = E.report_only(root=repo, out_dir=repo / "out")
    assert [r["window"] for r in rebuilt["runs"]] == [5, 15]
    for a, b in zip(e4_run["rows"], rebuilt["rows"]):
        assert a["key"] == b["key"] and a["window_min"] == b["window_min"]
        for k in (*E.CHECK_KEYS, "false_alarm_minutes_per_machine_day", "ref_faults_caught", "mean_delay_min"):
            assert a[k] == pytest.approx(b[k], nan_ok=True), (a["key"], a["window_min"], k)
    assert "--report-only" in (repo / "out/e4.md").read_text(encoding="utf-8")


def test_spread_labels_no_overlap():
    ys = [0.50, 0.49, 0.10, 0.495]
    out = E.spread_labels(ys, gap=0.06)
    assert out[2] == pytest.approx(0.10)                              # far label untouched
    near = sorted(out[i] for i in (0, 1, 3))
    assert all(b - a >= 0.06 - 1e-12 for a, b in zip(near, near[1:]))
    assert [i for i in sorted(range(4), key=lambda i: out[i])] == [2, 1, 3, 0]   # order kept
    top = E.spread_labels([0.99, 0.995, 1.0], gap=0.06)
    assert max(top) <= 1.0 and min(b - a for a, b in zip(sorted(top), sorted(top)[1:])) >= 0.06 - 1e-12
