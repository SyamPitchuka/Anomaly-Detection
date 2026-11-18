"""E5b: corrbreak faults (inject_faults.py), Alibaba AE (autoencoder.py) and the experiment's scoring wiring."""
import random
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from experiments import e5b_alibaba_ae as X
from src.detection import autoencoder as M
from src.producer import inject_faults as I

MACHINES = [f"m_{i}" for i in range(1, 6)]
HOURS, CLEAN_HOURS, TRAIN_FROM = 6, 2, 3
STEP_S = 10
N = HOURS * 3600 // STEP_S


def _cols(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(N) * STEP_S
    cpu = 50 + 20 * np.sin(t / 3600 * 2 * np.pi / 3) + rng.normal(0, 1, N)
    net_in = 20 + 5 * np.sin(t / 1800) + rng.normal(0, 1, N)
    return {"cpu": cpu, "mem": 10 + 0.6 * cpu + rng.normal(0, 0.5, N), "mem_gps": 5 + rng.normal(0, 0.1, N),
            "mkpi": 1.5 + rng.normal(0, 0.05, N), "net_in": net_in, "net_out": 0.5 * net_in + rng.normal(0, 0.3, N),
            "disk_io": 3 + rng.normal(0, 0.2, N)}


def _table(entity: str, cols: dict, rows=slice(None)) -> pa.Table:
    t = (np.arange(N, dtype=np.int64) * STEP_S)[rows]
    metrics = [[(k, float(v[rows][i])) for k, v in cols.items()] for i in range(len(t))]
    schema = S.arrow_schema("raw")
    return pa.table({"entity_id": pa.array([entity] * len(t), pa.string()),
                     "event_time": pa.array((I.BASE_S + t) * 1_000_000, pa.int64()).cast(schema.field("event_time").type),
                     "metrics": pa.array(metrics, schema.field("metrics").type),
                     "attrs": pa.array([[] for _ in t], schema.field("attrs").type),
                     "label": pa.nulls(len(t), pa.int32())}, schema=schema)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("e5b")
    full = root / "data/raw/source=alibaba"
    head = root / "head/data/raw/source=alibaba"              # hours 0..TRAIN_FROM only
    full.mkdir(parents=True)
    head.mkdir(parents=True)
    for seed, entity in enumerate(MACHINES, start=1):
        cols = _cols(seed)
        pq.write_table(_table(entity, cols), full / f"{entity}.parquet")
        pq.write_table(_table(entity, cols, slice(0, TRAIN_FROM * 3600 // STEP_S)), head / f"{entity}.parquet")
    return root


@pytest.fixture(scope="module")
def injected(repo):
    return I.inject(repo, repo / "out", machines=5, hours=HOURS, clean_hours=CLEAN_HOURS, per_type=1, seed=3,
                    types=I.ALL_TYPES)


def _secs(text):
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def _frame(path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    out = pd.DataFrame([dict(m) for m in df["metrics"]]).reindex(columns=I.METRICS)
    since = pd.to_datetime(df["event_time"], utc=True) - pd.Timestamp(0, tz="UTC")
    out.insert(0, "ts", (since // pd.Timedelta(seconds=1)).to_numpy())
    out["label"] = df["label"].to_numpy()
    return out


# ---------------------------------------------------------------- corrbreak

def test_corrbreak_stays_inside_the_machines_clean_range_and_changes_only_its_metric(repo, injected):
    f = next(x for x in injected["faults"] if x["type"] == "corrbreak")
    before = _frame(repo / f"data/raw/source=alibaba/{f['entity_id']}.parquet")
    after = _frame(repo / f"out/data/raw/source=alibaba/{f['entity_id']}.parquet")
    start, end, m = _secs(f["start_time"]), _secs(f["end_time"]), f["metric"]
    inside = ((before["ts"] >= start) & (before["ts"] < end)).to_numpy()
    clean = before.loc[before["ts"] < I.BASE_S + CLEAN_HOURS * 3600, m]
    assert end - start == 900 and start >= I.BASE_S + CLEAN_HOURS * 3600 and inside.sum() == f["n_rows"]
    new = after.loc[inside, m]
    assert new.between(clean.min(), clean.max()).all()                       # every value inside the clean min-max
    assert (f["params"]["clean_min"], f["params"]["clean_max"]) == (clean.min(), clean.max())
    donor = _secs(f["params"]["donor_start"])
    assert donor + 900 <= I.BASE_S + CLEAN_HOURS * 3600                     # donor stretch from the clean part
    expected = before.loc[(before["ts"] >= donor) & (before["ts"] < donor + 900), m].to_numpy()
    np.testing.assert_array_equal(new.to_numpy(), expected)
    assert not np.allclose(new.to_numpy(), before.loc[inside, m].to_numpy())   # the metric really changed
    np.testing.assert_array_equal(after.loc[~inside, m].to_numpy(), before.loc[~inside, m].to_numpy())
    others = [x for x in I.METRICS if x != m]
    np.testing.assert_array_equal(after[others].to_numpy(), before[others].to_numpy())
    assert (after["label"].to_numpy() == inside.astype(int)).all()
    assert f["params"]["partner"] != m and abs(f["params"]["r"]) >= I.CORR_MIN_R


def test_corrbreak_moves_the_metric_to_the_other_side_of_the_clean_median(repo, injected):
    f = next(x for x in injected["faults"] if x["type"] == "corrbreak")
    before = _frame(repo / f"data/raw/source=alibaba/{f['entity_id']}.parquet")
    after = _frame(repo / f"out/data/raw/source=alibaba/{f['entity_id']}.parquet")
    start, end, m = _secs(f["start_time"]), _secs(f["end_time"]), f["metric"]
    inside = (before["ts"] >= start) & (before["ts"] < end)
    median = before.loc[before["ts"] < I.BASE_S + CLEAN_HOURS * 3600, m].median()
    assert (before.loc[inside, m].mean() - median) * (after.loc[inside, m].mean() - median) < 0


def test_adding_corrbreak_keeps_the_e5_faults_of_a_seed(repo, injected):
    e5 = I.inject(repo, repo / "out_e5", machines=5, hours=HOURS, clean_hours=CLEAN_HOURS, per_type=1, seed=3)
    assert [f for f in injected["faults"] if f["type"] != "corrbreak"] == e5["faults"]
    assert sorted(f["type"] for f in injected["faults"]) == ["break", "corrbreak", "drift", "spike"]
    assert len({f["entity_id"] for f in injected["faults"]}) == 4


def test_corrbreak_targets_prefer_correlated_metrics():
    df = pd.DataFrame({"ts": I.BASE_S + np.arange(N) * STEP_S, **_cols(1)})
    targets = I.corrbreak_targets(df, I.BASE_S + CLEAN_HOURS * 3600, I.METRICS)
    assert {"cpu", "mem", "net_in", "net_out"} <= set(targets)
    assert targets["mem"][0] == "cpu" and targets["net_out"][0] == "net_in"
    assert "disk_io" not in targets and "mkpi" not in targets


def test_unknown_fault_type_is_an_error(repo):
    series = I.read_slice(repo, 5, 0, HOURS)
    with pytest.raises(ValueError, match="unknown fault types"):
        I.plan_faults(series, I.BASE_S + CLEAN_HOURS * 3600, 1, random.Random(0), ("spike", "melt"))


# ---------------------------------------------------------------- AE on Alibaba

@pytest.fixture(scope="module")
def trained(repo):
    res = M.run("alibaba", root=repo, epochs=2, verbose=False, train_from_hours=TRAIN_FROM,
                model_dir=repo / "models/ae_alibaba", scores_dir=repo / "scores_ae")
    return res


def test_alibaba_time_split_trains_only_on_later_hours(repo, trained):
    meta = trained["meta"]
    cut = (I.BASE_S + TRAIN_FROM * 3600) * 1_000_000
    assert ((meta["split"] == "train") == (meta["window_start"] >= cut)).all()
    assert meta.loc[meta["split"] == "train", "label"].isna().all()
    assert (meta.loc[meta["split"] == "test", "label"] == 0).all()
    assert trained["ae_score"].shape == (len(meta),)
    scaler = (repo / "models/ae_alibaba/scaler.json").read_text()
    assert '"metrics": ["cpu", "mem", "mem_gps", "mkpi", "net_in", "net_out", "disk_io"]' in scaler


def test_alibaba_without_time_split_is_an_error(repo):
    with pytest.raises(ValueError, match="train-from-hours"):
        M.run("alibaba", root=repo, epochs=1, verbose=False, model_dir=repo / "m2", scores_dir=repo / "s2")


def test_score_layer_matches_the_training_runs_test_windows(repo, trained):
    """The test windows of run() come first in time, so scoring hours 0..TRAIN_FROM alone gives the same answer."""
    meta, score, flag, top = M.score_layer(repo / "models/ae_alibaba", repo / "head/data/raw/source=alibaba")
    test = (trained["meta"]["split"] == "test").to_numpy()
    ref = trained["meta"][test].reset_index(drop=True)
    pd.testing.assert_series_equal(meta["window_start"], ref["window_start"])
    np.testing.assert_allclose(score, trained["ae_score"][test], rtol=1e-6)
    np.testing.assert_array_equal(flag, trained["ae_flag"][test])
    assert all(len(t) == 3 and set(t) <= set(S.ALIBABA_METRICS) for t in top)


def test_score_layer_leaves_unknown_machines_unscored(repo, trained, tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    pq.write_table(_table("m_99", _cols(9)), folder / "m_99.parquet")
    pq.write_table(_table("m_1", _cols(1)), folder / "m_1.parquet")
    meta, score, flag, _ = M.score_layer(repo / "models/ae_alibaba", folder)
    unknown = (meta["entity_id"] == "m_99").to_numpy()
    assert np.isnan(score[unknown]).all() and not flag[unknown].any() and np.isfinite(score[~unknown]).all()


# ---------------------------------------------------------------- experiment wiring

def test_replayed_records_round_trip_to_a_raw_layer(repo, injected, tmp_path):
    records = I.replay_records(repo / "out")
    X.records_to_raw(records, tmp_path / "raw")
    for entity in MACHINES:
        a = _frame(repo / f"out/data/raw/source=alibaba/{entity}.parquet")
        b = _frame(tmp_path / f"raw/{entity}.parquet")
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)


def test_ae_flags_feed_the_e5_report(repo, injected, trained):
    meta, _, flag, _ = M.score_layer(repo / "models/ae_alibaba", repo / "out/data/raw/source=alibaba")
    flags = X.ae_flags(meta, flag)
    assert list(flags.columns) == ["entity_id", "window", "flag"] and (flags["window"] % 300 == 0).all()
    windows = I.window_labels(I.to_frame(I.replay_records(repo / "out")))
    table = I.report(injected, windows, flags)
    assert list(table.index) == [*I.ALL_TYPES, "all"] and table.loc["all", "events"] == 4
    # every fault caught -> recall 1 in the report built from these flags
    hit = flags.copy()
    for f in injected["faults"]:
        hit.loc[(hit["entity_id"] == f["entity_id"]) & (hit["window"] == _secs(f["start_time"])), "flag"] = True
    assert (I.report(injected, windows, hit)["recall"] == 1.0).all()


def test_matched_threshold_picks_the_closest_false_alarm_rate(repo, injected):
    df = I.to_frame(I.replay_records(repo / "out"))
    windows = I.window_labels(df)
    scores = I.detect(df, _secs(injected["clean_end"]))
    t, table, sweep = X.matched_threshold(scores, injected, windows, target_fa=0.0, thresholds=(3.0, 1e9))
    assert t == 1e9 and table.loc["all", "false_alarms/day"] == 0.0 and len(sweep) == 2
    fas = {s["threshold"]: s["false_alarms_per_day"] for s in sweep}
    t2, _, _ = X.matched_threshold(scores, injected, windows, target_fa=fas[3.0], thresholds=(3.0, 1e9))
    assert t2 == 3.0


def test_score_separation_uses_the_clean_part_as_normal(injected):
    f = injected["faults"][0]
    start = _secs(f["start_time"])
    clean_end = _secs(injected["clean_end"])
    meta = pd.DataFrame({"entity_id": [f["entity_id"]] * 3,
                         "window_start": np.array([clean_end - 600, clean_end - 300, start], dtype=np.int64) * 1_000_000})
    info = {"global": 1.0, "per_machine": {f["entity_id"]: 0.3}}
    sep = X.score_separation({"faults": [f]}, meta, np.array([0.1, 0.1, 0.5]), info, clean_end)
    assert sep[0]["peak"] == 0.5 and sep[0]["normal_median"] == 0.1 and sep[0]["ratio"] == pytest.approx(5.0)
    assert sep[0]["over_global"] is False and sep[0]["over_own_threshold"] is True
