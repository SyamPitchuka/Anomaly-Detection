"""The stateful scoring of src/streaming/score_job.py gives the batch AE v2 answers (no Spark needed)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from contracts import schemas as S
from src.detection import autoencoder as AE
from src.streaming import score_job as M

BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
N_TRAIN, N_TEST = 120, 80          # rows (minutes) per machine -> 24 train + 16 test windows
RECENTER, ADAPTIVE = 4, 6          # small, so both kick in within tens of windows
MACHINES = ("machine-a", "machine-b")


def _rows(machine: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    vals = rng.normal(size=(n, len(S.SMD_METRICS))) + 5 * seed
    vals[N_TRAIN + 30:N_TRAIN + 40, :4] += 6.0         # anomaly in the test part
    vals[N_TRAIN + 50:, 5] += 3.0                      # level shift (exercises the rolling median)
    vals[7, 3] = np.nan                                # missing values
    vals[N_TRAIN + 12, :] = np.nan
    df = pd.DataFrame(vals, columns=S.SMD_METRICS)
    df.insert(0, "event_us", BASE_US + np.arange(n, dtype=np.int64) * 60_000_000)
    df["split"] = ["train"] * N_TRAIN + ["test"] * N_TEST
    label = np.full(n, np.nan)
    label[N_TRAIN:] = 0
    label[N_TRAIN + 30:N_TRAIN + 40] = 1
    df["label"] = label
    df.insert(0, "entity_id", machine)
    return df


def _raw_table(df: pd.DataFrame) -> pa.Table:
    metrics = [[(k, v) for k, v in zip(S.SMD_METRICS, r) if not np.isnan(v)]
               for r in df[S.SMD_METRICS].to_numpy()]
    return pa.Table.from_arrays(
        [pa.array(df["entity_id"]), pa.array(df["event_us"]).cast(pa.timestamp("us", tz="UTC")),
         pa.array(metrics, type=pa.map_(pa.string(), pa.float64())),
         pa.array([[("split", s)] for s in df["split"]], type=pa.map_(pa.string(), pa.string())),
         pa.array(df["label"], from_pandas=True).cast(pa.int32())],
        schema=S.arrow_schema("raw"))


@pytest.fixture(scope="module")
def setup():
    raw = pd.concat([_rows(m, i + 1) for i, m in enumerate(MACHINES)], ignore_index=True)
    aggs = ("mean", "max")
    feature_metrics = [m for _ in aggs for m in S.SMD_METRICS]
    meta, X = AE.build_windows(_raw_table(raw), S.SMD_METRICS, aggs)
    scalers = AE.fit_scalers(meta, X)
    for s in scalers.values():
        s["masked"] = ["m37"]
    torch.manual_seed(0)
    model = AE.DenseAE(X.shape[1]).eval()

    # batch reference, exactly as AE.run() does it
    centers = AE.recenter_centers(meta, X, scalers, RECENTER)
    Z, mask = AE.standardise(meta, X, scalers, feature_metrics, centers)
    ae_score, top = AE.score(model, Z, mask, S.SMD_METRICS)
    threshold = {"mode": "global", "global": float(np.nanmedian(ae_score)), "per_machine": {},
                 "adaptive": {"windows": ADAPTIVE, "percentile": 99, "min_periods": ADAPTIVE // 2}}
    limit = AE.adaptive_limits(meta, ae_score, ADAPTIVE, 99)
    ae_flag = AE.flags(meta, ae_score, threshold, "global", limit)
    batch = meta.assign(ae_score=ae_score, ae_flag=ae_flag, top=top)

    scaler = {"metrics": S.SMD_METRICS, "inputs": "mean_max", "entities": scalers,
              "recenter": {"windows": RECENTER, "min_periods": RECENTER // 2}}
    models = {"ae": M.ae_bundle(model, scaler, threshold), "tier1": None, "baselines_module": None}
    return raw, batch, models


def _stream(raw: pd.DataFrame, models: dict, chunk: int) -> pd.DataFrame:
    """Feed each machine's rows in chunks (windows straddle chunks), round-tripping the state through the
    tuple form GroupState stores, as Spark does between micro-batches."""
    n_in = len(models["ae"]["feature_metrics"])
    outs = []
    for machine, rows in raw.groupby("entity_id"):
        state = None
        rows = rows.sample(frac=1.0, random_state=1)  # rows inside a micro-batch need not be ordered
        rows = rows.sort_values("event_us", kind="stable")
        for start in range(0, len(rows), chunk):
            st = M.decode_state(state, n_in) if state is not None else M.new_state(n_in)
            part = rows.iloc[start:start + chunk].sample(frac=1.0, random_state=start)
            out, late = M.step(machine, part.drop(columns="entity_id"), st, models)
            assert late == 0
            state = M.encode_state(st)
            if len(out):
                outs.append(out)
    return pd.concat(outs, ignore_index=True)


@pytest.mark.parametrize("chunk", [7, 37, 10_000])
def test_stream_matches_batch(setup, chunk):
    raw, batch, models = setup
    out = _stream(raw, models, chunk)
    assert list(out.columns) == M.OUTPUT_COLUMNS
    merged = out.merge(batch, left_on=["entity_id", "ws_us"], right_on=["entity_id", "window_start"],
                       suffixes=("", "_b"), validate="one_to_one")
    # every batch window except each machine's last ones (still within the 10-minute watermark) is emitted
    assert len(merged) == len(out)
    assert len(batch) - len(out) == 3 * len(MACHINES)   # 40 windows, last 3 end after minute 199 - 10
    np.testing.assert_allclose(merged["ae_score"], merged["ae_score_b"], rtol=0, atol=1e-6)
    assert (merged["ae_flag"].astype(bool) == merged["ae_flag_b"]).all()
    assert merged["ae_flag_b"].any() and not merged["ae_flag_b"].all()
    assert [list(t) for t in merged["top_metrics"]] == [list(t) for t in merged["top"]]
    assert merged["split"].tolist() == merged["split_b"].tolist()
    np.testing.assert_array_equal(merged["label"].to_numpy(), merged["label_b"].to_numpy())
    assert out["tier1_score"].isna().all() and out["tier1_flag"].isna().all()


def test_late_rows_dropped(setup):
    raw, _, models = setup
    n_in = len(models["ae"]["feature_metrics"])
    rows = raw[raw["entity_id"] == MACHINES[0]].drop(columns="entity_id")
    st = M.new_state(n_in)
    out, late = M.step(MACHINES[0], rows.iloc[:60], st, models)
    assert late == 0 and len(out) == 9                # minutes 0..59: windows ending by minute 49
    out, late = M.step(MACHINES[0], rows.iloc[:3], st, models)   # replay of rows in an emitted window
    assert late == 3 and len(out) == 0


def test_tier1_hook(setup, tmp_path):
    raw, _, models = setup
    path = tmp_path / "tier1_smd" / "baselines.parquet"
    path.parent.mkdir()
    pd.DataFrame({"entity_id": ["machine-x"] * 3 + ["other"], "metric": ["m00", "m01", "m02", "m00"],
                  "median": [10.0, 0.0, 1.0, 0.0], "mad": [1.0, 0.0, 2.0, 1.0]}).to_parquet(path)
    assert M.load_tier1_baselines(tmp_path / "missing.parquet") is None
    models = {**models, "tier1": M.load_tier1_baselines(path)}

    # window 0: m00 mean 10 + 2*1.4826 -> z 2; window 1: m00 mean 10 + 5*1.4826 -> z 5.
    # m01 is far off but has mad 0 (skipped); m02 is at its median (z 0); other metrics have no baseline.
    minutes = np.arange(25)
    m00 = np.where(minutes < 5, 10 + 2 * M.TIER1_MAD_SCALE, 10 + 5 * M.TIER1_MAD_SCALE)
    rows = pd.DataFrame({m: 1.0 for m in S.SMD_METRICS}, index=minutes)
    rows["m00"], rows["m01"] = m00, 1e6
    rows.insert(0, "event_us", BASE_US + minutes * 60_000_000)
    rows["split"], rows["label"] = "test", 0.0
    out, _ = M.step("machine-x", rows, M.new_state(len(models["ae"]["feature_metrics"])), models)
    assert out["tier1_score"].iloc[:2].tolist() == pytest.approx([2.0, 5.0])
    assert out["tier1_flag"].iloc[:2].tolist() == [False, True]
    assert out["ae_score"].isna().all()               # machine-x is unknown to the autoencoder

    out, _ = M.step(MACHINES[0], raw[raw["entity_id"] == MACHINES[0]].drop(columns="entity_id").iloc[:30],
                    M.new_state(len(models["ae"]["feature_metrics"])), models)
    assert out["tier1_score"].isna().all() and out["tier1_flag"].isna().all()   # no baselines for it


def test_write_scores_file(setup, tmp_path):
    raw, _, models = setup
    out = _stream(raw, models, 10_000)
    test = out[out["split"] == "test"].assign(is_anomaly=lambda d: d["ae_flag"].astype(bool))
    path = M.write_scores_file(test.drop(columns="split"), 3, tmp_path)
    table = pq.read_table(path)
    assert table.schema.names == list(S.LAYERS["scores"])
    assert all(S.type_matches(S.LAYERS["scores"][f.name], f.type) for f in table.schema)
    assert table.num_rows == len(test)
