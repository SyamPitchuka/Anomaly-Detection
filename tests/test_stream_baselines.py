"""src/detection/stream_baselines.py gives the batch Tier 2 / 2b answers of baselines.py, whatever the
micro-batch split, and score_job.step() feeds it the right history."""
import functools
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from contracts import schemas as S
from src.detection import autoencoder as AE
from src.detection import baselines as B
from src.detection import stream_baselines as SB
from src.streaming import score_job as SJ

BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
N_TRAIN, N_TEST = 300, 150          # rows (minutes) per machine -> 60 train + 30 test windows
RECENTER = 6
MACHINES = ("machine-1-1", "machine-1-2")
METRICS = S.SMD_METRICS
MEAN_COLS = [f"{m}_mean" for m in METRICS]


def _rows(machine: str, seed: int) -> pd.DataFrame:
    """Rows as score_job.step() takes them (event_us, m00..m37, split, label) plus entity_id."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    t = np.arange(n)
    latent = np.stack([np.sin(t / 30), np.cos(t / 50)], axis=1)
    vals = latent @ rng.normal(size=(2, len(METRICS))) + 0.1 * rng.normal(size=(n, len(METRICS))) + 5 * seed
    vals[N_TRAIN + 40:N_TRAIN + 50, :4] += 6.0          # spike in the test part
    vals[N_TRAIN + 80:, 5:8] += 3.0                     # lasting level shift (exercises the rolling median)
    vals[N_TRAIN + 3, 4] = np.nan                       # missing value
    vals[N_TRAIN + 20:N_TRAIN + 25, 9] = np.nan         # metric absent from a whole test window
    df = pd.DataFrame(vals, columns=METRICS)
    df.insert(0, "event_us", BASE_US + t.astype(np.int64) * 60_000_000)
    df["split"] = ["train"] * N_TRAIN + ["test"] * N_TEST
    label = np.full(n, np.nan)
    label[N_TRAIN:] = 0
    label[N_TRAIN + 40:N_TRAIN + 50] = 1
    df["label"] = label
    df.insert(0, "entity_id", machine)
    return df


def _raw_table(df: pd.DataFrame) -> pa.Table:
    metrics = [[(k, v) for k, v in zip(METRICS, r) if not np.isnan(v)] for r in df[METRICS].to_numpy()]
    return pa.Table.from_arrays(
        [pa.array(df["entity_id"]), pa.array(df["event_us"]).cast(pa.timestamp("us", tz="UTC")),
         pa.array(metrics, type=pa.map_(pa.string(), pa.float64())),
         pa.array([[("split", s)] for s in df["split"]], type=pa.map_(pa.string(), pa.string())),
         pa.array(df["label"], from_pandas=True).cast(pa.int32())],
        schema=S.arrow_schema("raw"))


def _windows(rows: pd.DataFrame) -> pd.DataFrame:
    """One machine's windows as score_job hands them over: entity_id, window_start, split, <metric>_mean."""
    meta, X = AE.build_windows(_raw_table(rows), METRICS, ("mean",))
    pdf = pd.DataFrame(X, columns=MEAN_COLS)
    pdf.insert(0, "entity_id", meta["entity_id"].to_numpy())
    pdf.insert(1, "window_start", meta["window_start"].to_numpy())
    pdf.insert(2, "split", meta["split"].to_numpy())
    return pdf


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """baselines.py run on the fixture with --recenter RECENTER; returns (root, run result, raw rows)."""
    root = tmp_path_factory.mktemp("repo")
    raw_dir = root / "data/raw/source=smd"
    raw_dir.mkdir(parents=True)
    rows = {m: _rows(m, i + 1) for i, m in enumerate(MACHINES)}
    for m, df in rows.items():
        pq.write_table(_raw_table(df), raw_dir / f"{m}.parquet")
    spark = B.build_spark("test_stream_baselines")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        result = B.run("smd", kmax=3, root=root, spark=spark, recenter=RECENTER)
    finally:
        spark.stop()
    return root, result, rows


@pytest.fixture(scope="module")
def models(batch):
    root = batch[0]
    return SB.load_models(root / "models/kmeans_smd", root / "models/pca_smd")


def _check_vs_batch(got: pd.DataFrame, windows: pd.DataFrame, result: dict, machine: str):
    ref = result["test"][result["test"]["entity_id"] == machine].set_index("ws_us")
    test = (windows["split"] == "test").to_numpy()
    starts = windows.loc[test, "window_start"].to_numpy()
    assert len(starts) == len(ref) == N_TEST // 5
    ref = ref.loc[starts]
    np.testing.assert_allclose(got.loc[test, "tier2_score"], ref["tier2_score"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(got.loc[test, "pca_score"], ref["pca_score"], rtol=0, atol=1e-9)
    assert got.loc[test, "tier2_flag"].tolist() == ref["tier2_flag"].astype(bool).tolist()


def test_load_models_matches_pipeline_model(batch, models):
    _, result, _ = batch
    np.testing.assert_array_equal(models["centers"], result["centers"])
    np.testing.assert_array_equal(models["pc"], result["pc"])
    assert models["recenter"] == RECENTER
    assert models["threshold"]["tier2_score"] == pytest.approx(result["thresholds"]["tier2_score"], rel=1e-12)
    assert models["threshold"]["pca_score"] == pytest.approx(result["thresholds"]["pca_score"], rel=1e-12)
    assert set(models["scalers"]) == set(MACHINES)


def test_one_call_equals_batch(batch, models):
    _, result, rows = batch
    flagged = 0
    for machine in MACHINES:
        windows = _windows(rows[machine])
        got = SB.score_baselines(windows, models)
        assert list(got.columns) == SB.OUTPUT_COLUMNS and (got.index == windows.index).all()
        _check_vs_batch(got, windows, result, machine)
        flagged += sum(got["tier2_flag"])
    assert flagged > 0   # the fixture exercises the flag, not only the score


@pytest.mark.parametrize("sizes", [[1], [7], [13, 1, 29, 2], [59, 1, 1, 3]])
@pytest.mark.parametrize("keep", [None, RECENTER])
def test_split_calls_with_history(batch, models, sizes, keep):
    """Micro-batch independence: windows split over calls, history = previous window means (all of them,
    or only the last RECENTER as score_job keeps), give the one-call answer."""
    _, _, rows = batch
    for machine in MACHINES:
        windows = _windows(rows[machine])
        whole = SB.score_baselines(windows, models)
        X = windows[MEAN_COLS].to_numpy()
        parts, start, i = [], 0, 0
        while start < len(windows):
            stop = start + sizes[i % len(sizes)]
            history = X[:start] if keep is None else X[max(0, start - keep):start]
            parts.append(SB.score_baselines(windows.iloc[start:stop], {"baselines": models}, history))
            start, i = stop, i + 1
        got = pd.concat(parts)
        np.testing.assert_allclose(got["tier2_score"], whole["tier2_score"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(got["pca_score"], whole["pca_score"], rtol=0, atol=1e-12)
        assert got["tier2_flag"].tolist() == whole["tier2_flag"].tolist()


def test_unknown_machine_and_no_models(batch, models):
    windows = _windows(batch[2][MACHINES[0]]).iloc[:5].assign(entity_id="machine-9-9")
    for m in (models, None, {"baselines": None}):
        got = SB.score_baselines(windows, m)
        assert got["tier2_score"].isna().all() and got["pca_score"].isna().all()
        assert got["tier2_flag"].isna().all()
    assert SB.load_models(batch[0] / "models/nope", batch[0] / "models/pca_smd") is None


@pytest.mark.parametrize("chunk", [11, 97])
def test_score_job_step_passes_history(batch, models, chunk):
    """score_job.step() over row chunks (windows straddle chunks) gives the batch tier2/pca answers."""
    _, result, rows = batch
    torch.manual_seed(0)
    ae_scaler = {"metrics": METRICS, "inputs": "mean_max", "recenter": {"windows": RECENTER},
                 "entities": {m: {"mean": [0.0] * 76, "std": [1.0] * 76} for m in MACHINES}}
    sj_models = {"ae": SJ.ae_bundle(AE.DenseAE(76).eval(), ae_scaler, {"mode": "global", "global": 1.0}),
                 "tier1": None, "baselines_module": SB, "baselines": models}
    n_in = len(sj_models["ae"]["feature_metrics"])
    for machine in MACHINES:
        st, outs = SJ.new_state(n_in), []
        df = rows[machine].drop(columns="entity_id")
        for start in range(0, len(df), chunk):
            st = SJ.decode_state(SJ.encode_state(st), n_in)   # as GroupState stores it between batches
            out, _ = SJ.step(machine, df.iloc[start:start + chunk], st, sj_models)
            if len(out):
                outs.append(out)
        out = pd.concat(outs, ignore_index=True)
        test = out[out["split"] == "test"].set_index("ws_us")
        ref = result["test"][result["test"]["entity_id"] == machine].set_index("ws_us").loc[test.index]
        assert len(test) == N_TEST // 5 - 3          # the last 3 windows stay open (watermark)
        np.testing.assert_allclose(test["tier2_score"], ref["tier2_score"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(test["pca_score"], ref["pca_score"], rtol=0, atol=1e-9)
        assert test["tier2_flag"].tolist() == ref["tier2_flag"].astype(bool).tolist()


def test_parity_compare():
    """experiments/baselines_parity.compare: duplicates removed, missing windows and diffs counted."""
    from experiments import baselines_parity as P
    ws = pd.to_datetime([BASE_US, BASE_US + 300_000_000, BASE_US + 600_000_000], unit="us", utc=True)
    batch = pd.DataFrame({"entity_id": "m", "window_start": ws, "tier2_score": [1.0, 2.0, 3.0],
                          "tier2_flag": [False, True, True], "pca_score": [0.1, 0.2, 0.3]})
    stream = pd.concat([batch.iloc[:2], batch.iloc[1:2]]).assign(tier2_score=[1.0, 2.5, 2.0])
    row = P.compare(stream, batch).iloc[-1]
    assert (row["stream"], row["matched"], row["missing"]) == (2, 2, 1)
    assert row["max_d_tier2"] == 0.0 and row["max_d_pca"] == 0.0 and row["flag_identical_%"] == 100


@pytest.mark.parametrize("kept", [0, RECENTER - 1])
def test_history_too_short_raises(batch, kept):
    root = batch[0]
    with pytest.raises(ValueError, match=f"previous {RECENTER} windows.*only {kept} previous"):
        SB.load_models(root / "models/kmeans_smd", root / "models/pca_smd", history_windows=kept)
    for ok in (RECENTER, RECENTER + 10, None):
        assert SB.load_models(root / "models/kmeans_smd", root / "models/pca_smd", history_windows=ok)


def _save_ae(model_dir, recenter: int):
    """A tiny AE folder in the layout autoencoder.run() writes, with the given recenter."""
    model_dir.mkdir(parents=True)
    model = AE.DenseAE(76)
    torch.save({"state_dict": model.state_dict(), "n_in": 76, "hidden": list(AE.HIDDEN)}, model_dir / "model.pt")
    scaler = {"metrics": METRICS, "inputs": "mean_max", "recenter": {"windows": recenter} if recenter else None,
              "entities": {}}
    (model_dir / "scaler.json").write_text(json.dumps(scaler))
    (model_dir / "threshold.json").write_text(json.dumps({"mode": "global", "global": 1.0}))
    return model_dir


@pytest.mark.parametrize("ae_recenter, fails", [(RECENTER, False), (RECENTER - 2, True), (0, True)])
def test_score_job_checks_history_at_startup(batch, monkeypatch, tmp_path, ae_recenter, fails):
    """score_job.load_models passes the AE recenter (hist_x length) to stream_baselines.load_models."""
    root = batch[0]
    monkeypatch.setattr(SB, "load_models", functools.partial(
        SB.load_models, root / "models/kmeans_smd", root / "models/pca_smd"))
    ae_dir = _save_ae(tmp_path / "ae_smd", ae_recenter)
    if fails:
        with pytest.raises(ValueError, match="tier2/pca scores would drift"):
            SJ.load_models(ae_dir, tier1="off")
    else:
        assert SJ.load_models(ae_dir, tier1="off")["baselines"]["recenter"] == RECENTER
