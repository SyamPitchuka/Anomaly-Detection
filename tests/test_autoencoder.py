import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from contracts import schemas as S
from src.detection import autoencoder as M

N_TRAIN, N_TEST = 1500, 300               # rows (minutes) per machine
ANOMALY_ROWS = range(150, 165)            # test rows 150..164 -> 3 full windows
BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000


def _machine_table(machine: str, seed: int) -> pa.Table:
    """38 metrics driven by 3 latent signals (so the AE can learn them); planted break in the test part."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    t = np.arange(n, dtype=np.int64)  # int32 on Windows would overflow in t * 60_000_000
    latent = np.stack([np.sin(t / 30), np.cos(t / 50), np.sin(t / 17 + seed)], axis=1)
    values = latent @ rng.normal(size=(3, 38)) + 0.05 * rng.normal(size=(n, 38)) + 10 * seed
    label = np.zeros(N_TEST, dtype=np.int32)
    for i in ANOMALY_ROWS:
        values[N_TRAIN + i, :6] += 8.0     # breaks the latent structure on 6 metrics
        label[i] = 1

    keys = [S.SMD_METRICS] * n
    rows_metrics = [list(zip(k, v)) for k, v in zip(keys, values.tolist())]
    rows_metrics[N_TRAIN + 10] = [kv for kv in rows_metrics[N_TRAIN + 10] if kv[0] != "m04"]  # missing value
    split = ["train"] * N_TRAIN + ["test"] * N_TEST
    return pa.Table.from_arrays(
        [
            pa.array([machine] * n, type=pa.string()),
            pa.array(BASE_US + t * 60_000_000, type=pa.int64()).cast(pa.timestamp("us", tz="UTC")),
            pa.array(rows_metrics, type=pa.map_(pa.string(), pa.float64())),
            pa.array([[("split", s)] for s in split], type=pa.map_(pa.string(), pa.string())),
            pa.concat_arrays([pa.nulls(N_TRAIN, pa.int32()), pa.array(label, type=pa.int32())]),
        ],
        schema=S.arrow_schema("raw"),
    )


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    raw = root / "data/raw/source=smd"
    raw.mkdir(parents=True)
    for seed, machine in enumerate(["machine-1-1", "machine-1-2"], start=1):
        pq.write_table(_machine_table(machine, seed), raw / f"{machine}.parquet")
    pq.write_table(_machine_table("fake", 9).slice(0, 10), raw / "fake.parquet")  # must be skipped
    return root


@pytest.fixture(scope="module")
def result(repo):
    return M.run("smd", epochs=40, root=repo, verbose=False)


def test_build_windows():
    meta, X = M.build_windows(_machine_table("machine-1-1", 1), S.SMD_METRICS)
    assert len(meta) == (N_TRAIN + N_TEST) // 5
    assert (meta["split"] == "train").sum() == N_TRAIN // 5
    assert meta.loc[meta["split"] == "train", "label"].isna().all()
    test = meta[meta["split"] == "test"].reset_index(drop=True)
    assert test["label"].tolist() == [1.0 if 30 <= w < 33 else 0.0 for w in range(N_TEST // 5)]
    assert np.diff(meta["window_start"]).tolist() == [M.WINDOW_US] * (len(meta) - 1)
    assert not np.isnan(X).any()  # one missing m04 value in a window is averaged over the other 4 rows


def test_outputs(result, repo):
    model_dir = repo / "models/ae_smd"
    for name in ("model.pt", "scaler.json", "threshold.json"):
        assert (model_dir / name).exists()
    scalers = json.loads((model_dir / "scaler.json").read_text())
    assert set(scalers["entities"]) == {"machine-1-1", "machine-1-2"}
    assert scalers["std_floor"] == M.STD_FLOOR
    assert min(min(s["std"]) for s in scalers["entities"].values()) >= M.STD_FLOOR
    saved = json.loads((model_dir / "threshold.json").read_text())
    assert saved["mode"] == "global"
    assert saved["global"] == pytest.approx(result["thresholds"]["global"])
    assert saved["per_machine"] == pytest.approx(result["thresholds"]["per_machine"])

    table = pq.read_table(repo / "data/scores_ae/source=smd/scores.parquet")
    assert table.schema.equals(S.arrow_schema("scores"))
    assert table.num_rows == 2 * N_TEST // 5
    assert table["tier1_score"].null_count == table.num_rows
    assert table["is_anomaly"].to_pylist() == table["ae_flag"].to_pylist()
    assert all(len(m) == 3 for m in table["top_metrics"].to_pylist())


def test_planted_anomaly_scores_higher(result):
    meta, scores = result["meta"], result["ae_score"]
    test = (meta["split"] == "test").to_numpy()
    for entity in ("machine-1-1", "machine-1-2"):
        own = test & (meta["entity_id"] == entity).to_numpy()
        anomalous = scores[own & (meta["label"] == 1).to_numpy()]
        normal = scores[own & (meta["label"] == 0).to_numpy()]
        assert len(anomalous) == 3
        assert anomalous.min() > normal.max()
    assert result["overall"]["event"]["recall"] == pytest.approx(1.0)


def test_overall_sums_counts_over_machines():
    # machine a: tp=1; machine b: fn=1, fp=1. The segment at the end of a and the one at the start of b must not merge.
    meta = pd.DataFrame({"entity_id": ["a", "a", "b", "b"], "window_start": [0, 1, 0, 1],
                         "split": ["test"] * 4, "label": [0.0, 1.0, 1.0, 0.0]})
    per_machine, overall = M.evaluate(meta, np.array([False, True, False, True]))
    assert overall["point"]["precision"] == pytest.approx(0.5)
    assert overall["point"]["recall"] == pytest.approx(0.5)
    assert overall["event"]["recall"] == pytest.approx(0.5)
    assert per_machine["pt_f1"].tolist() == pytest.approx([1.0, 0.0])


def test_std_floor():
    meta = pd.DataFrame({"entity_id": ["a"] * 3, "split": ["train"] * 3})
    X = np.array([[0.0, 0.50, 1.0], [0.0, 0.51, 3.0], [0.0, 0.52, 5.0]])  # std 0, ~0.008, ~1.63
    floored = M.fit_scalers(meta, X, std_floor=0.05)["a"]["std"]
    assert floored[0] == 1.0                      # std 0 -> 1 rule is kept
    assert floored[1] == pytest.approx(0.05)      # tiny std raised to the floor
    assert floored[2] == pytest.approx(np.std(X[:, 2]))
    assert M.fit_scalers(meta, X, std_floor=0.0)["a"]["std"][1] == pytest.approx(np.std(X[:, 1]))


def test_thresholds_global_and_per_machine():
    meta = pd.DataFrame({"entity_id": ["a"] * 200 + ["b"] * 200 + ["a", "b"],
                         "split": ["train"] * 400 + ["test"] * 2})
    scores = np.concatenate([np.linspace(0, 1, 200), np.linspace(0, 10, 200), [2.0, 2.0]])
    info = M.thresholds(meta, scores)
    assert info["per_machine"]["a"] == pytest.approx(np.percentile(scores[:200], 99.5) * 1.1)
    assert info["per_machine"]["b"] == pytest.approx(np.percentile(scores[200:400], 99.5) * 1.1)
    assert info["global"] == pytest.approx(np.percentile(scores[:400], 99.5) * 1.1)
    # a test score of 2.0 is above machine a's threshold (1.1) but below machine b's (11)
    assert M.flags(meta, scores, info, "per-machine")[-2:].tolist() == [True, False]
    assert M.flags(meta, scores, info, "global")[-2:].tolist() == [False, False]


# ---------------------------------------------------------------- v2 options

def _zero_model(n_in: int) -> M.DenseAE:
    """Autoencoder that outputs 0, so the squared error of each input is Z ** 2."""
    model = M.DenseAE(n_in)
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    return model


def test_defaults_are_v2(result, repo):
    scalers = json.loads((repo / "models/ae_smd/scaler.json").read_text())
    assert scalers["inputs"] == "mean_max" and len(scalers["features"]) == 76
    assert scalers["flat_std"] == M.FLAT_STD == 0.0
    assert scalers["recenter"]["windows"] == 24 and scalers["recenter"]["min_periods"] == 12
    assert all(e["masked"] == [] for e in scalers["entities"].values())  # no flat metric in this fixture
    adaptive = json.loads((repo / "models/ae_smd/threshold.json").read_text())["adaptive"]
    assert adaptive == {"windows": 288, "percentile": 99, "min_periods": 144}
    assert result["overall"]["composite"]["events_detected"] == 2  # one planted fault per machine


def test_build_windows_mean_max():
    table = _machine_table("machine-1-1", 1)
    meta, X = M.build_windows(table, S.SMD_METRICS, ("mean", "max"))
    assert X.shape == (len(meta), 76)
    first_window = [dict(r)["m00"] for r in table["metrics"].to_pylist()[:5]]
    assert X[0, 0] == pytest.approx(np.mean(first_window))
    assert X[0, 38] == pytest.approx(np.max(first_window))
    assert M.feature_names(["a", "b"], ("mean", "max")) == ["a_mean", "b_mean", "a_max", "b_max"]
    assert M.feature_names(["a", "b"], ("mean",)) == ["a", "b"]


def test_flat_metric_is_masked_in_input_error_and_top_metrics():
    meta = pd.DataFrame({"entity_id": ["a"] * 4, "split": ["train"] * 3 + ["test"]})
    # metric x: train std 0.005 (flat), metric y: std 0.8; columns = [x_mean, y_mean, x_max, y_max]
    X = np.array([[0.50, 1.0, 0.51, 1.5], [0.51, 2.0, 0.52, 2.5], [0.50, 3.0, 0.51, 3.5], [5.0, 2.0, 5.0, 2.5]])
    flat = M.flat_metrics(meta, X, ["x", "y"], flat_std=0.01)
    assert flat == {"a": ["x"]}
    assert M.flat_metrics(meta, X, ["x", "y"], flat_std=0.0) == {"a": []}
    scalers = M.fit_scalers(meta, X)
    scalers["a"]["masked"] = flat["a"]
    Z, mask = M.standardise(meta, X, scalers, ["x", "y", "x", "y"])
    assert (Z[:, [0, 2]] == 0).all() and not mask[:, [0, 2]].any()  # the jump of x to 5.0 is invisible
    assert mask[:, [1, 3]].all()
    ae_score, top = M.score(_zero_model(4), Z, mask, ["x", "y"])
    assert all(t == ["y"] for t in top)
    assert ae_score[3] == pytest.approx((Z[3, 1] ** 2 + Z[3, 3] ** 2) / 2)


def test_top_metrics_add_mean_and_max_errors():
    # per-metric error = mean error + max error: a = 4 + 0, b = 2.25 + 2.25, c = 0 + 1
    Z = np.array([[2.0, 1.5, 0.0, 0.0, 1.5, 1.0]])
    ae_score, top = M.score(_zero_model(6), Z, np.ones_like(Z, dtype=bool), ["a", "b", "c"])
    assert top == [["b", "a", "c"]]
    assert ae_score[0] == pytest.approx((4 + 2.25 + 2.25 + 1) / 6)


def test_recenter_centers_rolling_median_causal():
    n_train, n_test = 4, 20
    meta = pd.DataFrame({"entity_id": ["a"] * (n_train + n_test),
                         "split": ["train"] * n_train + ["test"] * n_test})
    X = np.arange(n_train + n_test, dtype=float)[:, None]  # value = index
    scalers = {"a": {"mean": [100.0], "std": [1.0]}}
    c = M.recenter_centers(meta, X, scalers, windows=6).ravel()
    assert c[:n_train].tolist() == [100.0] * n_train          # train windows keep the train mean
    # index 4: previous values 0..3 (at least 3 needed) -> median 1.5; index 10: median of 4..9 = 6.5
    assert c[4] == pytest.approx(1.5)
    assert c[10] == pytest.approx(6.5)
    X[10, 0] = 1e6                                            # the current window never moves its own centre
    assert M.recenter_centers(meta, X, scalers, windows=6)[10, 0] == pytest.approx(6.5)
    short = pd.DataFrame({"entity_id": ["b"] * 3, "split": ["train", "test", "test"]})
    c = M.recenter_centers(short, np.array([[1.0], [2.0], [3.0]]), {"b": {"mean": [7.0], "std": [1.0]}}, 6)
    assert c.ravel().tolist() == [7.0, 7.0, 7.0]              # fewer than 3 previous values -> train mean


def test_adaptive_limits_and_flags():
    meta = pd.DataFrame({"entity_id": ["a"] * 5 + ["b"], "split": ["test"] * 6})
    scores = np.array([1.0, 2.0, 3.0, 10.0, 4.0, 100.0])
    limit = M.adaptive_limits(meta, scores, n=4)
    assert np.isnan(limit[[0, 1, 5]]).all()                 # fewer than 2 previous scores of the same machine
    assert limit[2] == pytest.approx(1.99)                    # p99 of [1, 2]
    assert limit[3] == pytest.approx(2.98)                    # p99 of [1, 2, 3]
    assert limit[4] == pytest.approx(9.79)                    # p99 of [1, 2, 3, 10]
    info = {"global": 0.5}
    assert M.flags(meta, scores, info, "global").all()
    assert M.flags(meta, scores, info, "global", limit).tolist() == [True, True, True, True, False, True]


DRIFT_TRAIN, DRIFT_TEST = 3000, 300       # rows; test windows 0..59
SHIFT_WINDOW, SPIKE_WINDOW = 10, 50       # lasting level shift in m37 from test window 10, one-window spike at 50


def _drift_table(machine: str = "machine-9-1", seed: int = 3) -> pa.Table:
    """Stationary metrics plus a flat m37 (0.5 + tiny noise) that shifts up for good in the test part."""
    rng = np.random.default_rng(seed)
    n = DRIFT_TRAIN + DRIFT_TEST
    t = np.arange(n, dtype=np.int64)
    values = rng.normal(size=(n, 3)) @ rng.normal(size=(3, 38)) * 0.2 + 0.02 * rng.normal(size=(n, 38)) + 1.0
    values[:, 37] = 0.5 + 0.001 * rng.normal(size=n)
    values[DRIFT_TRAIN + 5 * SHIFT_WINDOW:, 37] += 0.3
    values[DRIFT_TRAIN + 5 * SPIKE_WINDOW:DRIFT_TRAIN + 5 * SPIKE_WINDOW + 5, 37] += 0.3
    split = ["train"] * DRIFT_TRAIN + ["test"] * DRIFT_TEST
    return pa.Table.from_arrays(
        [
            pa.array([machine] * n, type=pa.string()),
            pa.array(BASE_US + t * 60_000_000, type=pa.int64()).cast(pa.timestamp("us", tz="UTC")),
            pa.array([list(zip(S.SMD_METRICS, v)) for v in values.tolist()],
                     type=pa.map_(pa.string(), pa.float64())),
            pa.array([[("split", s)] for s in split], type=pa.map_(pa.string(), pa.string())),
            pa.concat_arrays([pa.nulls(DRIFT_TRAIN, pa.int32()),
                              pa.array(np.zeros(DRIFT_TEST, dtype=np.int32), type=pa.int32())]),
        ],
        schema=S.arrow_schema("raw"),
    )


@pytest.fixture(scope="module")
def drift_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("drift")
    raw = root / "data/raw/source=smd"
    raw.mkdir(parents=True)
    pq.write_table(_drift_table(), raw / "machine-9-1.parquet")
    return root


def _test_flags(res) -> np.ndarray:
    return res["ae_flag"][(res["meta"]["split"] == "test").to_numpy()]


def test_recenter_absorbs_level_shift_but_keeps_spike(drift_repo):
    # flat_std 0 keeps m37 in the model and adaptive 0 keeps the plain threshold, so only recentering
    # can stop the shift from being flagged
    res = M.run("smd", epochs=30, root=drift_repo, verbose=False, flat_std=0.0, recenter=24, adaptive=0)
    f = _test_flags(res)
    assert f[SHIFT_WINDOW]                                    # the shift itself raises an alarm
    assert not f[SHIFT_WINDOW + 24:SPIKE_WINDOW].any()        # ... but not after the recenter period
    assert f[SPIKE_WINDOW]                                    # a sudden spike is still flagged
    assert not f[SPIKE_WINDOW + 1:].any()


def test_without_recenter_level_shift_is_flagged_forever(drift_repo):
    res = M.run("smd", epochs=30, root=drift_repo, verbose=False, flat_std=0.0, recenter=0, adaptive=0)
    assert _test_flags(res)[SHIFT_WINDOW:].all()


def test_flat_std_masks_the_drifting_flat_metric(drift_repo):
    res = M.run("smd", epochs=30, root=drift_repo, verbose=False, flat_std=0.01, recenter=0, adaptive=0)
    assert res["scalers"]["machine-9-1"]["masked"] == ["m37"]
    saved = json.loads((drift_repo / "models/ae_smd/scaler.json").read_text())
    assert saved["entities"]["machine-9-1"]["masked"] == ["m37"]
    assert saved["recenter"] is None
    # m37 is out of the error: the shift is not flagged (at most a stray noise alarm) and never named
    assert _test_flags(res)[SHIFT_WINDOW:].sum() <= 1
    top = pq.read_table(drift_repo / "data/scores_ae/source=smd/scores.parquet")["top_metrics"].to_pylist()
    assert not any("m37" in t for t in top)
