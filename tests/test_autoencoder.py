import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
