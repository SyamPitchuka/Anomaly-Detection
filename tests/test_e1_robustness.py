"""E1 robustness on a tiny fixture: 2 machines x 100 raw rows (60 train + 40 test), plus a 2-machine scores layer."""
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from experiments import e1_robustness as E
from src.detection import autoencoder as M

N_TRAIN, N_TEST = 60, 40                  # raw rows (minutes) per machine -> 12 train + 8 test windows
ANOMALY_ROWS = range(10, 20)              # test rows 10..19 -> test windows 2 and 3
BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
MACHINES = ["machine-1-1", "machine-1-2"]
REPO = Path(__file__).resolve().parents[1]
PCA_THR = 0.5


def _raw_machine(machine: str, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    t = np.arange(n, dtype=np.int64)
    values = np.outer(np.sin(t / 7), rng.normal(size=38)) + 0.05 * rng.normal(size=(n, 38)) + 5 * seed
    label = np.zeros(N_TEST, dtype=np.int32)
    for i in ANOMALY_ROWS:
        values[N_TRAIN + i, :5] += 6.0
        label[i] = 1
    rows = [list(zip(S.SMD_METRICS, v)) for v in values.tolist()]
    split = ["train"] * N_TRAIN + ["test"] * N_TEST
    return pa.Table.from_arrays(
        [pa.array([machine] * n, type=pa.string()),
         pa.array(BASE_US + t * 60_000_000, type=pa.int64()).cast(pa.timestamp("us", tz="UTC")),
         pa.array(rows, type=pa.map_(pa.string(), pa.float64())),
         pa.array([[("split", s)] for s in split], type=pa.map_(pa.string(), pa.string())),
         pa.concat_arrays([pa.nulls(N_TRAIN, pa.int32()), pa.array(label, type=pa.int32())])],
        schema=S.arrow_schema("raw"))


def _scores_machine(entity: str, labels, tier2, pca, ae, start: int) -> pd.DataFrame:
    n = len(labels)
    t0 = datetime(2026, 1, 20, tzinfo=timezone.utc) + timedelta(minutes=start)
    ws = [t0 + timedelta(minutes=5 * i) for i in range(n)]
    rng = np.random.default_rng(start)
    ae = np.array(ae, bool)
    return pd.DataFrame({
        "entity_id": entity, "window_start": ws, "window_end": [w + timedelta(minutes=5) for w in ws],
        "tier1_score": rng.random(n), "tier1_flag": np.zeros(n, bool),
        "tier2_score": rng.random(n), "tier2_flag": np.array(tier2, bool),
        "pca_score": np.array(pca, float), "ae_score": rng.random(n) + ae, "ae_flag": ae,
        "is_anomaly": ae.copy(), "top_metrics": [["m00", "m01", "m02"]] * n, "label": np.array(labels, np.int32)})


def _bits(n, *idx):
    a = np.zeros(n, int)
    a[list(idx)] = 1
    return a


def scores_frame() -> pd.DataFrame:
    n = 20
    a = _scores_machine("machine-a", _bits(n, 3, 4, 5, 12, 13), tier2=_bits(n, 4, 9),
                        pca=_bits(n, 5, 6, 13) * 0.9 + 0.1, ae=_bits(n, 3, 17), start=0)
    b = _scores_machine("machine-b", _bits(n, 0, 1, 15), tier2=_bits(n, 1, 15), pca=_bits(n, 0) * 0.9 + 0.1,
                        ae=_bits(n, 1, 2), start=7)
    return pd.concat([a, b], ignore_index=True)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    raw = root / S.PATHS["raw"] / "source=smd"
    raw.mkdir(parents=True)
    for seed, m in enumerate(MACHINES, start=1):
        pq.write_table(_raw_machine(m, seed), raw / f"{m}.parquet")
    scores = root / S.PATHS["scores"] / "source=smd"
    scores.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(scores_frame(), schema=S.arrow_schema("scores"), preserve_index=False),
                   scores / "part-0.parquet")
    (root / "models/pca_smd").mkdir(parents=True)
    (root / "models/pca_smd/thresholds.json").write_text(json.dumps({"mode": "global", "global": PCA_THR}))
    (root / "docs/results").mkdir(parents=True)
    shutil.copy(REPO / "docs/results/e1.json", root / "docs/results/e1.json")
    return root


@pytest.fixture(scope="module")
def windows(repo):
    return M.load_windows(repo / S.PATHS["raw"] / "source=smd", M.METRICS["smd"], None, M.INPUTS["mean_max"])


def test_holdout_masks(windows):
    meta, _ = windows
    fit, held = E.holdout_masks(meta, 0.8)
    assert not (fit & held).any()
    assert not ((fit | held) & (meta["split"] == "test").to_numpy()).any()
    for _, g in meta.assign(fit=fit, held=held).groupby("entity_id"):
        assert g["fit"].sum() == 9 and g["held"].sum() == 3      # floor(0.8 * 12) = 9 train windows fit
        assert g.loc[g["fit"], "window_start"].max() < g.loc[g["held"], "window_start"].min()


def test_fit_and_score_matches_autoencoder_run(repo, windows, tmp_path):
    """With the full train split, the step-by-step pipeline gives exactly what autoencoder.run gives."""
    meta, X = windows
    train = (meta["split"] == "train").to_numpy()
    mine = E.fit_and_score(meta, X, seed=3, fit_mask=train, threshold_mask=train, epochs=3)
    ref = M.run("smd", epochs=3, root=repo, verbose=False, seed=3, model_dir=tmp_path / "m",
                scores_dir=tmp_path / "s")
    np.testing.assert_array_equal(mine["ae_score"], ref["ae_score"])
    np.testing.assert_array_equal(mine["flags"]["global"], ref["ae_flag"])
    assert mine["thresholds"]["global"] == ref["thresholds"]["global"]
    assert (tmp_path / "m/model.pt").exists() and (tmp_path / "s/scores.parquet").exists()
    assert not (repo / "models/ae_smd").exists() and not (repo / "data/scores_ae").exists()


def test_seed_changes_model(repo, windows):
    meta, X = windows
    train = (meta["split"] == "train").to_numpy()
    a = E.fit_and_score(meta, X, seed=0, fit_mask=train, threshold_mask=train, epochs=2)
    b = E.fit_and_score(meta, X, seed=1, fit_mask=train, threshold_mask=train, epochs=2)
    a2 = E.fit_and_score(meta, X, seed=0, fit_mask=train, threshold_mask=train, epochs=2)
    np.testing.assert_array_equal(a["ae_score"], a2["ae_score"])
    assert not np.array_equal(a["ae_score"], b["ae_score"])


def test_holdout_threshold_uses_only_held_out_scores(windows):
    meta, X = windows
    fit, held = E.holdout_masks(meta)
    res = E.fit_and_score(meta, X, seed=0, fit_mask=fit, threshold_mask=held, epochs=2)
    s = res["ae_score"]
    assert res["thresholds"]["global"] == pytest.approx(np.percentile(s[held], 99.5) * 1.1)
    for entity, idx in meta.groupby("entity_id").indices.items():
        own = s[idx][held[idx]]
        assert res["thresholds"]["per_machine"][entity] == pytest.approx(np.percentile(own, 99.5) * 1.1)


def test_counts_give_evaluate_totals():
    df = scores_frame()
    meta = pd.DataFrame({"entity_id": df["entity_id"], "window_start": df["window_start"], "split": "test",
                         "label": df["label"].astype(float)})
    for key, flag in E.detector_flags(df, PCA_THR).items():
        _, overall = M.evaluate(meta, flag, None)
        machines, c = E.machine_counts(df, flag)
        got = E.from_counts(c.sum(axis=0))
        assert machines == ["machine-a", "machine-b"]
        assert float(got["composite_f1"]) == pytest.approx(overall["composite"]["composite_f1"])
        assert float(got["false_alarms_per_machine_day"]) == pytest.approx(overall["fa_day"])
        assert int(c[:, 1].sum()) == overall["composite"]["events_detected"]


def test_bootstrap_paired_and_seeded():
    c = np.array([[2, 1, 3, 1, 20], [1, 1, 1, 4, 20], [3, 0, 0, 0, 20]])
    b1 = E.bootstrap({"x": c, "y": c.copy()}, n_boot=50, seed=0)
    b2 = E.bootstrap({"x": c, "y": c.copy()}, n_boot=50, seed=0)
    assert b1["x"]["composite_f1"].shape == (50,)
    np.testing.assert_array_equal(b1["x"]["composite_f1"], b1["y"]["composite_f1"])   # same draw for both
    np.testing.assert_array_equal(b1["x"]["composite_f1"], b2["x"]["composite_f1"])   # seeded
    assert len(np.unique(b1["x"]["composite_f1"])) > 1


def test_raw_missing_prints_command(tmp_path, capsys):
    assert E.run(tmp_path, parts=("seeds",), seeds=(0,)) is None
    assert E.RAW_BUILD_COMMAND in capsys.readouterr().out


def test_end_to_end_sample(repo, capsys):
    doc = E.run(repo, seeds=(0, 1), sample=2, epochs=2, n_boot=40)
    out = capsys.readouterr().out
    assert out.count("[seeds]   seed") == 2 and out.count("[holdout] seed") == 2 and "total runtime" in out
    assert [r["seed"] for r in doc["seeds"]["rows"]] == [0, 1]
    assert len(doc["holdout"]["global"]) == 2 and len(doc["holdout"]["per-machine"]) == 2
    assert doc["bootstrap"]["machines"] == 2 and doc["bootstrap"]["n_boot"] == 40
    report = repo / "data/robust/sample2"
    text = (report / "e1_robustness.md").read_text(encoding="utf-8")
    assert "## 1. Seeds" in text and "## 2. Held-out threshold" in text and "## 3. Bootstrap" in text
    assert "Runtime: seeds" in text and "; total" in text
    assert "Does the E1 ranking survive?" in text and "false-alarm advantage survive?" in text
    assert json.loads((report / "e1_robustness.json").read_text())["bootstrap"]["pca_threshold"] == PCA_THR
    for name in ("ae_seed0_sample2", "ae_holdout_seed1_sample2"):
        assert (repo / "models/robust" / name / "model.pt").exists()
        assert (repo / "data/robust" / name / "scores/source=smd/scores.parquet").exists()
    # production paths and docs/results untouched
    assert not (repo / "models/ae_smd").exists() and not (repo / "data/scores_ae").exists()
    assert not (repo / "docs/results/e1_robustness.md").exists()
    assert sorted(p.name for p in (repo / "data/scores/source=smd").iterdir()) == ["part-0.parquet"]
    # report-only rebuilds the same file from the saved parts
    E.run(repo, sample=2, report_only=True)
    assert "## 3. Bootstrap" in (report / "e1_robustness.md").read_text(encoding="utf-8")
