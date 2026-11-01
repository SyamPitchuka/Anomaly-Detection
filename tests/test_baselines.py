import json
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from pyspark.ml import PipelineModel

from contracts import schemas as S
from src.detection import baselines as M

N_TRAIN, N_TEST = 400, 100                # rows (minutes) per machine -> 80 train + 20 test windows
ANOMALY_ROWS = range(50, 60)              # test rows 50..59 -> test windows 10 and 11
BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
WINDOW_US = 5 * 60 * 1_000_000
MACHINES = ["machine-1-1", "machine-1-2"]


def _machine_table(machine: str, seed: int) -> pa.Table:
    """38 metrics driven by 3 latent signals plus noise; a +8 break on 6 metrics in the test part."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    t = np.arange(n, dtype=np.int64)
    latent = np.stack([np.sin(t / 30), np.cos(t / 50), np.sin(t / 17 + seed)], axis=1)
    values = latent @ rng.normal(size=(3, 38)) + 0.05 * rng.normal(size=(n, 38)) + 10 * seed
    label = np.zeros(N_TEST, dtype=np.int32)
    for i in ANOMALY_ROWS:
        values[N_TRAIN + i, :6] += 8.0
        label[i] = 1
    rows_metrics = [list(zip(S.SMD_METRICS, v)) for v in values.tolist()]
    rows_metrics[N_TRAIN + 3] = [kv for kv in rows_metrics[N_TRAIN + 3] if kv[0] != "m04"]  # missing value
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


def _fake_ae_scores() -> pa.Table:
    """AE scores for machine-1-1's test windows only; ae_flag true on its first test window."""
    n = N_TEST // 5
    start = BASE_US + N_TRAIN * 60_000_000 + np.arange(n, dtype=np.int64) * WINDOW_US
    schema = S.arrow_schema("scores")
    cols = {
        "entity_id": pa.array(["machine-1-1"] * n, type=pa.string()),
        "window_start": pa.array(start).cast(schema.field("window_start").type),
        "window_end": pa.array(start + WINDOW_US).cast(schema.field("window_end").type),
        "ae_score": pa.array(np.linspace(0, 1, n)),
        "ae_flag": pa.array([True] + [False] * (n - 1)),
        "top_metrics": pa.array([["m00", "m01", "m02"]] * n, type=pa.list_(pa.string())),
    }
    return pa.Table.from_arrays([cols.get(f.name, pa.nulls(n, type=f.type)) for f in schema], schema=schema)


@pytest.fixture(scope="module")
def spark():
    session = M.build_spark("test_baselines")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    raw = root / "data/raw/source=smd"
    raw.mkdir(parents=True)
    for seed, machine in enumerate(MACHINES, start=1):
        pq.write_table(_machine_table(machine, seed), raw / f"{machine}.parquet")
    pq.write_table(_machine_table("fake", 9).slice(0, 10), raw / "fake.parquet")  # must be skipped
    ae = root / "data/scores_ae/source=smd"
    ae.mkdir(parents=True)
    pq.write_table(_fake_ae_scores(), ae / "scores.parquet")
    return root


@pytest.fixture(scope="module")
def result(repo, spark):
    return M.run("smd", kmax=5, root=repo, spark=spark)


def test_build_windows(spark, repo):
    raw = spark.read.parquet((repo / "data/raw/source=smd/machine-1-1.parquet").as_posix())
    w = M.build_windows(raw, S.SMD_METRICS).orderBy("ws_us").toPandas()
    assert len(w) == (N_TRAIN + N_TEST) // 5
    assert (w["split"] == "train").sum() == N_TRAIN // 5
    assert w.loc[w["split"] == "train", "label"].isna().all()
    test = w[w["split"] == "test"].reset_index(drop=True)
    assert test["label"].tolist() == [1 if w_ in (10, 11) else 0 for w_ in range(N_TEST // 5)]
    assert np.diff(w["ws_us"]).tolist() == [WINDOW_US] * (len(w) - 1)
    assert w["ws_us"].iloc[0] == BASE_US
    assert not w[S.SMD_METRICS].isna().any().any()   # the missing m04 is averaged over the other 4 rows
    # same window mean as numpy over the raw rows
    table = pq.read_table(repo / "data/raw/source=smd/machine-1-1.parquet")
    m00 = [dict(r)["m00"] for r in table["metrics"].to_pylist()[:5]]
    assert w["m00"].iloc[0] == pytest.approx(np.mean(m00))


def test_planted_anomaly_scores_higher(result):
    test = result["test"]
    for machine in MACHINES:
        own = test[test["entity_id"] == machine]
        for col in ("tier2_score", "pca_score"):
            anomalous, normal = own.loc[own["label"] == 1, col], own.loc[own["label"] == 0, col]
            assert len(anomalous) == 2
            assert anomalous.min() > normal.max(), (machine, col)
    for name in ("Tier 2 KMeans", "Tier 2b PCA"):
        assert result["results"][name]["composite"]["events_detected"] == 2


def test_model_choice(result):
    assert M.K_MIN <= result["k"] <= 5
    assert set(result["silhouette"]) == set(range(M.K_MIN, 6))
    assert 1 <= result["n_components"] <= len(S.SMD_METRICS)


def test_models_saved(result, repo):
    for name, score in (("kmeans_smd", "tier2_score"), ("pca_smd", "pca_score")):
        model_dir = repo / "models" / name
        model = PipelineModel.load(model_dir.as_posix())
        assert len(model.stages) == 2
        info = json.loads((model_dir / "thresholds.json").read_text())
        assert info["global"] == pytest.approx(result["thresholds"][score])
        assert info["percentile"] == 99.5 and info["factor"] == 1.1
        scaler = json.loads((model_dir / "scaler.json").read_text())
        assert set(scaler["entities"]) == set(MACHINES)
        assert min(min(e["std"]) for e in scaler["entities"].values()) > 0


def test_flags_follow_thresholds(result):
    test = result["test"]
    for score, flag in (("tier2_score", "tier2_flag"), ("pca_score", "pca_flag")):
        assert (test[flag] == (test[score] > result["thresholds"][score])).all()


def test_scores_layer(result, repo):
    dset = ds.dataset(repo / "data/scores", format="parquet", partitioning="hive")
    got = {f.name: f.type for f in dset.schema}
    assert list(got)[:len(S.LAYERS["scores"])] == list(S.LAYERS["scores"])
    assert set(got) - set(S.LAYERS["scores"]) <= {"source"}
    for name, code in S.LAYERS["scores"].items():
        assert S.type_matches(code, got[name]), name
    table = dset.to_table().to_pandas().sort_values(["entity_id", "window_start"]).reset_index(drop=True)
    assert len(table) == len(MACHINES) * N_TEST // 5
    assert table["tier1_score"].isna().all() and table["tier1_flag"].isna().all()   # no scores_tier1 folder
    m11 = table["entity_id"] == "machine-1-1"
    assert table.loc[m11, "ae_score"].notna().all() and table.loc[~m11, "ae_score"].isna().all()
    # is_anomaly = ae_flag OR tier1_flag, nulls as false: only machine-1-1's first test window
    assert table["is_anomaly"].sum() == 1
    assert bool(table.loc[m11, "is_anomaly"].iloc[0])
    assert table.loc[m11, "top_metrics"].iloc[0].tolist() == ["m00", "m01", "m02"]
