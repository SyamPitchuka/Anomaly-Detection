from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from src.detection import tier1_zscore as T

N_TRAIN, N_TEST = 200, 50                 # rows (minutes) per machine -> 40 train + 10 test windows
BASE_US = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
MACHINES = ["machine-1-1", "machine-1-2"]
ANOMALY_ROWS = {"machine-1-1": range(20, 25), "machine-1-2": range(30, 35)}   # test windows 4 and 6
ANOMALY_WINDOW = {"machine-1-1": 4, "machine-1-2": 6}


def _machine_table(machine: str, seed: int) -> pa.Table:
    """m00 noisy (break +50 / +80 in the test part), m01 sparse so its train MAD is 0,
    m02 constant (MAD 0 and mean absolute deviation 0)."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    m00 = 10 * seed + rng.normal(size=n)
    m01 = np.zeros(n)
    m01[0:N_TRAIN:50] = 1.0                 # 4 train windows with mean 0.2, all other windows 0
    m02 = np.full(n, 5.0)
    label = np.zeros(N_TEST, dtype=np.int32)
    for i in ANOMALY_ROWS[machine]:
        label[i] = 1
        m00[N_TRAIN + i] += 50 if machine == "machine-1-1" else 80
    rows_metrics = [[("m00", a), ("m01", b), ("m02", c)] for a, b, c in zip(m00, m01, m02)]
    rows_metrics[N_TRAIN + 3] = rows_metrics[N_TRAIN + 3][:1]       # missing values are left out
    split = ["train"] * N_TRAIN + ["test"] * N_TEST
    return pa.Table.from_arrays(
        [
            pa.array([machine] * n, type=pa.string()),
            pa.array(BASE_US + np.arange(n, dtype=np.int64) * 60_000_000, type=pa.int64())
            .cast(pa.timestamp("us", tz="UTC")),
            pa.array(rows_metrics, type=pa.map_(pa.string(), pa.float64())),
            pa.array([[("split", s)] for s in split], type=pa.map_(pa.string(), pa.string())),
            pa.concat_arrays([pa.nulls(N_TRAIN, pa.int32()), pa.array(label, type=pa.int32())]),
        ],
        schema=S.arrow_schema("raw"),
    )


@pytest.fixture(scope="module")
def spark():
    session = T.build_spark("test_tier1_zscore")
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
    pq.write_table(_machine_table(MACHINES[0], 9).slice(0, 10), raw / "fake.parquet")   # must be skipped
    return root


@pytest.fixture(scope="module")
def result(repo, spark):
    return T.run("smd", root=repo, spark=spark)


def test_stats_match_numpy_on_train_windows(result, repo):
    stats = result["stats"].set_index(["entity_id", "metric"])
    assert len(stats) == len(MACHINES) * 3
    table = pq.read_table(repo / "data/raw/source=smd/machine-1-1.parquet").to_pylist()
    m00 = np.array([dict(r["metrics"])["m00"] for r in table[:N_TRAIN]]).reshape(-1, 5).mean(axis=1)
    median = np.median(m00)
    mad = np.median(np.abs(m00 - median))
    row = stats.loc[("machine-1-1", "m00")]
    assert row["median"] == pytest.approx(median)
    assert row["mad"] == pytest.approx(mad)
    assert row["scale"] == pytest.approx(1.4826 * mad)
    assert row["n_train"] == N_TRAIN // 5


def test_mad_zero_metrics_are_skipped(result):
    stats = result["stats"].set_index(["entity_id", "metric"])
    for machine in MACHINES:
        for metric in ("m01", "m02"):
            assert stats.loc[(machine, metric), "mad"] == 0
            assert np.isnan(stats.loc[(machine, metric), "scale"])      # not scored, as in the stream
        assert stats.loc[(machine, "m00"), "scale"] > 0
    assert np.isfinite(result["test"]["tier1_score"]).all()             # m00 is still scored everywhere


def test_mad_zero_fallback_scales(spark, repo):
    raw = spark.read.parquet((repo / "data/raw/source=smd/machine-1-1.parquet").as_posix())
    means, meta = T.build_windows(raw, S.SMD_METRICS)
    stats = T.fit_stats(means, meta, "fallback").toPandas().set_index("metric")
    mean_ad = 4 * 0.2 / (N_TRAIN // 5)          # median 0, so |x - median| is 0.2 in 4 of 40 windows
    assert stats.loc["m01", "scale"] == pytest.approx(T.MEAN_AD_TO_SIGMA * mean_ad)
    assert stats.loc["m02", "scale"] == T.SCALE_FLOOR


def test_planted_anomalies_are_flagged(result):
    test = result["test"]
    assert len(test) == len(MACHINES) * N_TEST // 5
    for machine in MACHINES:
        own = test[test["entity_id"] == machine].reset_index(drop=True)
        assert own["label"].tolist() == [int(w == ANOMALY_WINDOW[machine]) for w in range(N_TEST // 5)]
        assert own["tier1_flag"].tolist() == own["label"].astype(bool).tolist()
    assert (test["tier1_flag"] == (test["tier1_score"] > 3)).all()
    assert result["overall"]["f1"] == pytest.approx(1.0)


def test_scores_written(result, repo):
    dset = ds.dataset(repo / "data/scores_tier1", format="parquet", partitioning="hive")
    got = {f.name: f.type for f in dset.schema}
    assert set(got) == {"entity_id", "window_start", "window_end", "tier1_score", "tier1_flag", "label", "source"}
    for name in got.keys() - {"source"}:
        assert S.type_matches(S.LAYERS["scores"][name], got[name]), name
    assert len(dset.to_table()) == len(MACHINES) * N_TEST // 5


def test_baselines_file_matches_stream_contract(result, repo):
    path = repo / "models/tier1_smd/baselines.parquet"
    assert [p.name for p in path.parent.iterdir()] == ["baselines.parquet"]      # one file, nothing else
    table = pq.read_table(path)
    assert table.schema.names == ["entity_id", "metric", "median", "mad"]
    assert pa.types.is_string(table.schema.field("entity_id").type) and pa.types.is_float64(table.schema.field("mad").type)
    df = table.to_pandas()
    assert len(df) == len(MACHINES) * 3 and set(df["metric"]) == {"m00", "m01", "m02"}
    assert df[df["metric"] == "m00"]["mad"].gt(0).all()
