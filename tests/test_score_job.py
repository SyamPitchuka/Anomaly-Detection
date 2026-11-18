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


class _FakeGroupState:
    """The part of GroupState that score_group uses, holding the tuple form between calls."""

    def __init__(self):
        self.value = None

    @property
    def exists(self):
        return self.value is not None

    @property
    def get(self):
        return self.value

    def update(self, value):
        self.value = value


def _run_score_group(rows: pd.DataFrame, machine: str, chunk: int) -> pd.DataFrame:
    state, outs = _FakeGroupState(), []
    for start in range(0, len(rows), chunk):
        part = rows.iloc[start:start + chunk]
        # split each micro-batch into pieces, as several read tasks / Arrow batches hand them to Spark
        pieces = [part.iloc[i:i + 5] for i in range(0, len(part), 5)]
        outs += list(M.score_group((machine,), iter(pieces), state))
    return pd.concat(outs, ignore_index=True)


@pytest.mark.parametrize("chunk", [37, 10_000])
def test_score_group_ignores_row_order(setup, monkeypatch, chunk):
    """Rows of a micro-batch in any order (Kafka order, minPartitions split, shuffle) give the same output."""
    raw, _, models = setup
    monkeypatch.setattr(M, "get_models", lambda: models)
    rows = raw[raw["entity_id"] == MACHINES[0]].drop(columns="entity_id").sort_values("event_us")
    rows = rows.reset_index(drop=True)
    want = _run_score_group(rows, MACHINES[0], chunk)
    # shuffle inside each micro-batch only: the batches themselves still arrive in offset order
    shuffled = pd.concat([rows.iloc[s:s + chunk].sample(frac=1.0, random_state=s)
                          for s in range(0, len(rows), chunk)])
    assert not shuffled["event_us"].is_monotonic_increasing
    got = _run_score_group(shuffled, MACHINES[0], chunk)
    assert len(want) > 0
    pd.testing.assert_frame_equal(got, want, check_exact=True)


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


def test_tier1_switch(monkeypatch, tmp_path):
    """--tier1 off (default) loads no baselines even if the file exists; v1 loads them; bad values fail."""
    path = tmp_path / "baselines_v1.parquet"
    pd.DataFrame({"entity_id": ["machine-x"], "metric": ["m00"], "median": [1.0], "mad": [1.0]}).to_parquet(path)
    monkeypatch.setattr(M.AE, "load_model", lambda d: ("model", "scaler", "threshold"))
    monkeypatch.setattr(M, "ae_bundle", lambda *a: {"fake": True})
    monkeypatch.setattr(M, "BASELINES_MODULE", tmp_path / "missing.py")
    monkeypatch.delenv(M.TIER1_ENV, raising=False)
    assert M.tier1_mode() == "off"
    assert M.load_models(tmp_path, path)["tier1"] is None
    monkeypatch.setenv(M.TIER1_ENV, "v1")
    assert M.tier1_mode() == "v1"
    assert set(M.load_models(tmp_path, path)["tier1"]) == {"machine-x"}
    assert M.load_models(tmp_path, path, tier1="off")["tier1"] is None
    monkeypatch.setenv(M.TIER1_ENV, "on")
    with pytest.raises(ValueError):
        M.tier1_mode()
    assert M.TIER1_PATH.name == "baselines_v1.parquet"


def test_write_scores_file(setup, tmp_path):
    raw, _, models = setup
    out = _stream(raw, models, 10_000)
    test = out[out["split"] == "test"].assign(is_anomaly=lambda d: d["ae_flag"].astype(bool))
    path = M.write_scores_file(test.drop(columns="split"), 3, tmp_path)
    table = pq.read_table(path)
    assert table.schema.names == list(S.LAYERS["scores"])
    assert all(S.type_matches(S.LAYERS["scores"][f.name], f.type) for f in table.schema)
    assert table.num_rows == len(test)


def test_model_entity_maps_replicas_to_their_machine():
    known = {"machine-1-1": 1, "machine-1-1-r2x": 1}
    assert M.model_entity("machine-1-1", known) == "machine-1-1"
    assert M.model_entity("machine-1-1-r1", known) == "machine-1-1"
    assert M.model_entity("machine-1-1-r12", known) == "machine-1-1"
    assert M.model_entity("machine-1-1-r2x", known) == "machine-1-1-r2x"     # known as it is
    assert M.model_entity("machine-9-9-r1", known) == "machine-9-9-r1"       # base unknown too: unchanged
    assert M.model_entity("machine-1-1-r0", known) == "machine-1-1-r0"       # replay numbers copies from 1
    assert M.model_entity("machine-1-1-rx", known) == "machine-1-1-rx"


def test_replica_is_scored_like_its_machine(setup):
    """replay.py --replicate: machine-a-r2 gets machine-a's AE scores but keeps its own entity_id."""
    raw, _, models = setup
    rows = raw[raw["entity_id"] == MACHINES[0]].drop(columns="entity_id")
    n_in = len(models["ae"]["feature_metrics"])
    orig, _ = M.step(MACHINES[0], rows, M.new_state(n_in), models)
    copy, _ = M.step(f"{MACHINES[0]}-r2", rows, M.new_state(n_in), models)
    assert (copy["entity_id"] == f"{MACHINES[0]}-r2").all()
    assert copy["ae_score"].notna().all()
    np.testing.assert_allclose(copy["ae_score"], orig["ae_score"], rtol=0, atol=0)
    assert copy["ae_flag"].tolist() == orig["ae_flag"].tolist()


def test_max_offsets_per_trigger_reaches_the_kafka_source(monkeypatch, tmp_path):
    """--max-offsets-per-trigger N -> read_telemetry(spark, N); unset -> None; < 1 is refused."""
    from src.streaming import features_job

    (tmp_path / "model.pt").write_bytes(b"")
    monkeypatch.setattr(M, "AE_DIR", tmp_path)
    seen = []

    class Stop(Exception):
        pass

    class FakeSpark:
        sparkContext = type("SC", (), {"setLogLevel": lambda self, level: None})()

    def fake_read(spark, max_offsets_per_trigger=None, min_partitions=None):
        seen.append((max_offsets_per_trigger, min_partitions))
        raise Stop

    monkeypatch.setattr(features_job, "build_spark", lambda *a, **k: FakeSpark())
    monkeypatch.setattr(features_job, "read_telemetry", fake_read)
    for argv in (["--input", "kafka", "--max-offsets-per-trigger", "200000"], ["--input", "kafka"],
                 ["--input", "kafka", "--min-partitions", "40"]):
        with pytest.raises(Stop):
            M.main(argv)
    assert seen == [(200000, None), (None, None), (None, 40)]
    for bad in (["--max-offsets-per-trigger", "0"], ["--min-partitions", "0"]):
        with pytest.raises(SystemExit):
            M.main(bad)


def test_read_telemetry_sets_max_offsets_per_trigger_only_when_given():
    from src.streaming import features_job

    class Reader:
        def __init__(self):
            self.options = {}

        def format(self, name):
            return self

        def option(self, key, value):
            self.options[key] = value
            return self

        def load(self):
            raise StopIteration(self.options)

    for cap, want in ((None, None), (5000, 5000)):
        reader = Reader()
        spark = type("Spark", (), {"readStream": reader})()
        with pytest.raises(StopIteration) as e:
            features_job.read_telemetry(spark, cap)
        assert e.value.args[0].get("maxOffsetsPerTrigger") == want
        assert e.value.args[0]["subscribe"] == S.TOPIC_SMD and e.value.args[0]["startingOffsets"] == "earliest"
    for parts, want in ((None, None), (40, 40)):
        reader = Reader()
        spark = type("Spark", (), {"readStream": reader})()
        with pytest.raises(StopIteration) as e:
            features_job.read_telemetry(spark, min_partitions=parts)
        assert e.value.args[0].get("minPartitions") == want


def test_state_store_conf():
    assert M.state_store_conf("hdfs") == {}
    conf = M.state_store_conf("rocksdb")
    assert conf["spark.sql.streaming.stateStore.providerClass"].endswith(".RocksDBStateStoreProvider")
    assert conf["spark.sql.streaming.stateStore.rocksdb.changelogCheckpointing.enabled"] == "true"


def test_experiment_switches_reach_spark(monkeypatch, tmp_path):
    """--state-store rocksdb sets the SQL confs before the query; --read-only builds no scoring plan;
    --checkpoint-dir is what --reset deletes (the default checkpoint is left alone)."""
    from src.streaming import features_job

    (tmp_path / "model.pt").write_bytes(b"")
    for name, sub in (("AE_DIR", ""), ("OUTPUT_DIR", "out"), ("LATENCY_ROOT", "lat"), ("CHECKPOINT_DIR", "default")):
        monkeypatch.setattr(M, name, tmp_path / sub if sub else tmp_path)
    (tmp_path / "default").mkdir()
    custom = tmp_path / "custom"
    custom.mkdir()

    class Stop(Exception):
        pass

    class FakeSpark:
        sparkContext = type("SC", (), {"setLogLevel": lambda self, level: None})()

        def __init__(self):
            self.conf = type("Conf", (), {"set": lambda _, k, v: confs.__setitem__(k, v)})()

    confs: dict = {}
    calls = []
    monkeypatch.setattr(features_job, "build_spark", lambda *a, **k: FakeSpark())
    monkeypatch.setattr(features_job, "read_telemetry", lambda spark, cap=None, **kw: "telemetry")
    monkeypatch.setattr(M, "build_scores", lambda t: (calls.append("build_scores"), (_ for _ in ()).throw(Stop))[1])
    monkeypatch.setattr(M, "telemetry_rows", lambda t: (calls.append(("rows", t)), (_ for _ in ()).throw(Stop))[1])

    with pytest.raises(Stop):
        M.main(["--input", "kafka", "--state-store", "rocksdb"])
    assert calls == ["build_scores"] and "spark.sql.streaming.stateStore.providerClass" in confs
    calls.clear()
    confs.clear()
    with pytest.raises(Stop):
        M.main(["--input", "kafka", "--read-only", "--reset", "--checkpoint-dir", str(custom)])
    assert calls == [("rows", "telemetry")] and confs == {}
    assert not custom.exists() and (tmp_path / "default").exists()
    with pytest.raises(SystemExit):
        M.main(["--state-store", "leveldb"])


def test_cassandra_switch(monkeypatch, tmp_path):
    """--cassandra HOSTS [--cassandra-cl CL] opens the store (schema first) before the query; unset = no store;
    --cassandra-cl alone, a bad level or --read-only with --cassandra are refused."""
    from src.streaming import features_job

    (tmp_path / "model.pt").write_bytes(b"")
    monkeypatch.setattr(M, "AE_DIR", tmp_path)

    class Stop(Exception):
        pass

    class FakeSpark:
        sparkContext = type("SC", (), {"setLogLevel": lambda self, level: None})()

    stores = []

    class FakeStore:
        def __init__(self, hosts, cl):
            self.hosts, self.consistency, self.schema = M.CS.parse_hosts(hosts), cl, False
            stores.append(self)

        def ensure_schema(self):
            self.schema = True

    monkeypatch.setattr(features_job, "build_spark", lambda *a, **k: FakeSpark())
    monkeypatch.setattr(features_job, "read_telemetry", lambda spark, cap=None, **kw: "telemetry")
    monkeypatch.setattr(M, "build_scores", lambda t: (_ for _ in ()).throw(Stop))
    monkeypatch.setattr(M.CS, "CassandraStore", FakeStore)
    with pytest.raises(Stop):
        M.main(["--input", "kafka"])
    assert stores == []
    with pytest.raises(Stop):
        M.main(["--input", "kafka", "--cassandra", "spark-w2,spark-w5"])
    with pytest.raises(Stop):
        M.main(["--input", "kafka", "--cassandra", "spark-w2", "--cassandra-cl", "all"])
    assert [(s.hosts, s.consistency, s.schema) for s in stores] == [
        (["spark-w2", "spark-w5"], "QUORUM", True), (["spark-w2"], "ALL", True)]
    for bad in (["--cassandra-cl", "ONE"], ["--cassandra", "h", "--cassandra-cl", "TWO"],
                ["--cassandra", "h", "--read-only"]):
        with pytest.raises(SystemExit):
            M.main(bad)


def test_cassandra_upsert_sends_only_the_batch_anomalies():
    sent = []

    class FakeStore:
        def upsert_anomalies(self, rows):
            sent.extend(rows)
            return M.CS.UpsertResult(rows=len(rows), ms=1.5)

    ws = 1_767_225_600_000_000   # 2026-01-01T00:00:00Z in microseconds
    pdf = pd.DataFrame({"entity_id": ["machine-1-1", "machine-1-1", "machine-1-2"],
                        "ws_us": [ws, ws + 300_000_000, ws], "ae_score": [4.0, 0.2, 5.0],
                        "top_metrics": [["m01", "m02", "m03"]] * 3, "is_anomaly": [True, False, True]})
    res = M.cassandra_upsert(FakeStore(), pdf)
    assert [(e, t.isoformat()) for e, t, _, _ in sent] == [("machine-1-1", "2026-01-01T00:00:00+00:00"),
                                                         ("machine-1-2", "2026-01-01T00:00:00+00:00")]
    assert M.cassandra_log_line(res, "QUORUM") == "cassandra upserts 2/2 anomalies at CL QUORUM in 2 ms"
    bad = M.CS.UpsertResult(rows=2, failed=1, ms=3.0, errors=["Unavailable: x"])
    assert "FAILED 1: Unavailable: x" in M.cassandra_log_line(bad, "ALL")
