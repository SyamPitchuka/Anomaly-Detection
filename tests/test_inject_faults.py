import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from src.producer import inject_faults as I
from src.producer import replay as R

MACHINES = [f"m_{i}" for i in range(1, 6)]
HOURS, CLEAN_HOURS = 6, 2
STEP_S = 10
N = HOURS * 3600 // STEP_S                     # 2160 rows per machine
BASE_S = I.BASE_S


def _machine_table(entity: str, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.int64) * STEP_S
    cols = {
        "cpu": 50 + 20 * np.sin(t / 3600 * 2 * np.pi / 3) + rng.normal(0, 1, N),
        "mem": 40 + rng.normal(0, 0.5, N),
        "mem_gps": 5 + rng.normal(0, 0.1, N),
        "mkpi": 1.5 + rng.normal(0, 0.05, N),
        "net_in": 20 + 5 * np.sin(t / 1800) + rng.normal(0, 1, N),
        "net_out": 10 + rng.normal(0, 0.5, N),
        "disk_io": 3 + rng.normal(0, 0.2, N),
    }
    metrics = [[(k, float(v[i])) for k, v in cols.items() if not (k == "mkpi" and i % 7 == 0)] for i in range(N)]
    times = pa.array((BASE_S + t) * 1_000_000, type=pa.int64()).cast(pa.timestamp("us", tz="UTC"))
    return pa.table({"entity_id": pa.array([entity] * N, pa.string()), "event_time": times,
                     "metrics": pa.array(metrics, S.arrow_schema("raw").field("metrics").type),
                     "attrs": pa.array([[] for _ in range(N)], S.arrow_schema("raw").field("attrs").type),
                     "label": pa.nulls(N, pa.int32())}, schema=S.arrow_schema("raw"))


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    folder = root / "data/raw/source=alibaba"
    folder.mkdir(parents=True)
    for seed, entity in enumerate(MACHINES, start=1):
        pq.write_table(_machine_table(entity, seed), folder / f"{entity}.parquet")
    pq.write_table(_machine_table("m_1", 99).slice(0, 5), folder / "fake.parquet")      # must be skipped
    return root


@pytest.fixture(scope="module")
def injected(repo):
    log = I.inject(repo, repo / "out", machines=5, hours=HOURS, clean_hours=CLEAN_HOURS, per_type=1, seed=3)
    return repo / "out", log


def _frame(path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    ts = ((pd.to_datetime(df["event_time"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
    out = pd.DataFrame([dict(m) for m in df["metrics"]]).reindex(columns=I.METRICS)
    out.insert(0, "ts", ts)
    out["label"] = df["label"].to_numpy()
    return out


def _original(repo, entity) -> pd.DataFrame:
    return _frame(repo / f"data/raw/source=alibaba/{entity}.parquet")


def _injected(out, entity) -> pd.DataFrame:
    return _frame(out / f"data/raw/source=alibaba/{entity}.parquet")


def _fault(log, kind):
    return next(f for f in log["faults"] if f["type"] == kind)


def _secs(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def test_log_has_one_fault_per_type_on_distinct_machines(injected):
    out, log = injected
    assert sorted(f["type"] for f in log["faults"]) == ["break", "drift", "spike"]
    assert len({f["entity_id"] for f in log["faults"]}) == 3
    assert json.loads((out / "injection_log.json").read_text()) == log
    clean_end = BASE_S + CLEAN_HOURS * 3600
    for f in log["faults"]:
        start, end = _secs(f["start_time"]), _secs(f["end_time"])
        assert start % 300 == 0 and start >= clean_end and end <= BASE_S + HOURS * 3600
        assert end - start == {"spike": 300, "drift": 1800, "break": 900}[f["type"]]
        assert f["n_rows"] == (end - start) // STEP_S
    assert _fault(log, "break")["metric"] == "net_in"


def test_labels_are_one_inside_the_fault_only(repo, injected):
    out, log = injected
    for entity in MACHINES:
        df = _injected(out, entity)
        assert set(df["label"]) <= {0, 1}
        faults = [f for f in log["faults"] if f["entity_id"] == entity]
        inside = np.zeros(len(df), dtype=bool)
        for f in faults:
            inside |= ((df["ts"] >= _secs(f["start_time"])) & (df["ts"] < _secs(f["end_time"]))).to_numpy()
        assert (df["label"].to_numpy() == inside.astype(int)).all()


def test_spike_drift_and_break_change_the_metric_as_specified(repo, injected):
    out, log = injected
    for kind, factor in (("spike", lambda m: 3.0), ("drift", lambda m: 1 + 0.02 * m), ("break", lambda m: 0.1)):
        f = _fault(log, kind)
        before, after = _original(repo, f["entity_id"]), _injected(out, f["entity_id"])
        start, end = _secs(f["start_time"]), _secs(f["end_time"])
        inside = ((before["ts"] >= start) & (before["ts"] < end)).to_numpy()
        minutes = ((before["ts"] - start) / 60).to_numpy()
        m = f["metric"]
        expected = before[m].to_numpy() * np.where(inside, factor(minutes), 1.0)
        np.testing.assert_allclose(after[m].to_numpy(), expected, equal_nan=True)
        others = [x for x in I.METRICS if x != m]
        np.testing.assert_array_equal(after[others].to_numpy(), before[others].to_numpy())     # nothing else moves


def test_break_keeps_cpu_high(repo, injected):
    out, log = injected
    f = _fault(log, "break")
    df = _injected(out, f["entity_id"])
    window_cpu = df.groupby(df["ts"] // 300 * 300)["cpu"].mean()
    start = _secs(f["start_time"])
    assert window_cpu.loc[start:start + 899].min() >= window_cpu.quantile(0.75)


def test_untouched_machines_and_source_data(repo, injected):
    out, log = injected
    hit = {f["entity_id"] for f in log["faults"]}
    for entity in set(MACHINES) - hit:
        after, before = _injected(out, entity), _original(repo, entity)
        np.testing.assert_array_equal(after[I.METRICS].to_numpy(), before[I.METRICS].to_numpy())
        assert (after["label"] == 0).all()
    assert pq.read_table(repo / "data/raw/source=alibaba/m_1.parquet").column("label").null_count == N   # input unchanged


def test_missing_values_stay_missing_and_schema_matches_contract(injected):
    out, _ = injected
    table = pq.read_table(out / "data/raw/source=alibaba/m_2.parquet")
    assert table.schema.names == list(S.LAYERS["raw"])
    for name, code in S.LAYERS["raw"].items():
        assert S.type_matches(code, table.schema.field(name).type), name
    first = dict(table.column("metrics").to_pylist()[0])
    seventh = dict(table.column("metrics").to_pylist()[7])
    assert "mkpi" not in first and "mkpi" not in seventh and len(first) == 6      # rows 0 and 7 had no mkpi


def test_same_seed_same_log_other_seed_other_log(repo, injected):
    _, log = injected
    again = I.inject(repo, repo / "out2", machines=5, hours=HOURS, clean_hours=CLEAN_HOURS, per_type=1, seed=3)
    assert again == log
    other = I.inject(repo, repo / "out3", machines=5, hours=HOURS, clean_hours=CLEAN_HOURS, per_type=1, seed=4)
    assert [(f["fault_id"], f["start_time"]) for f in other["faults"]] != [(f["fault_id"], f["start_time"]) for f in log["faults"]]


def test_not_enough_machines_is_an_error(repo):
    with pytest.raises(ValueError, match="need 2 machines"):
        I.inject(repo, repo / "out4", machines=5, hours=HOURS, clean_hours=CLEAN_HOURS, per_type=2, seed=1)


def test_replay_sends_every_row_in_event_order(injected):
    out, log = injected
    records = I.replay_records(out)
    assert len(records) == log["rows"] == len(MACHINES) * N
    assert list(records[0]) == ["source", "entity_id", "event_time", "metrics", "attrs", "label"]
    times = [(r["event_time"], r["entity_id"]) for r in records]
    assert times == sorted(times)
    assert sum(r["label"] for r in records) == sum(f["n_rows"] for f in log["faults"])
    assert R.raw_files(out, "alibaba")                                    # the folder is a replayable raw layer


def test_report_with_perfect_and_late_flags(injected):
    out, log = injected
    windows = I.window_labels(I.to_frame(I.replay_records(out)))
    clean_end = _secs(log["clean_end"])
    late = {"spike": 0, "drift": 2, "break": 1}                          # windows after the fault start
    flagged = []
    for f in log["faults"]:
        first = _secs(f["start_time"]) + late[f["type"]] * 300
        flagged += [{"entity_id": f["entity_id"], "window": first, "flag": True}]
    table = I.report(log, windows, pd.DataFrame(flagged))
    assert table.loc[["spike", "drift", "break", "all"], "recall"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert table["mean_delay_min"].to_dict() == {"spike": 0.0, "drift": 10.0, "break": 5.0, "all": 5.0}
    assert (table["false_alarms/day"] == 0).all() and table["events"].tolist() == [1, 1, 1, 3]
    none = I.report(log, windows, pd.DataFrame({"entity_id": [], "window": [], "flag": []}))
    assert (none["recall"] == 0).all() and none["mean_delay_min"].isna().all() and clean_end


def test_builtin_detector_catches_spike_and_break(repo):
    log, table = I.run(repo, repo / "out5", machines=5, hours=HOURS, clean_hours=CLEAN_HOURS, per_type=1, seed=3)
    assert list(table.index) == ["spike", "drift", "break", "all"]
    assert table.loc["spike", "recall"] == 1.0 and table.loc["break", "recall"] == 1.0
    assert table.loc["spike", "mean_delay_min"] == 0.0
    assert table.loc["all", "events"] == len(log["faults"]) == 3


def test_external_scores_layer_is_read_as_flags(injected, tmp_path):
    out, log = injected
    f = _fault(log, "spike")
    start = datetime.fromtimestamp(_secs(f["start_time"]), tz=timezone.utc)
    schema = S.arrow_schema("scores")
    row = {n: None for n in schema.names}
    row.update(entity_id=f["entity_id"], window_start=start, window_end=start, is_anomaly=True)
    folder = tmp_path / "scores/source=alibaba"
    folder.mkdir(parents=True)
    pq.write_table(pa.table({n: [row[n]] for n in schema.names}, schema=schema), folder / "part.parquet")
    flags = I.external_flags(tmp_path / "scores", "is_anomaly")
    assert flags.to_dict("records") == [{"entity_id": f["entity_id"], "window": _secs(f["start_time"]), "flag": True}]
