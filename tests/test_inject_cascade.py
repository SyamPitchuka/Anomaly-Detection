"""src/producer/inject_faults.py --mode cascade (E8c): the planning rules and the written raw layer + ground truth."""
import json
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from src.producer import inject_faults as I

H = 3600
MACHINES = [f"m_{k}" for k in range(1, 13)]


def series(hours=(20, 31), step=60, machines=MACHINES, no_net_in=()):
    """entity -> frame as read_slice returns it: one row per `step` seconds in the hour range."""
    ts = np.arange(I.BASE_S + hours[0] * H, I.BASE_S + hours[1] * H, step, dtype=np.int64)
    out = {}
    for k, m in enumerate(machines):
        rng = np.random.default_rng(k)
        df = pd.DataFrame({"ts": ts})
        for metric in I.METRICS:
            df[metric] = 10 + rng.random(len(ts))
        if m in no_net_in:
            df["net_in"] = np.nan
        df["label"] = 0
        out[m] = df
    return out


def plan(seed=7, **kw):
    args = {"n": 2, "window_h": (24, 31), "avoid": [(23.5, 55)], **kw}
    return I.plan_cascades(series(), random.Random(seed), **args)


def test_plan_follows_the_rules():
    p = plan()
    assert len(p) == 2
    starts = [c["start_s"] for c in p]
    assert starts == sorted(starts) and starts[1] - starts[0] >= 2 * H
    used = [m for c in p for m in c["members"]]
    assert len(used) == len(set(used)) == 10                        # 5 per cascade, disjoint
    for c in p:
        t = c["start_s"]
        assert t % 300 == 0 and t >= I.BASE_S + (24.42 + 1) * H     # 1 h after the 23.5 h + 55 min wave
        assert t + (8 + 15) * 60 <= I.BASE_S + 31 * H
        assert c["root"] == c["members"][0] and c["faults"][0]["role"] == "root"
        assert c["faults"][0]["offset_min"] == 0 and c["faults"][0]["start_s"] == t
        for f in c["faults"][1:]:
            assert f["role"] == "dependent" and 2 <= f["offset_min"] <= 8 and f["start_s"] == t + 60 * f["offset_min"]
        assert all(f["end_s"] - f["start_s"] == 15 * 60 and f["metric"] == "net_in" and f["type"] == "break"
                   for f in c["faults"])


def test_plan_is_seeded_and_spike_uses_random_metrics():
    assert plan(7) == plan(7) and plan(7) != plan(8)
    sp = plan(7, kind="spike")
    assert all(f["type"] == "spike" and f["metric"] in I.METRICS for c in sp for f in c["faults"])


def test_break_needs_net_in_and_enough_machines():
    s = series(no_net_in=MACHINES[:3])
    p = I.plan_cascades(s, random.Random(1), n=1, window_h=(24, 31))
    assert not set(p[0]["members"]) & set(MACHINES[:3])
    with pytest.raises(ValueError, match="free machines"):
        I.plan_cascades(series(machines=MACHINES[:9]), random.Random(1), n=2, window_h=(24, 31))
    with pytest.raises(ValueError):
        I.plan_cascades(series(), random.Random(1), n=1, kind="drift")


def test_start_candidates_respect_window_and_avoid():
    c = I.cascade_start_candidates((24, 30), [(25.0, 60)])
    hours = [(t - I.BASE_S) / H for t in c]
    assert min(hours) == 27 and max(hours) <= 30 - 23 / 60   # [24, 24 + 23 min) would end inside the margin
    assert all(h + 23 / 60 <= 24.0 + 1e-9 or h >= 27.0 for h in hours)   # 1 h margin around [25, 26)
    assert I.avoid_arg("23.5:55") == (23.5, 55.0)


def test_inject_cascades_writes_raw_truth_and_log(tmp_path):
    folder = tmp_path / S.PATHS["raw"] / "source=alibaba"
    folder.mkdir(parents=True)
    rows = series(hours=(0, 8))
    for m, df in rows.items():          # one Spark-like part file per machine
        tbl = pa.table({"entity_id": [m] * len(df),
                        "event_time": pa.array(df["ts"].to_numpy() * 1_000_000).cast(pa.timestamp("us", tz="UTC")),
                        "metrics": [[(k, float(v)) for k, v in zip(I.METRICS, r)] for r in df[I.METRICS].to_numpy()],
                        "attrs": [[] for _ in range(len(df))], "label": pa.nulls(len(df), pa.int32())},
                       schema=S.arrow_schema("raw"))
        pq.write_table(tbl, folder / f"part-{m}.parquet")
    out = tmp_path / "inj"
    truth = I.inject_cascades(tmp_path, out, machines=12, hours=8, seed=3, n=1, window_h=(2, 8))
    saved = json.loads((out / I.CASCADE_LOG).read_text())
    assert saved == truth and saved["mode"] == "cascade" and len(saved["cascades"]) == 1
    c = saved["cascades"][0]
    assert len(c["members"]) == 5 and c["root"] == c["members"][0]
    assert c["faults"][0]["start_time"] == c["start_time"] and all(f["n_rows"] == 15 for f in c["faults"])
    log = json.loads((out / I.LOG_NAME).read_text())
    assert len(log["faults"]) == 5 and log["clean_end"].startswith("2026-01-01T02:00")
    # the written raw layer: net_in x 0.1 and label 1 exactly inside the root's fault
    root = pq.read_table(out / S.PATHS["raw"] / "source=alibaba" / f"{c['root']}.parquet").to_pandas()
    t0 = pd.Timestamp(c["faults"][0]["start_time"])
    inside = (root["event_time"] >= t0) & (root["event_time"] < t0 + pd.Timedelta(minutes=15))
    assert root.loc[inside, "label"].eq(1).all() and root.loc[~inside, "label"].eq(0).all()
    before = rows[c["root"]].set_index("ts")["net_in"]
    after = root.assign(ts=root["event_time"].map(lambda t: int(t.timestamp())),
                        net_in=root["metrics"].map(lambda kv: dict(kv)["net_in"])).set_index("ts")["net_in"]
    ratio = (after / before.reindex(after.index)).round(9)
    assert set(ratio[inside.to_numpy()]) == {0.1} and set(ratio[~inside.to_numpy()]) == {1.0}


def test_no_cascade_starts_inside_a_trace_gap():
    s = series(hours=(20, 31))
    gap = (I.BASE_S + 25 * H, I.BASE_S + 29 * H)      # every machine silent in hours 25-29
    s = {m: df[(df["ts"] < gap[0]) | (df["ts"] >= gap[1])].reset_index(drop=True) for m, df in s.items()}
    for seed in range(5):
        for c in I.plan_cascades(s, random.Random(seed), n=1, window_h=(24, 31)):
            t = c["start_s"]
            assert t + 23 * 60 <= gap[0] or t >= gap[1]
