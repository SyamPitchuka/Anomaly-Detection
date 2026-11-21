"""experiments/e8_onset.py: the refined-onset rule, roots with the 10-s tie, scoring, Pregel parity, report."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from experiments import e8_cascade as C
from experiments import e8_onset as O

B = C.T0_MIN + 25 * 60            # hour 25, in minutes
WS = B * 60                       # a window start, epoch seconds


def readings(drop_at=None, factor=0.1, spike_at=None, n_before=420, n_after=60, mad0=False):
    """10-s readings from WS - n_before*10 s; baseline ~ 10 +- noise; x factor from drop_at (epoch s)."""
    ts = WS - n_before * 10 + np.arange(n_before + n_after, dtype=np.int64) * 10
    rng = np.random.default_rng(0)
    x = np.full(len(ts), 10.0) if mad0 else 10 + rng.normal(0, 0.2, len(ts))
    if drop_at is not None:
        x[ts >= drop_at] *= factor
    if spike_at is not None:
        x[ts == spike_at] = 100.0
    return ts, x


def test_refined_onset_rule():
    ts, x = readings(drop_at=WS - 170)               # fault 170 s before the window start
    assert O.refined_onset(ts, x, WS) == (WS - 170, "refined")
    ts, x = readings(spike_at=WS - 100)              # one reading only: no run of 2
    assert O.refined_onset(ts, x, WS) == (WS, "none")
    ts, x = readings(drop_at=WS - 170)
    x[ts == WS - 170] = np.nan                       # missing readings are skipped, the run continues
    assert O.refined_onset(ts, x, WS) == (WS - 160, "refined")
    ts, x = readings(drop_at=WS - 170, mad0=True)
    assert O.refined_onset(ts, x, WS) == (WS, "mad0")
    ts, x = readings(n_before=20)                    # 20 baseline readings < 30
    assert O.refined_onset(ts, x, WS) == (WS, "no_baseline")
    ts, x = readings(drop_at=WS + 280, n_after=60)   # last two readings before the window end
    assert O.refined_onset(ts, x, WS) == (WS + 280, "refined")
    ts, x = readings(drop_at=WS + 290)               # one reading inside the window: no run
    assert O.refined_onset(ts, x, WS)[1] == "none"


def test_incident_onsets_reads_top_metric_and_raw(tmp_path):
    ts, x = readings(drop_at=WS - 60)
    tbl = pa.table({"event_time": pa.array(ts * 1_000_000).cast(pa.timestamp("us", tz="UTC")),
                    "metrics": pa.array([[("cpu", 50.0), ("net_in", float(v))] for v in x],
                                        type=pa.map_(pa.string(), pa.float64()))})
    pq.write_table(tbl, tmp_path / "m_1.parquet")
    inc = pd.DataFrame({"entity_id": ["m_1", "m_1"], "start": [B, B + 60], "end": [B + 15, B + 70]})
    tops = {("m_1", B): "net_in", ("m_1", B + 60): None}
    out = O.incident_onsets(inc, tops, tmp_path)
    assert out["onset_refined"].tolist() == [WS - 60, (B + 60) * 60]
    assert out["how"].tolist() == ["refined", "no_top_metric"] and out["onset_window"].tolist() == [WS, (B + 60) * 60]
    assert out["id"].tolist() == [f"m_1@{B}", f"m_1@{B + 60}"]


def inc_frame(rows):
    df = pd.DataFrame(rows, columns=["entity_id", "start", "end", "refined_offset_s"])
    df[["start", "end"]] += B
    df["id"] = [f"{e}@{s}" for e, s in zip(df["entity_id"], df["start"])]
    df["onset_window"] = df["start"] * 60
    df["onset_refined"] = df["onset_window"] + df.pop("refined_offset_s")
    return df.sort_values(["entity_id", "start"]).reset_index(drop=True)


def truth(root_offsets=None):
    members = ["m_1", "m_2", "m_3", "m_4", "m_5"]
    offs = root_offsets or {"m_1": 0, "m_2": 120, "m_3": 180, "m_4": 360, "m_5": 420}

    def iso(sec):
        return pd.Timestamp(sec, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return {"cascades": [{"cascade_id": "c1", "root": "m_1", "members": members, "start_time": iso(WS),
                          "end_time": iso(WS + 420 + 900),
                          "faults": [{"entity_id": m, "start_time": iso(WS + offs[m]),
                                      "end_time": iso(WS + offs[m] + 900)} for m in members]}]}


# root m_1 and m_2, m_3 flagged in the same window; refined onsets 0, 130, 185 s after it (m_3 within 10 s of nobody)
ROWS = [("m_1", 0, 15, 0), ("m_2", 0, 15, 130), ("m_3", 0, 15, 185), ("m_4", 5, 20, 70), ("m_5", 5, 20, 125),
        ("m_9", 600, 610, 0), ("m_10", 601, 611, 0), ("m_11", 602, 612, 0)]


def test_roots_scoring_and_false_groups():
    inc = inc_frame(ROWS)
    groups = O.reference_groups(inc)
    g = next(x for x in groups if "m_1" in x["machines"])
    assert g["roots"]["window"] == ["m_1", "m_2", "m_3"] and g["roots"]["refined"] == ["m_1"]
    res = O.score_run(truth(), inc, groups)
    c = res["cascades"][0]
    assert c["recall"] == 1.0 and c["precision"] == 1.0 and c["f1"] == 1.0 and c["root_alarmed"]
    assert not c["methods"]["window"]["root_correct"] and c["methods"]["refined"]["root_correct"]
    assert c["methods"]["window"]["onset_errors_s"] == {"m_1": 0, "m_2": 120, "m_3": 180, "m_4": 60, "m_5": 120}
    assert c["methods"]["refined"]["onset_errors_s"] == {"m_1": 0, "m_2": 10, "m_3": 5, "m_4": 10, "m_5": 5}
    s = res["summary"]
    assert s["window"]["root_accuracy"] == 0 and s["refined"]["root_accuracy"] == 1
    assert s["window"]["onset_error_median_s"] == 120 and s["refined"]["onset_error_median_s"] == 5
    assert s["false_groups_per_24h"] == 1.0 and s["members_detected"] == 5
    o = O.over_seeds([s, s])
    assert o["refined"]["root_accuracy"]["mean"] == 1 and o["f1"]["mean"] == 1


def test_tie_within_10s_and_silent_root():
    rows = [("m_1", 0, 15, 0), ("m_2", 0, 15, 8), ("m_3", 5, 20, 0), ("m_4", 5, 20, 0), ("m_5", 5, 20, 0)]
    g = O.reference_groups(inc_frame(rows))[0]
    assert g["roots"]["refined"] == ["m_1", "m_2"]                        # 8 s apart: a tie
    silent = [r for r in rows if r[0] != "m_1"]                            # case C: the root never alarms
    inc = inc_frame(silent)
    res = O.score_run(truth(), inc, O.reference_groups(inc))
    c = res["cascades"][0]
    assert not c["root_alarmed"] and c["recall"] == 0.8 and c["methods"]["refined"]["roots"] == ["m_2"]
    assert res["summary"]["refined"]["root_accuracy"] == 0


def test_undetected_cascade():
    inc = inc_frame([("m_9", 600, 610, 0)])
    res = O.score_run(truth(), inc, O.reference_groups(inc))
    s = res["summary"]
    assert s["f1"] == 0 and s["precision"] is None and s["root_alarmed"] == 0
    assert s["window"]["onset_error_median_s"] is None


def fake_doc():
    inc = inc_frame(ROWS)
    groups = O.reference_groups(inc)
    runs = []
    for set_name, seed, case in O.all_runs():
        sc = O.score_run(truth(), inc, groups)
        runs.append({"set": set_name, "seed": seed, "case": case, "name": O.run_name(seed, case),
                     "factors": {"root": 0.1, "dependent": 0.1}, "how": {"refined": 7, "none": 1},
                     "agreement": {"pregel_equals_pandas": True}, "score": sc, "groups": groups,
                     "incidents": inc.to_dict(orient="records"), "truth": truth()})
    sets = {}
    for r in runs:
        sets.setdefault(r["set"], []).append(r["score"]["summary"])
    return {"runs": runs, "sets": {k: O.over_seeds(v) for k, v in sets.items()}, "total_s": 10.0}


def test_section_lines():
    text = "\n".join(O.section_lines({"pregel": fake_doc(), "neo4j": None, "stream": None}))
    assert "## E8c onset refinement" in text and "Rules, fixed before" in text
    assert "| held-out C | refined |" in text and "| seeds 7-9 | window |" in text
    assert "Root = earliest visible fault is an assumption" in text and "Why the refined root was still wrong" in text
    assert O.failure_modes(fake_doc()["runs"])["correct"] == 12
    assert "seeds 7-9: 0.00 -> 1.00 (onset error 120 -> 5 s)" in text and "84 of 96 incidents" in text


def test_case_factors_keep_the_plan(tmp_path):
    from src.producer import inject_faults as I
    folder = tmp_path / S.PATHS["raw"] / "source=alibaba"
    folder.mkdir(parents=True)
    ts = np.arange(I.BASE_S, I.BASE_S + 8 * 3600, 60, dtype=np.int64)
    for k in range(12):
        tbl = pa.table({"entity_id": [f"m_{k}"] * len(ts),
                        "event_time": pa.array(ts * 1_000_000).cast(pa.timestamp("us", tz="UTC")),
                        "metrics": [[(m, 10.0 + k) for m in I.METRICS] for _ in ts],
                        "attrs": [[] for _ in ts], "label": pa.nulls(len(ts), pa.int32())},
                       schema=S.arrow_schema("raw"))
        pq.write_table(tbl, folder / f"p{k}.parquet")
    a = I.inject_cascades(tmp_path, tmp_path / "a", machines=12, hours=8, seed=3, n=1, window_h=(2, 8))
    c = I.inject_cascades(tmp_path, tmp_path / "c", machines=12, hours=8, seed=3, n=1, window_h=(2, 8),
                          root_factor=0.8, dependent_factor=0.5)
    strip = [{k: v for k, v in f.items() if k != "factor"} for f in a["cascades"][0]["faults"]]
    assert strip == [{k: v for k, v in f.items() if k != "factor"} for f in c["cascades"][0]["faults"]]
    assert c["factors"] == {"root": 0.8, "dependent": 0.5} and a["factors"] == {"root": 0.1, "dependent": 0.1}
    assert [f["factor"] for f in c["cascades"][0]["faults"]] == [0.8, 0.5, 0.5, 0.5, 0.5]
    root = c["cascades"][0]["root"]
    t = pq.read_table(tmp_path / "c" / S.PATHS["raw"] / "source=alibaba" / f"{root}.parquet").to_pandas()
    vals = t["metrics"].map(lambda kv: dict(kv)["net_in"])
    assert sorted(set(vals.round(6))) == sorted({round((10.0 + int(root[2:])) * 0.8, 6), 10.0 + int(root[2:])})


class Result(list):
    def consume(self):
        return None


class FakeSession:
    """Answers like Neo4j would for the loaded incident graph (chain = component of the FOLLOWS edges)."""

    def __init__(self):
        self.nodes, self.edges = {}, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **p):
        if "DETACH DELETE" in cypher:
            self.nodes, self.edges = {}, []
        elif "MERGE (i:OnsetIncident" in cypher:
            self.nodes = {r["id"]: dict(r) for r in p["rows"]}
        elif "MERGE (a)-[:FOLLOWS]->(b)" in cypher:
            self.edges = [(r["src"], r["dst"]) for r in p["rows"]]
        elif "gds.wcc.write" in cypher:
            lab = C._union_find(sorted(self.nodes), self.edges)
            for i, c in lab.items():
                self.nodes[i]["chain"] = c
        elif "RETURN i.id AS id, i.chain AS chain" in cypher:
            return Result({"id": i, "chain": n["chain"]} for i, n in self.nodes.items())
        elif "collect(DISTINCT r.machine)" in cypher:
            out = {}
            for n in self.nodes.values():
                out.setdefault(n["chain"], []).append(n)
            rows = []
            for c, ns in out.items():
                first = min(n[p["p"]] for n in ns)
                rows.append({"chain": c, "roots": sorted({n["machine"] for n in ns if n[p["p"]] <= first + p["tie"]})})
            return Result(rows)
        return Result()


def test_neo4j_run_agrees_with_reference():
    doc = fake_doc()
    doc["runs"] = doc["runs"][:2]
    driver = type("D", (), {"session": lambda self, s=FakeSession(): s})()
    out = O.neo4j_run(driver, doc)
    assert all(v["agrees_with_pregel"] for v in out["runs"].values()) and len(out["runs"]) == 2


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from experiments import e8_graph as E
    try:
        s = E.build_spark(master="local[2]", ckpt=tmp_path_factory.mktemp("ckpt"), app="test_e8_onset")
        import graphframes  # noqa: F401
    except Exception as exc:  # pragma: no cover - offline machine without the package in the Ivy cache
        pytest.skip(f"GraphFrames not available: {exc}")
    yield s
    s.stop()


def test_pregel_groups_equal_reference(spark):
    inc = inc_frame(ROWS + [("m_6", 300, 310, 0)])
    groups, info = O.pregel_groups(spark, inc)
    assert O.canonical(groups) == O.canonical(O.reference_groups(inc))
    assert info["edges"] == len(C.directed_pairs(inc))
    json.dumps(groups)
