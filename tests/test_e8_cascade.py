"""experiments/e8_cascade.py: directed cascade graph, roots, scoring rules, report, Pregel parity, Neo4j calls.

Fixture (minutes after hour 25 of the trace = B):
  cascade c1 (root m_1, members m_1..m_5): m_1 [0,15], m_3 [0,15] (same onset as the root -> tie),
      m_2 [5,20], m_4 [10,25], m_5 no incident; m_6 (not a member) [20,30] joins, and m_6 also alarms at
      [605,615] together with the background chain m_17 [600,610], m_18 [605,615], m_19 [608,618].
  cascade c2 (root m_7, members m_7..m_11): m_7 [180,195], m_8 and m_9 [185,200], m_10 and m_11 [190,205].
  cascade c3 (members m_12..m_16): no incidents (missed).
  m_20: no incidents at all.
"""
import json

import numpy as np
import pandas as pd
import pytest

from experiments import e8_cascade as C

B = C.T0_MIN + 25 * 60
MACHINES = [f"m_{k}" for k in range(1, 21)]
INC = [("m_1", 0, 15), ("m_3", 0, 15), ("m_2", 5, 20), ("m_4", 10, 25), ("m_6", 20, 30), ("m_6", 605, 615),
       ("m_17", 600, 610), ("m_18", 605, 615), ("m_19", 608, 618),
       ("m_7", 180, 195), ("m_8", 185, 200), ("m_9", 185, 200), ("m_10", 190, 205), ("m_11", 190, 205)]


def iso(minute):
    return pd.Timestamp(minute * 60, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")


def inc_df():
    df = pd.DataFrame(INC, columns=["entity_id", "start", "end"])
    df[["start", "end"]] += B
    return df.sort_values(["entity_id", "start"]).reset_index(drop=True)


def truth():
    def casc(cid, members, t):
        return {"cascade_id": cid, "root": members[0], "members": members, "start_time": iso(B + t),
                "end_time": iso(B + t + 23), "faults": [{"entity_id": m, "start_time": iso(B + t + (0 if k == 0 else 5)),
                                                         "end_time": iso(B + t + 15 + (0 if k == 0 else 5))}
                                                        for k, m in enumerate(members)]}
    return {"machines": MACHINES, "cascades": [casc("c1", ["m_1", "m_2", "m_3", "m_4", "m_5"], 0),
                                               casc("c2", ["m_7", "m_8", "m_9", "m_10", "m_11"], 180),
                                               casc("c3", [f"m_{k}" for k in range(12, 17)], 400)]}


def test_directed_edges_rule():
    inc = inc_df()
    e = C.directed_edges(inc).set_index(["src", "dst"])["weight"].to_dict()
    assert e[("m_1", "m_2")] == 1 and e[("m_1", "m_6")] == 1     # m_6 starts at 20 = m_1's end + 10: allowed
    assert ("m_1", "m_3") not in e and ("m_3", "m_1") not in e   # same onset: strictly after fails both ways
    assert ("m_2", "m_1") not in e                               # later incident never points back
    assert ("m_6", "m_19") in e and ("m_17", "m_6") in e and ("m_6", "m_18") not in e   # m_6 second incident
    assert not any(a == b for a, b in e)                          # no self loops
    assert ("m_7", "m_8") in e and ("m_8", "m_9") not in e       # m_8, m_9 same onset


def test_components_roots_and_time_resolved_chains():
    ref = C.reference(MACHINES, inc_df())
    groups = C.members_of(ref["components"])
    assert groups["m_1"] == ["m_1", "m_17", "m_18", "m_19", "m_2", "m_3", "m_4", "m_6"]  # merged through m_6
    assert groups["m_10"] == ["m_10", "m_11", "m_7", "m_8", "m_9"]
    assert groups["m_20"] == ["m_20"] and groups["m_5"] == ["m_5"]
    assert ref["roots"]["m_1"] == ["m_1", "m_3"] and ref["roots"]["m_10"] == ["m_7"] and ref["roots"]["m_20"] == []
    chains = {tuple(c["machines"]): c for c in ref["chains"]}
    assert ("m_1", "m_2", "m_3", "m_4", "m_6") in chains and ("m_17", "m_18", "m_19", "m_6") in chains
    assert chains[("m_17", "m_18", "m_19", "m_6")]["roots"] == ["m_17"]
    rp = ref["revpr"]
    assert max(["m_7", "m_8", "m_9", "m_10", "m_11"], key=rp.get) == "m_7"
    assert rp["m_20"] == pytest.approx(0.15)


def test_scoring_rules():
    inc = inc_df()
    ref = C.reference(MACHINES, inc)
    res = C.score_run(truth(), inc, ref["components"], ref["roots"], ref["revpr"], ref["chains"])
    c1, c2, c3 = res["cascades"]
    assert c1["detected"] and c1["members_alarmed"] == ["m_1", "m_2", "m_3", "m_4"] and c1["root_delay_min"] == 0
    assert c1["main"]["recall"] == 0.8 and c1["main"]["precision"] == 0.5
    assert not c1["main"]["root_correct"] and c1["main"]["root_in_tie"]
    assert c1["time"]["recall"] == 0.8 and c1["time"]["precision"] == 0.8 and c1["time"]["roots"] == ["m_1", "m_3"]
    assert c2["main"]["recall"] == c2["main"]["precision"] == 1.0 and c2["main"]["root_correct"]
    assert c2["main"]["revpr_correct"] and c2["time"]["root_correct"]
    assert not c3["detected"] and c3["main"]["recall"] == 0 and c3["main"]["precision"] is None
    fg = res["false_groups"]
    assert fg["main"] == 0 and fg["time"] == 1 and fg["time_groups"][0]["machines"] == ["m_17", "m_18", "m_19", "m_6"]
    s = res["summary"]
    assert s["detected"] == 2 and s["members_alarmed"] == 9 and s["roots_alarmed"] == 2
    assert s["main"]["recall_mean"] == pytest.approx(0.6) and s["main"]["precision_mean"] == pytest.approx(0.75)
    assert s["main"]["root_correct"] == 1 and s["main"]["root_in_tie"] == 1 and s["main"]["full_recovery"] == 1
    assert s["time"]["precision_mean"] == pytest.approx(0.9)
    o = C.over_seeds([s, s])
    assert o["detected"] == {"mean": 2.0, "min": 2.0, "max": 2.0} and "revpr_correct" not in o["time"]


def test_scope_and_kafka_command():
    inc = inc_df()
    assert len(C.in_scope(inc, (25.0, 26.0))) == 5          # starts at 0..20 min after hour 25
    assert len(C.in_scope(inc, None)) == len(inc)
    js = C.starting_offsets_json({1: 5, 0: 7})
    assert json.loads(js) == {"telemetry.alibaba": {"0": 7, "1": 5}}
    cmd = C.score_job_cmd("out", "ck", {0: 0})
    assert cmd[1:] == ["src/streaming/score_job.py", "--input", "kafka", "--source", "alibaba", "--once",
                       "--out-dir", "out", "--checkpoint-dir", "ck", "--starting-offsets",
                       '{"telemetry.alibaba": {"0": 0}}']


def fake_results(tmp_path):
    inc = inc_df()
    ref = C.reference(MACHINES, inc)
    sc = C.score_run(truth(), inc, ref["components"], ref["roots"], ref["revpr"], ref["chains"])
    agree = {"spark_edges_equal_pandas": True, "components_equal": True, "roots_equal": True,
             "revpr_max_abs_diff": 1e-16}
    run = {"name": "seed_7", "seed": 7, "truth": truth(), "score": sc, "agreement": agree,
           "n_edges": len(ref["edges"]), "edges": ref["edges"].to_dict(orient="records"), "onsets": ref["onsets"],
           "roots": ref["roots"], "components": ref["components"], "revpr": ref["revpr"]}
    base = {"name": "clean_0_72", "false_groups": {"main": 3, "time": 2}, "agreement": agree, "n_edges": 9}
    pregel = {"seeds": [run], "baseline": base, "over_seeds": C.over_seeds([sc["summary"]]), "total_s": 60.0,
              "spark_start_s": 5.0, "rules": C.rules()}
    neo = {"total_s": 2.0, "seeds": {"7": {"compare": {"components_equal": True, "roots_equal": True,
                                                       "revpr_max_abs_diff": 1e-15}}},
           "demo": {"seed": 7, "cascade": "c1", "true_root": "m_1", "members": truth()["cascades"][0]["members"],
                    "machine": "m_4", "queries": [{"key": "paths", "title": "(b) path", "cypher": C.Q_PATHS,
                                                   "rows": [{"root": "m_1", "member": "m_4", "path": ["m_1", "m_4"],
                                                             "hops": 1}]}]}}
    stream = {"total_s": 300.0, "seeds": [{"seed": 7, "kafka_records": 100, "incidents": 14, "score_job_s": 90.0,
                                           "parity": {"stream_windows": 50, "stream_flags": 9, "same_flag": 50,
                                                      "common_windows": 50}}]}
    return {"pregel": pregel, "neo4j": neo, "stream": stream}


def test_section_lines_and_plot(tmp_path, monkeypatch):
    pytest.importorskip("networkx")
    res = fake_results(tmp_path)
    text = "\n".join(C.section_lines(res, "e8_graph_cascade.png"))
    assert "## E8c" in text and "**These are injected cascades**" in text
    assert "Rules, fixed before any result was looked at" in text
    assert "| 7 | 2/3 | 9/15 | 0.60 | 0.75 | 1 | 1/3 | 1 | 1/3 | 0 |" in text      # main rule row
    assert "| 7 | 2/3 | 9/15 | 0.60 | 0.90 |" in text                                # time-resolved row
    assert "m_1 -> m_4" in text and "![E8c cascade](e8_graph_cascade.png)" in text
    assert "50/50" in text and "(1.0)" in text                                     # parity, clean false groups / day
    # plot reads the demo seed's incidents
    d = tmp_path / "seed_7" / "data" / "incidents" / "source=alibaba"
    d.mkdir(parents=True)
    inc = inc_df()
    pd.DataFrame({"entity_id": inc["entity_id"], "start_time": pd.to_datetime(inc["start"] * 60, unit="s", utc=True),
                  "end_time": pd.to_datetime(inc["end"] * 60, unit="s", utc=True)}).to_parquet(d / "i.parquet")
    monkeypatch.setattr(C, "seed_dir", lambda seed, root=None: tmp_path / f"seed_{seed}")
    out = tmp_path / "c.png"
    C.plot(out, res)
    assert out.stat().st_size > 10_000


class Result:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None

    def consume(self):
        return None


class FakeSession:
    def __init__(self, run):
        self.run_doc, self.calls = run, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **p):
        self.calls.append((cypher, p))
        c = " ".join(cypher.split())
        r = self.run_doc
        if "dbms.components" in c:
            return Result([{"v": "5.26.0", "e": "community"}])
        if "gds.version" in c:
            return Result([{"v": "2.13.2"}])
        if "count(m) AS n" in c or "count(r) AS n" in c:
            return Result([{"n": 20}])
        if "m.component AS v" in c:
            return Result([{"id": m, "v": hash(k) % 97} for m, k in r["components"].items()])
        if "gds.pageRank.stream" in c:
            return Result([{"id": m, "v": v} for m, v in r["revpr"].items()])
        if c.startswith("MATCH (c:CascadeMachine) WHERE c.onset IS NOT NULL"):
            return Result([{"root": m} for rs in r["roots"].values() for m in rs])
        if "$machine" in c or "{id: $machine}" in c:
            return Result([{"machine": p["machine"]}])
        return Result([])


def test_neo4j_run_with_fake_session(tmp_path):
    run = fake_results(tmp_path)["pregel"]["seeds"][0]
    s = FakeSession(run)
    driver = type("D", (), {"session": lambda self: s})()
    out = C.neo4j_run(driver, {"seeds": [run]})
    texts = [c for c, _ in s.calls]
    assert "MATCH (m:CascadeMachine) DETACH DELETE m" in texts
    edges = [p for c, p in s.calls if "MERGE (a)-[e:CASCADE_NEXT]->(b)" in c][0]["rows"]
    assert edges == run["edges"]
    proj = [p for c, p in s.calls if "gds.graph.project" in c]
    assert [p["rels"]["CASCADE_NEXT"]["orientation"] for p in proj] == ["NATURAL", "REVERSE"]
    cmp = out["seeds"]["7"]["compare"]
    assert cmp["components_equal"] and cmp["roots_equal"] and cmp["revpr_max_abs_diff"] == 0.0
    dm = out["demo"]
    assert dm["machine"] == "m_4" and dm["true_root"] == "m_1"            # last alarmed member of c1
    assert [q["key"] for q in dm["queries"]] == ["members", "paths"] and dm["queries"][0]["rows"]


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from experiments import e8_graph as E
    try:
        s = E.build_spark(master="local[2]", ckpt=tmp_path_factory.mktemp("ckpt"), app="test_e8c")
        import graphframes  # noqa: F401
    except Exception as exc:  # pragma: no cover - offline machine without the package in the Ivy cache
        pytest.skip(f"GraphFrames not available: {exc}")
    yield s
    s.stop()


def test_pregel_matches_reference_and_scoring(spark):
    inc = inc_df()
    r = C.analyse_graph(spark, "t", MACHINES, inc, truth())
    a = r["agreement"]
    assert a["spark_edges_equal_pandas"] and a["components_equal"] and a["roots_equal"]
    assert a["revpr_max_abs_diff"] < 1e-12
    ref = C.reference(MACHINES, inc)
    want = C.score_run(truth(), inc, ref["components"], ref["roots"], ref["revpr"], ref["chains"])
    assert r["score"]["summary"] == want["summary"]
    assert r["roots"]["m_1"] == ["m_1", "m_3"] and r["pregel_iterations"] == 7
    assert np.isclose(r["revpr"]["m_20"], 0.15)


def test_root_onset_diagnostic():
    inc = inc_df()
    ref = C.reference(MACHINES, inc)
    t = truth()
    t["cascades"][0]["faults"][2].update(role="dependent", offset_min=3)   # m_3: in the root's window
    res = C.score_run(t, inc, ref["components"], ref["roots"], ref["revpr"], ref["chains"])
    c1, c2, c3 = res["cascades"]
    assert c1["root_onset"] == "tied" and c1["tied_offsets_min"] == [3]
    assert c2["root_onset"] == "first" and c3["root_onset"] is None
    assert res["summary"]["root_onset"] == {"first": 1, "tied": 1, "later": 0}
