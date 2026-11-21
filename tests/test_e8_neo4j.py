"""experiments/e8_neo4j.py against a mock neo4j driver: MERGE load, GDS calls, operator queries, output JSON."""
import json
import re

import pandas as pd
import pytest

from experiments import e8_graph as E
from experiments import e8_neo4j as N

SMD = ["machine-1-1", "machine-1-2", "machine-2-1"]
ALI = ["m_1", "m_2", "m_3"]


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
    """Answers the statements e8_neo4j sends; records every (cypher, params)."""

    def __init__(self, machines):
        self.calls = []
        self.machines = machines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        c = " ".join(cypher.split())
        ms = self.machines
        if "dbms.components" in c:
            return Result([{"v": "5.26.0", "e": "community"}])
        if "gds.version" in c:
            return Result([{"v": "2.13.2"}])
        if c.startswith("MATCH ()-[r:"):
            return Result([{"n": 2}])
        if "count(m) AS n" in c:
            return Result([{"n": 3}])
        if "gds.wcc.stream" in c or "gds.louvain.stream" in c or "gds.labelPropagation.stream" in c:
            return Result([{"id": m, "v": 0 if m != ms[2] else 1} for m in ms])
        if "gds.pageRank.stream" in c:
            return Result([{"id": m, "v": 1.0 + k} for k, m in enumerate(ms)])
        if "gds.louvain.stats" in c:
            return Result([{"modularity": 0.25}])
        if "$machine" in c:
            return Result([{"machine": ms[1], "overlapping_pairs": 2}])
        return Result([])


class FakeDriver:
    def __init__(self, machines=SMD):
        self.s = FakeSession(machines)

    def session(self):
        return self.s


def write_graph(tmp_path, source="smd", ms=SMD):
    vert = E.vertices(ms)
    e10 = pd.DataFrame({"src": [ms[0], ms[1]], "dst": [ms[1], ms[2]],
                        "weight": [2, 1], "null_mean": [0.5, 0.9], "p": [0.01, 0.5], "sig": [True, False]})
    sig = pd.DataFrame({"src": [ms[1], ms[0]], "dst": [ms[2], ms[2]], "weight": [3, 1], "null_mean": [0.1, 0.1],
                        "p": [0.01, 0.01], "sig": [True, True]})
    E.write_graph_files(tmp_path, vert, {"gap10": e10, "gap0": e10.iloc[:1], "sig": sig}, source)


def test_names_per_source():
    assert N.names("smd") == {"label": "Machine", "gap10": "CO_FAIL_GAP10", "gap0": "CO_FAIL_GAP0",
                              "sig": "CO_FAIL_SIG"}
    assert N.names("alibaba")["label"] == "AlibabaMachine" and N.names("alibaba")["sig"] == "ALI_CO_FAIL_SIG"
    q3 = N.QUERIES[2]
    assert "same_smd_group" in N.query_text(q3, "smd") and "same_smd_group" not in N.query_text(q3, "alibaba")
    assert ":AlibabaMachine" in N.query_text(N.QUERIES[0], "alibaba")
    assert "ALI_CO_FAIL_GAP10" in N.query_text(N.QUERIES[0], "alibaba")


def test_focus_machine(tmp_path):
    write_graph(tmp_path, "alibaba", ALI)
    vert, edges = E.read_graph_files(tmp_path, "alibaba")
    assert N.focus_machine(edges, vert, "alibaba") == "m_3"  # two sig partners, most weight
    assert N.focus_machine(edges, vert, "smd") == "machine-1-1"  # fixed for SMD
    empty = {g: edges[g].iloc[:0] for g in E.GRAPHS}
    assert N.focus_machine(empty, vert, "alibaba") == "m_1"


def test_run_loads_with_merge_and_calls_gds(tmp_path):
    write_graph(tmp_path)
    d = FakeDriver()
    out = N.run(d, tmp_path, "smd", machine="machine-1-1")
    calls = d.s.calls
    texts = [c for c, _ in calls]
    assert texts[2] == "MATCH (m:Machine) DETACH DELETE m"  # after the two version queries
    node_calls = [p for c, p in calls if "MERGE (m:Machine" in c]
    assert node_calls and node_calls[0]["rows"][0] == {"id": "machine-1-1", "group": "1"}
    for rel in ("CO_FAIL_GAP10", "CO_FAIL_GAP0", "CO_FAIL_SIG"):
        assert [p for c, p in calls if re.search(rf"MERGE \(a\)-\[r:{rel}\]->\(b\)", c)], rel
    gap10 = [p for c, p in calls if "CO_FAIL_GAP10]->(b)" in c][0]["rows"]
    assert gap10[0] == {"src": "machine-1-1", "dst": "machine-1-2", "weight": 2, "null_mean": 0.5, "p": 0.01,
                        "sig": True}
    proj = [p for c, p in calls if "gds.graph.project" in c]
    assert [list(p["rels"]) for p in proj] == [["CO_FAIL_GAP10"], ["CO_FAIL_GAP0"], ["CO_FAIL_SIG"]]
    assert all(p["label"] == "Machine" for p in proj)
    assert all(p["rels"][r]["orientation"] == "UNDIRECTED" for p in proj for r in p["rels"])
    pr = [p for c, p in calls if "gds.pageRank.stream" in c][0]["c"]
    assert pr["dampingFactor"] == E.DAMPING and pr["maxIterations"] == E.PR_ITERS and pr["tolerance"] == 0.0
    assert sum("gds.wcc.write" in c for c in texts) == 1  # only for the sig graph (query Q2)
    assert out["env"] == {"neo4j": "5.26.0 community", "gds": "2.13.2"}
    assert out["results"]["gap10"]["wcc"] == {"machine-1-1": 0, "machine-1-2": 0, "machine-2-1": 1}
    assert out["results"]["sig"]["louvain_modularity"] == 0.25
    assert [q["key"] for q in out["queries"]] == ["partners", "largest_group", "strongest_pairs"]
    assert out["queries"][0]["rows"] == [{"machine": "machine-1-2", "overlapping_pairs": 2}]
    assert "machine-1-1" in out["queries"][0]["title"] and "cross SMD groups" in out["queries"][2]["title"]
    json.dumps(out, default=str)


def test_run_alibaba_uses_its_own_label_and_types(tmp_path):
    write_graph(tmp_path, "alibaba", ALI)
    d = FakeDriver(ALI)
    out = N.run(d, tmp_path, "alibaba")
    texts = [c for c, _ in d.s.calls]
    assert "MATCH (m:AlibabaMachine) DETACH DELETE m" in texts
    assert not any("(m:Machine" in c or ":CO_FAIL_" in c.replace("ALI_CO_FAIL_", "") for c in texts)
    proj = [p for c, p in d.s.calls if "gds.graph.project" in c]
    assert [p["g"] for p in proj] == ["e8_alibaba_gap10", "e8_alibaba_gap0", "e8_alibaba_sig"]
    assert out["focus_machine"] == "m_3" and "m_3" in out["queries"][0]["title"]
    assert [p for c, p in d.s.calls if "$machine" in c][0]["machine"] == "m_3"


def test_no_reset_keeps_nodes(tmp_path):
    write_graph(tmp_path)
    d = FakeDriver()
    N.run(d, tmp_path, reset=False)
    assert not any("DETACH DELETE" in c for c, _ in d.s.calls)


def test_main_needs_password(monkeypatch, tmp_path):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    assert N.main(["--out-dir", str(tmp_path)]) == 2


def test_main_with_mock_driver(monkeypatch, tmp_path):
    write_graph(tmp_path, "alibaba", ALI)
    neo4j = pytest.importorskip("neo4j")

    class Ctx(FakeDriver):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def verify_connectivity(self):
            return None

    seen = {}

    def fake_driver(uri, auth):
        seen["uri"], seen["auth"] = uri, auth
        return Ctx(ALI)

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", staticmethod(fake_driver))
    monkeypatch.setenv("NEO4J_PASSWORD", "secret-pw")
    assert N.main(["--source", "alibaba", "--out-dir", str(tmp_path)]) == 0
    assert seen == {"uri": N.URI, "auth": ("neo4j", "secret-pw")}
    out = json.loads((tmp_path / "neo4j_alibaba.json").read_text())
    assert out["loaded"]["nodes"] == 3 and out["source"] == "alibaba"
