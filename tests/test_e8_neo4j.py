"""experiments/e8_neo4j.py against a mock neo4j driver: MERGE load, GDS calls, operator queries, output JSON."""
import json
import re

import pandas as pd
import pytest

from experiments import e8_graph as E
from experiments import e8_neo4j as N

MACHINES = ["machine-1-1", "machine-1-2", "machine-2-1"]


class Result:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None

    def consume(self):
        return None

    def data(self):
        return self.rows


class FakeSession:
    """Answers the statements e8_neo4j sends; records every (cypher, params)."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        c = " ".join(cypher.split())
        if "dbms.components" in c:
            return Result([{"v": "5.26.0", "e": "community"}])
        if "gds.version" in c:
            return Result([{"v": "2.13.2"}])
        if c.startswith("MATCH ()-[r:"):
            return Result([{"n": 2}])
        if "count(m) AS n" in c:
            return Result([{"n": 3}])
        if "gds.wcc.stream" in c or "gds.louvain.stream" in c or "gds.labelPropagation.stream" in c:
            return Result([{"id": m, "v": 0 if m != "machine-2-1" else 1} for m in MACHINES])
        if "gds.pageRank.stream" in c:
            return Result([{"id": m, "v": 1.0 + k} for k, m in enumerate(MACHINES)])
        if "gds.louvain.stats" in c:
            return Result([{"modularity": 0.25}])
        if "$machine" in c:
            return Result([{"machine": "machine-1-2", "overlapping_pairs": 2}])
        return Result([])


class FakeDriver:
    def __init__(self):
        self.s = FakeSession()

    def session(self):
        return self.s


def write_graph(tmp_path):
    vert = E.vertices(MACHINES)
    e10 = pd.DataFrame({"src": ["machine-1-1", "machine-1-2"], "dst": ["machine-1-2", "machine-2-1"],
                        "weight": [2, 1], "null_mean": [0.5, 0.9], "p": [0.01, 0.5], "sig": [True, False]})
    E.write_graph_files(tmp_path, vert, {"gap10": e10, "gap0": e10.iloc[:1], "sig": e10.iloc[:1]})


def test_run_loads_with_merge_and_calls_gds(tmp_path):
    write_graph(tmp_path)
    d = FakeDriver()
    out = N.run(d, tmp_path, machine="machine-1-1")
    calls = d.s.calls
    texts = [c for c, _ in calls]
    assert texts[2] == "MATCH (m:Machine) DETACH DELETE m"  # after the two version queries
    node_calls = [p for c, p in calls if "MERGE (m:Machine" in c]
    assert node_calls and node_calls[0]["rows"][0] == {"id": "machine-1-1", "smd_group": "1"}
    for name, rel in N.REL.items():
        merges = [p for c, p in calls if re.search(rf"MERGE \(a\)-\[r:{rel}\]->\(b\)", c)]
        assert merges, rel
    gap10 = [p for c, p in calls if "CO_FAIL_GAP10]->(b)" in c][0]["rows"]
    assert gap10[0] == {"src": "machine-1-1", "dst": "machine-1-2", "weight": 2, "null_mean": 0.5, "p": 0.01,
                        "sig": True}
    proj = [p for c, p in calls if "gds.graph.project" in c]
    assert [list(p["rels"]) for p in proj] == [["CO_FAIL_GAP10"], ["CO_FAIL_GAP0"], ["CO_FAIL_SIG"]]
    assert all(p["rels"][r]["orientation"] == "UNDIRECTED" for p in proj for r in p["rels"])
    pr = [p for c, p in calls if "gds.pageRank.stream" in c][0]["c"]
    assert pr["dampingFactor"] == E.DAMPING and pr["maxIterations"] == E.PR_ITERS and pr["tolerance"] == 0.0
    assert sum("gds.wcc.write" in c for c in texts) == 1  # only for the sig graph (query Q2)
    assert out["env"] == {"neo4j": "5.26.0 community", "gds": "2.13.2"}
    assert out["results"]["gap10"]["wcc"] == {"machine-1-1": 0, "machine-1-2": 0, "machine-2-1": 1}
    assert out["results"]["sig"]["louvain_modularity"] == 0.25
    assert [q["key"] for q in out["queries"]] == ["partners", "largest_group", "strongest_pairs"]
    assert out["queries"][0]["rows"] == [{"machine": "machine-1-2", "overlapping_pairs": 2}]
    assert "machine-1-1" in out["queries"][0]["title"]
    json.dumps(out, default=str)


def test_no_reset_keeps_nodes(tmp_path):
    write_graph(tmp_path)
    d = FakeDriver()
    N.run(d, tmp_path, reset=False)
    assert not any("DETACH DELETE" in c for c, _ in d.s.calls)


def test_main_needs_password(monkeypatch, tmp_path):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    assert N.main(["--out-dir", str(tmp_path)]) == 2


def test_main_with_mock_driver(monkeypatch, tmp_path):
    write_graph(tmp_path)
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
        return Ctx()

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", staticmethod(fake_driver))
    monkeypatch.setenv("NEO4J_PASSWORD", "secret-pw")
    assert N.main(["--out-dir", str(tmp_path)]) == 0
    assert seen == {"uri": N.URI, "auth": ("neo4j", "secret-pw")}
    assert json.loads((tmp_path / "neo4j.json").read_text())["loaded"]["nodes"] == 3
