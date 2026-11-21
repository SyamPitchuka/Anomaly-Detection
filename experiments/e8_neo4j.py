"""E8, Neo4j part: the co-failure graphs of experiments/e8_graph.py in Neo4j 5 + Graph Data Science (GDS).

Run from the repo root after `python experiments/e8_graph.py pregel` (which writes data/e8/vertices.csv and
data/e8/edges_<graph>.csv, so both systems use the same graph):
    $env:NEO4J_PASSWORD = "<password>"                              # PowerShell (Git Bash: export NEO4J_PASSWORD=...)
    docker compose -f infra/docker-compose.neo4j.yml up -d          # Neo4j 5 Community + GDS plugin
    python experiments/e8_neo4j.py                                  # -> data/e8/neo4j.json
    python experiments/e8_graph.py report                           # -> docs/results/e8_graph.md / .json / .png

Load (neo4j Python driver, MERGE, idempotent): (:Machine {id, smd_group}) and one relationship type per graph,
(:Machine)-[:CO_FAIL_GAP10 | CO_FAIL_GAP0 | CO_FAIL_SIG {weight}]->(:Machine) with src < dst; gap10 relationships also
carry null_mean, p and beyond_chance. --no-reset keeps existing :Machine nodes (default: delete them first, so a
rerun with other incidents leaves no stale edges).
Algorithms per graph, on an UNDIRECTED GDS projection with concurrency 1 (deterministic): WCC (unweighted),
Louvain (weighted), Label Propagation (unweighted, to compare with GraphFrames LPA) and weighted PageRank
(d = 0.85, the same number of iterations as the Pregel version, tolerance 0 so it does not stop early).
Then three operator queries in plain Cypher (see QUERIES) and their rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments import e8_graph as G  # noqa: E402

URI = "bolt://localhost:7687"
USER = "neo4j"
REL = {"gap10": "CO_FAIL_GAP10", "gap0": "CO_FAIL_GAP0", "sig": "CO_FAIL_SIG"}
BATCH = 500

LOAD_NODES = """
UNWIND $rows AS row
MERGE (m:Machine {id: row.id})
SET m.smd_group = row.smd_group
"""
# Relationship types cannot be parameters in Cypher, hence one statement per type.
LOAD_EDGES = """
UNWIND $rows AS row
MATCH (a:Machine {id: row.src}), (b:Machine {id: row.dst})
MERGE (a)-[r:%s]->(b)
SET r.weight = row.weight, r.null_mean = row.null_mean, r.p = row.p, r.beyond_chance = row.sig
"""

QUERIES = [
    {"key": "partners", "title": "Q1. Which machines co-fail with {machine} (gap10), strongest first?",
     "cypher": """
MATCH (m:Machine {id: $machine})-[r:CO_FAIL_GAP10]-(o:Machine)
RETURN o.id AS machine, o.smd_group AS smd_group, r.weight AS overlapping_pairs,
       round(r.null_mean, 2) AS expected_by_chance, r.beyond_chance AS beyond_chance
ORDER BY overlapping_pairs DESC, machine
"""},
    {"key": "largest_group", "title": "Q2. Largest co-failure group (connected component of the beyond-chance graph)",
     "cypher": """
MATCH (m:Machine)
WITH m.sig_component AS component, collect(m.id) AS members
WHERE size(members) > 1
OPTIONAL MATCH (a:Machine)-[r:CO_FAIL_SIG]->(b:Machine)
WHERE a.sig_component = component AND b.sig_component = component
WITH component, members, count(r) AS edges, sum(r.weight) AS overlapping_pairs
RETURN size(members) AS size, members, edges, overlapping_pairs
ORDER BY size DESC, overlapping_pairs DESC
LIMIT 3
"""},
    {"key": "strongest_pairs", "title": "Q3. Strongest beyond-chance pairs, and do they cross SMD groups?",
     "cypher": """
MATCH (a:Machine)-[r:CO_FAIL_SIG]->(b:Machine)
RETURN a.id AS machine_a, b.id AS machine_b, r.weight AS overlapping_pairs,
       round(r.null_mean, 2) AS expected_by_chance, a.smd_group = b.smd_group AS same_smd_group
ORDER BY overlapping_pairs DESC, machine_a, machine_b
LIMIT 8
"""},
]


def read_graph_files(out_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    vert = pd.read_csv(out_dir / "vertices.csv", dtype=str)
    edges = {g: pd.read_csv(out_dir / f"edges_{g}.csv", dtype={"src": str, "dst": str}) for g in G.GRAPHS}
    return vert, edges


def edge_rows(e: pd.DataFrame) -> list[dict]:
    rows = []
    for r in e.itertuples():
        rows.append({"src": r.src, "dst": r.dst, "weight": int(r.weight),
                     "null_mean": float(r.null_mean) if "null_mean" in e else None,
                     "p": float(r.p) if "p" in e else None, "sig": bool(r.sig) if "sig" in e else None})
    return rows


def _batches(rows: list, n: int = BATCH):
    for k in range(0, len(rows), n):
        yield rows[k:k + n]


def load(session, vert: pd.DataFrame, edges: dict[str, pd.DataFrame], reset: bool = True) -> dict:
    if reset:
        session.run("MATCH (m:Machine) DETACH DELETE m").consume()
    session.run("CREATE CONSTRAINT machine_id IF NOT EXISTS FOR (m:Machine) REQUIRE m.id IS UNIQUE").consume()
    nodes = [{"id": r.id, "smd_group": str(r.smd_group)} for r in vert.itertuples()]
    for b in _batches(nodes):
        session.run(LOAD_NODES, rows=b).consume()
    counts = {}
    for name, e in edges.items():
        rows = edge_rows(e)
        for b in _batches(rows):
            session.run(LOAD_EDGES % REL[name], rows=b).consume()
        counts[name] = session.run(f"MATCH ()-[r:{REL[name]}]->() RETURN count(r) AS n").single()["n"]
    counts["nodes"] = session.run("MATCH (m:Machine) RETURN count(m) AS n").single()["n"]
    return counts


def _stream(session, cypher: str, value: str, **params) -> dict:
    return {r["id"]: r[value] for r in session.run(cypher, **params)}


def run_algorithms(session, name: str, iters: int = G.PR_ITERS, damping: float = G.DAMPING) -> dict:
    gname = f"e8_{name}"
    session.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=gname).consume()
    session.run("CALL gds.graph.project($g, 'Machine', $rels) YIELD graphName RETURN graphName", g=gname,
                rels={REL[name]: {"orientation": "UNDIRECTED", "properties": "weight"}}).consume()
    cfg = {"concurrency": 1}
    out = {
        "wcc": _stream(session, "CALL gds.wcc.stream($g, $c) YIELD nodeId, componentId "
                                "RETURN gds.util.asNode(nodeId).id AS id, componentId AS v", "v", g=gname, c=cfg),
        "louvain": _stream(session, "CALL gds.louvain.stream($g, $c) YIELD nodeId, communityId "
                                    "RETURN gds.util.asNode(nodeId).id AS id, communityId AS v", "v", g=gname,
                           c={**cfg, "relationshipWeightProperty": "weight"}),
        "lpa": _stream(session, "CALL gds.labelPropagation.stream($g, $c) YIELD nodeId, communityId "
                                "RETURN gds.util.asNode(nodeId).id AS id, communityId AS v", "v", g=gname,
                       c={**cfg, "maxIterations": G.LPA_ITERS}),
        "pagerank": _stream(session, "CALL gds.pageRank.stream($g, $c) YIELD nodeId, score "
                                     "RETURN gds.util.asNode(nodeId).id AS id, score AS v", "v", g=gname,
                            c={**cfg, "relationshipWeightProperty": "weight", "dampingFactor": damping,
                               "maxIterations": iters, "tolerance": 0.0}),
    }
    stats = session.run("CALL gds.louvain.stats($g, $c) YIELD modularity RETURN modularity", g=gname,
                        c={**cfg, "relationshipWeightProperty": "weight"}).single()
    out["louvain_modularity"] = float(stats["modularity"]) if stats else None
    if name == "sig":  # Q2 reads the component of every machine in the beyond-chance graph
        session.run("CALL gds.wcc.write($g, $c) YIELD componentCount RETURN componentCount", g=gname,
                    c={**cfg, "writeProperty": "sig_component"}).consume()
    session.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=gname).consume()
    return out


def run_queries(session, machine: str = G.FOCUS_MACHINE) -> list[dict]:
    out = []
    for q in QUERIES:
        rows = [dict(r) for r in session.run(q["cypher"], machine=machine)]
        out.append({"key": q["key"], "title": q["title"].format(machine=machine), "cypher": q["cypher"],
                    "rows": rows})
    return out


def versions(session) -> dict:
    comp = session.run("CALL dbms.components() YIELD versions, edition RETURN versions[0] AS v, edition AS e").single()
    gds = session.run("RETURN gds.version() AS v").single()
    return {"neo4j": f"{comp['v']} {comp['e']}" if comp else None, "gds": gds["v"] if gds else None}


def run(driver, out_dir: Path, machine: str = G.FOCUS_MACHINE, reset: bool = True) -> dict:
    vert, edges = read_graph_files(out_dir)
    timings, t_all = {}, time.perf_counter()
    with driver.session() as s:
        env = versions(s)
        t = time.perf_counter()
        counts = load(s, vert, edges, reset)
        timings["load_s"] = time.perf_counter() - t
        t = time.perf_counter()
        results = {name: run_algorithms(s, name) for name in G.GRAPHS}
        timings["algorithms_s"] = time.perf_counter() - t
        t = time.perf_counter()
        queries = run_queries(s, machine)
        timings["queries_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t_all
    return {"env": env, "loaded": counts, "results": results, "queries": queries, "timings": timings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", URI))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", USER))
    ap.add_argument("--out-dir", type=Path, default=G.OUT_DIR)
    ap.add_argument("--machine", default=G.FOCUS_MACHINE, help="machine for query Q1")
    ap.add_argument("--no-reset", action="store_true", help="do not delete existing :Machine nodes first")
    args = ap.parse_args(argv)
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        print("set NEO4J_PASSWORD (the password of infra/docker-compose.neo4j.yml)", file=sys.stderr)
        return 2
    from neo4j import GraphDatabase
    with GraphDatabase.driver(args.uri, auth=(args.user, password)) as driver:
        driver.verify_connectivity()
        out = run(driver, args.out_dir, args.machine, reset=not args.no_reset)
    path = args.out_dir / "neo4j.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for name in G.GRAPHS:
        r = out["results"][name]
        print(f"  {name}: {len(G.groups_of(r['wcc']))} WCC, {len(G.groups_of(r['louvain']))} Louvain communities "
              f"(modularity {r['louvain_modularity']:.3f}), top PageRank {G.top_k(r['pagerank'], 3)}")
    for q in out["queries"]:
        print(f"{q['title']}: {len(q['rows'])} rows")
        for row in q["rows"][:8]:
            print("   ", row)
    print(f"wrote {path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path}; "
          f"Neo4j part runtime {out['timings']['total_s']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
