"""E8, Neo4j part: the co-failure graphs of experiments/e8_graph.py in Neo4j 5 + Graph Data Science (GDS).

Run from the repo root after `python experiments/e8_graph.py pregel --source <s>` (which writes
data/e8/vertices<suffix>.csv and data/e8/edges<suffix>_<graph>.csv, so both systems use the same graph):
    $env:NEO4J_PASSWORD = "<password>"                              # PowerShell (Git Bash: export NEO4J_PASSWORD=...)
    docker compose -f infra/docker-compose.neo4j.yml up -d          # Neo4j 5 Community + GDS plugin
    python experiments/e8_neo4j.py --source smd                     # -> data/e8/neo4j.json
    python experiments/e8_neo4j.py --source alibaba                 # -> data/e8/neo4j_alibaba.json
    python experiments/e8_graph.py report                           # -> docs/results/e8_graph.md / .json / .png

Load (neo4j Python driver, MERGE, idempotent), one node label and set of relationship types per source so both
graphs live side by side: SMD (:Machine)-[:CO_FAIL_GAP10 | CO_FAIL_GAP0 | CO_FAIL_SIG]->(:Machine), Alibaba
(:AlibabaMachine)-[:ALI_CO_FAIL_GAP10 | ALI_CO_FAIL_GAP0 | ALI_CO_FAIL_SIG]->(:AlibabaMachine); nodes {id, group},
relationships {weight, null_mean, p, beyond_chance} with src < dst. --no-reset keeps existing nodes of the source's
label (default: delete them first, so a rerun with other incidents leaves no stale edges).
Algorithms per graph, on an UNDIRECTED GDS projection with concurrency 1 (deterministic): WCC (unweighted),
Louvain (weighted), Label Propagation (unweighted, to compare with GraphFrames LPA) and weighted PageRank
(d = 0.85, the same number of iterations as the Pregel version, tolerance 0 so it does not stop early).
Then three operator queries in plain Cypher (see QUERIES) and their rows. Q1's machine: --machine, else
SOURCES[source]['focus'] (machine-1-1 for SMD), else the machine with the most beyond-chance partners (ties: more
co-occurring pairs, then id).
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
BATCH = 500
SCHEMA = {"smd": {"label": "Machine", "prefix": ""}, "alibaba": {"label": "AlibabaMachine", "prefix": "ALI_"}}

LOAD_NODES = """
UNWIND $rows AS row
MERGE (m:{label} {{id: row.id}})
SET m.group = row.group
"""
# Labels and relationship types cannot be parameters in Cypher, hence the formatting.
LOAD_EDGES = """
UNWIND $rows AS row
MATCH (a:{label} {{id: row.src}}), (b:{label} {{id: row.dst}})
MERGE (a)-[r:{rel}]->(b)
SET r.weight = row.weight, r.null_mean = row.null_mean, r.p = row.p, r.beyond_chance = row.sig
"""

QUERIES = [
    {"key": "partners", "title": "Q1. Which machines co-fail with {machine} (gap10), strongest first?",
     "cypher": """
MATCH (m:{label} {{id: $machine}})-[r:{gap10}]-(o:{label})
RETURN o.id AS machine, o.group AS group, r.weight AS overlapping_pairs,
       round(r.null_mean, 2) AS expected_by_chance, r.beyond_chance AS beyond_chance
ORDER BY overlapping_pairs DESC, machine
LIMIT 15
"""},
    {"key": "largest_group", "title": "Q2. Largest co-failure groups (connected components of the beyond-chance graph)",
     "cypher": """
MATCH (m:{label})
WITH m.sig_component AS component, collect(m.id) AS members
WHERE size(members) > 1
OPTIONAL MATCH (a:{label})-[r:{sig}]->(b:{label})
WHERE a.sig_component = component AND b.sig_component = component
WITH component, members, count(r) AS edges, sum(r.weight) AS overlapping_pairs
RETURN size(members) AS size, members, edges, overlapping_pairs
ORDER BY size DESC, overlapping_pairs DESC
LIMIT 3
"""},
    {"key": "strongest_pairs", "title": "Q3. Strongest beyond-chance pairs{group_note}",
     "cypher": """
MATCH (a:{label})-[r:{sig}]->(b:{label})
RETURN a.id AS machine_a, b.id AS machine_b, r.weight AS overlapping_pairs,
       round(r.null_mean, 2) AS expected_by_chance{group_col}
ORDER BY overlapping_pairs DESC, machine_a, machine_b
LIMIT 8
"""},
]


def names(source: str) -> dict:
    s = SCHEMA[source]
    return {"label": s["label"], **{g: f"{s['prefix']}CO_FAIL_{g.upper()}" for g in G.GRAPHS}}


def query_text(q: dict, source: str) -> str:
    n = names(source)
    group_col = ",\n       a.group = b.group AS same_smd_group" if source == "smd" else ""
    return q["cypher"].format(label=n["label"], gap10=n["gap10"], sig=n["sig"], group_col=group_col)


def focus_machine(edges: dict[str, pd.DataFrame], vert: pd.DataFrame, source: str) -> str:
    """SOURCES focus, else the machine with the most beyond-chance partners (then weight, then id)."""
    fixed = G.SOURCES[source]["focus"]
    if fixed:
        return fixed
    for name in ("sig", "gap10"):
        e = edges[name]
        if len(e):
            both = pd.concat([e[["src", "weight"]].rename(columns={"src": "m"}),
                              e[["dst", "weight"]].rename(columns={"dst": "m"})])
            agg = both.groupby("m").agg(partners=("weight", "size"), weight=("weight", "sum")).reset_index()
            agg = agg.sort_values(["partners", "weight", "m"], ascending=[False, False, True])
            return str(agg.iloc[0]["m"])
    return str(vert["id"].iloc[0])


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


def load(session, vert: pd.DataFrame, edges: dict[str, pd.DataFrame], source: str = "smd",
         reset: bool = True) -> dict:
    n = names(source)
    label = n["label"]
    if reset:
        session.run(f"MATCH (m:{label}) DETACH DELETE m").consume()
    session.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (m:{label}) REQUIRE m.id IS UNIQUE").consume()
    nodes = [{"id": r.id, "group": str(r.group)} for r in vert.itertuples()]
    for b in _batches(nodes):
        session.run(LOAD_NODES.format(label=label), rows=b).consume()
    counts = {}
    for name, e in edges.items():
        for b in _batches(edge_rows(e)):
            session.run(LOAD_EDGES.format(label=label, rel=n[name]), rows=b).consume()
        counts[name] = session.run(f"MATCH ()-[r:{n[name]}]->() RETURN count(r) AS n").single()["n"]
    counts["nodes"] = session.run(f"MATCH (m:{label}) RETURN count(m) AS n").single()["n"]
    return counts


def _stream(session, cypher: str, value: str, **params) -> dict:
    return {r["id"]: r[value] for r in session.run(cypher, **params)}


def run_algorithms(session, name: str, source: str = "smd", iters: int = G.PR_ITERS,
                   damping: float = G.DAMPING) -> dict:
    n = names(source)
    gname = f"e8_{source}_{name}"
    session.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=gname).consume()
    session.run("CALL gds.graph.project($g, $label, $rels) YIELD graphName RETURN graphName", g=gname,
                label=n["label"], rels={n[name]: {"orientation": "UNDIRECTED", "properties": "weight"}}).consume()
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


def run_queries(session, machine: str, source: str = "smd") -> list[dict]:
    out = []
    note = ", and do they cross SMD groups?" if source == "smd" else ""
    for q in QUERIES:
        cypher = query_text(q, source)
        rows = [dict(r) for r in session.run(cypher, machine=machine)]
        out.append({"key": q["key"], "title": q["title"].format(machine=machine, group_note=note),
                    "cypher": cypher, "rows": rows})
    return out


def versions(session) -> dict:
    comp = session.run("CALL dbms.components() YIELD versions, edition RETURN versions[0] AS v, edition AS e").single()
    gds = session.run("RETURN gds.version() AS v").single()
    return {"neo4j": f"{comp['v']} {comp['e']}" if comp else None, "gds": gds["v"] if gds else None}


def run(driver, out_dir: Path, source: str = "smd", machine: str | None = None, reset: bool = True) -> dict:
    vert, edges = G.read_graph_files(out_dir, source)
    machine = machine or focus_machine(edges, vert, source)
    timings, t_all = {}, time.perf_counter()
    with driver.session() as s:
        env = versions(s)
        t = time.perf_counter()
        counts = load(s, vert, edges, source, reset)
        timings["load_s"] = time.perf_counter() - t
        t = time.perf_counter()
        results = {name: run_algorithms(s, name, source) for name in G.GRAPHS}
        timings["algorithms_s"] = time.perf_counter() - t
        t = time.perf_counter()
        queries = run_queries(s, machine, source)
        timings["queries_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t_all
    return {"source": source, "env": env, "schema": names(source), "focus_machine": machine, "loaded": counts,
            "results": results, "queries": queries, "timings": timings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=sorted(SCHEMA), default="smd")
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", URI))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", USER))
    ap.add_argument("--out-dir", type=Path, default=G.OUT_DIR)
    ap.add_argument("--machine", default=None, help="machine for query Q1 (default: see module docstring)")
    ap.add_argument("--no-reset", action="store_true", help="do not delete the source's existing nodes first")
    args = ap.parse_args(argv)
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        print("set NEO4J_PASSWORD (the password of infra/docker-compose.neo4j.yml)", file=sys.stderr)
        return 2
    from neo4j import GraphDatabase
    with GraphDatabase.driver(args.uri, auth=(args.user, password)) as driver:
        driver.verify_connectivity()
        out = run(driver, args.out_dir, args.source, args.machine, reset=not args.no_reset)
    path = G.out_file(args.out_dir, "neo4j", args.source, "json")
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for name in G.GRAPHS:
        r = out["results"][name]
        print(f"  {name}: {len(G.groups_of(r['wcc']))} WCC, {len(G.groups_of(r['louvain']))} Louvain communities "
              f"(modularity {r['louvain_modularity']:.3f}), top PageRank {G.top_k(r['pagerank'], 3)}")
    for q in out["queries"]:
        print(f"{q['title']}: {len(q['rows'])} rows")
        for row in q["rows"][:8]:
            print("   ", row)
    print(f"wrote {G.rel_path(path)}; Neo4j part ({args.source}) runtime {out['timings']['total_s']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
