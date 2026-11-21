"""E8: graph processing on SMD incidents -- which machines fail together? Pregel part (Spark GraphFrames) + report.

Run from the repo root (venv active). Three steps; this file does 1 and 3, experiments/e8_neo4j.py does 2:
    python experiments/e8_graph.py pregel        # 1. graphs + null model + GraphFrames CC / LPA / Pregel PageRank
    python experiments/e8_neo4j.py               # 2. same graphs in Neo4j: GDS WCC / Louvain / PageRank + Cypher
    python experiments/e8_graph.py report        # 3. docs/results/e8_graph.md + .json + .png (needs 1 and 2)

Input: data/incidents/source=smd (the incidents layer, columns of LAYERS["incidents"]); --incidents DIR overrides.
Machine time spans for the null model come from the scores the incidents were built from (--scores DIR, default
data/scores/source=smd); a machine missing there spans its own first..last incident.

Graph: one vertex per machine (property smd_group = 1, 2 or 3 from machine-<g>-<k>). Two incidents a, b of
different machines co-occur with gap G minutes when  max(a.start, b.start) - min(a.end, b.end) <= G
(the second starts at most G minutes after the first ends); for G = 0 the intervals must really overlap (< 0,
touching end = start does not count). Edge weight = number of co-occurring incident pairs of the two machines.
Graphs: gap10 (main, G = 10, the incident session gap), gap0 (sensitivity, G = 0) and sig = the gap10 edges whose
weight is beyond chance (below).

Null model: each machine's incident times are shuffled N_SHUFFLES = 100 times, independently per machine: the
incident durations and the idle gaps between them (incl. before the first and after the last incident, within the
machine's span) are permuted, so every machine keeps its number of incidents, their durations and its span; only the
timing relative to other machines is destroyed. Reported: edges and total weight expected by chance. An edge is beyond
chance when its observed weight is larger than in all 100 shuffles (empirical p = 1/101); the number of such edges
expected by chance is estimated leave-one-out (each shuffle tested against the other 99).

Pregel part (all on the GraphFrames GraphFrame of each graph): connectedComponents, labelPropagation (GraphFrames
built-ins, both GraphX Pregel underneath) and weighted PageRank written with the GraphFrames Pregel API
(graphframes.lib.Pregel), with the Neo4j GDS formula  rank(v) = (1 - d) + d * sum_u rank(u) * w(u,v) / wdeg(u),
d = 0.85, PR_ITERS iterations, edges in both directions, starting from rank 0 (so iteration 1 gives 1 - d, which is
what GDS does: checked, GDS with maxIterations k equals this formula after k iterations from 0, to 1e-15), so the two
systems can be compared number for number.

GraphFrames 0.8.3 for Spark 3.5 (Scala 2.12) comes from the spark-packages repository via spark.jars.packages
(locally) or --packages (spark-submit). Its Python module is inside the jar; Spark puts the jar on sys.path.

Outputs: data/e8/vertices.csv, data/e8/edges_<graph>.csv (read by e8_neo4j.py: the same graph), data/e8/pregel.json.
The pipeline (scores, incidents, models) is not changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from contracts import schemas as S  # noqa: E402

SOURCE = "smd"
GAP_MAIN = 10
GAP_SENS = 0
GRAPHS = ("gap10", "gap0", "sig")
N_SHUFFLES = 100
SEED = 8
DAMPING = 0.85
PR_ITERS = 30
PR_INIT = 0.0  # the GDS starting point (see module docstring)
LPA_ITERS = 5
# Checkpoint the Pregel vertex state every iteration: driver-side planning of the growing plan dominated otherwise
# (30 iterations on 28 vertices, local[*]: 28 s with 1, 33 s with 2, 61-153 s with 5).
PREGEL_CKPT_EVERY = 1
TOP_K = 5
FOCUS_MACHINE = "machine-1-1"
GRAPHFRAMES_PACKAGE = "graphframes:graphframes:0.8.3-spark3.5-s_2.12"
SPARK_PACKAGES_REPO = "https://repos.spark-packages.org"

OUT_DIR = ROOT / "data" / "e8"
INCIDENTS_DIR = ROOT / S.PATHS["incidents"] / f"source={SOURCE}"
SCORES_DIR = ROOT / S.PATHS["scores"] / f"source={SOURCE}"
CKPT_DIR = ROOT / "checkpoints" / "e8_graph"
DOC_MD = ROOT / "docs" / "results" / "e8_graph.md"
DOC_JSON = ROOT / "docs" / "results" / "e8_graph.json"
DOC_PNG = ROOT / "docs" / "results" / "e8_graph.png"


# ---- graph construction (pandas / numpy; Spark builds the same edges in spark_edges) ----------------------------
def smd_group(entity_id: str) -> str:
    """'machine-2-7' -> '2' (the SMD machine group)."""
    parts = entity_id.split("-")
    return parts[1] if len(parts) == 3 else "?"


def _minutes(col: pd.Series) -> np.ndarray:
    # Not astype(int64): pandas 2 keeps Parquet's microsecond unit, so the integer would not be nanoseconds.
    ts = pd.to_datetime(col, utc=True)
    return ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy(np.int64)


def load_incidents(path: Path) -> pd.DataFrame:
    """entity_id, start, end (int minutes since 1970), sorted by machine and start."""
    df = pd.read_parquet(path, columns=["entity_id", "start_time", "end_time"])
    out = pd.DataFrame({"entity_id": df["entity_id"].astype(str), "start": _minutes(df["start_time"]),
                        "end": _minutes(df["end_time"])})
    return out.sort_values(["entity_id", "start"]).reset_index(drop=True)


def load_spans(scores_path: Path | None, inc: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Per machine (first window_start, last window_end) in minutes; falls back to the incidents' own range."""
    spans = {m: (int(g["start"].min()), int(g["end"].max())) for m, g in inc.groupby("entity_id")}
    if scores_path is not None and Path(scores_path).exists():
        sc = pd.read_parquet(scores_path, columns=["entity_id", "window_start", "window_end"])
        sc = sc[sc["entity_id"].isin(spans)]
        for m, g in sc.groupby("entity_id"):
            lo, hi = int(_minutes(g["window_start"]).min()), int(_minutes(g["window_end"]).max())
            spans[m] = (min(lo, spans[m][0]), max(hi, spans[m][1]))
    return spans


def co_occur(gap: float, s1, e1, s2, e2):
    """True where two incidents co-occur with the allowed gap (see module docstring)."""
    g = np.maximum(s1, s2) - np.minimum(e1, e2)
    return g <= gap if gap > 0 else g < 0


def edge_weights(inc: pd.DataFrame, gap: float) -> pd.DataFrame:
    """Columns src, dst (src < dst), weight = number of co-occurring incident pairs; sorted by src, dst."""
    ent = inc["entity_id"].to_numpy()
    s, e = inc["start"].to_numpy(), inc["end"].to_numpy()
    i, j = np.triu_indices(len(inc), 1)
    ok = (ent[i] != ent[j]) & co_occur(gap, s[i], e[i], s[j], e[j])
    a, b = ent[i][ok], ent[j][ok]
    pairs = pd.DataFrame({"src": np.minimum(a, b), "dst": np.maximum(a, b)})
    if pairs.empty:
        return pd.DataFrame({"src": pd.Series(dtype=str), "dst": pd.Series(dtype=str),
                             "weight": pd.Series(dtype=np.int64)})
    w = pairs.groupby(["src", "dst"]).size().rename("weight").reset_index()
    w["weight"] = w["weight"].astype(np.int64)
    return w.sort_values(["src", "dst"]).reset_index(drop=True)


def vertices(machines) -> pd.DataFrame:
    ms = sorted(set(machines))
    return pd.DataFrame({"id": ms, "smd_group": [smd_group(m) for m in ms]})


# ---- null model ---------------------------------------------------------------------------------------------------
def shuffle_incidents(inc: pd.DataFrame, spans: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Per machine: permute incident durations and the idle gaps (before, between, after) within the span."""
    parts = []
    for m, g in inc.groupby("entity_id", sort=True):
        s, e = g["start"].to_numpy(), g["end"].to_numpy()
        t0, t1 = spans[m]
        dur = e - s
        idle = np.concatenate([[s[0] - t0], s[1:] - e[:-1], [t1 - e[-1]]])
        idle = np.maximum(idle, 0)
        dur, idle = rng.permutation(dur), rng.permutation(idle)
        starts = t0 + np.cumsum(idle[:-1]) + np.concatenate([[0], np.cumsum(dur)[:-1]])
        parts.append(pd.DataFrame({"entity_id": m, "start": starts, "end": starts + dur}))
    return pd.concat(parts, ignore_index=True)


def all_pairs(machines) -> list[tuple[str, str]]:
    ms = sorted(set(machines))
    return [(a, b) for k, a in enumerate(ms) for b in ms[k + 1:]]


def pair_vector(edges: pd.DataFrame, pairs: list[tuple[str, str]]) -> np.ndarray:
    lookup = {(r.src, r.dst): int(r.weight) for r in edges.itertuples()}
    return np.array([lookup.get(p, 0) for p in pairs], dtype=np.int64)


def null_model(inc: pd.DataFrame, spans: dict, gaps=(GAP_MAIN, GAP_SENS), n: int = N_SHUFFLES,
               seed: int = SEED) -> dict:
    """{gap: (n, n_pairs) array of pair weights over the shuffles}; one shuffle feeds every gap."""
    rng = np.random.default_rng(seed)
    pairs = all_pairs(inc["entity_id"])
    out = {g: np.zeros((n, len(pairs)), dtype=np.int64) for g in gaps}
    for k in range(n):
        sh = shuffle_incidents(inc, spans, rng)
        for g in gaps:
            out[g][k] = pair_vector(edge_weights(sh, g), pairs)
    return out


def null_summary(observed: pd.DataFrame, null_w: np.ndarray) -> dict:
    n_edges = (null_w > 0).sum(axis=1)
    total = null_w.sum(axis=1)
    return {"observed_edges": int(len(observed)), "observed_weight": int(observed["weight"].sum()),
            "chance_edges_mean": float(n_edges.mean()), "chance_edges_sd": float(n_edges.std(ddof=1)),
            "chance_edges_p95": float(np.percentile(n_edges, 95)),
            "chance_weight_mean": float(total.mean()), "chance_weight_sd": float(total.std(ddof=1)),
            "chance_weight_p95": float(np.percentile(total, 95)),
            "p_edges": float((1 + (n_edges >= len(observed)).sum()) / (1 + len(n_edges))),
            "p_weight": float((1 + (total >= observed["weight"].sum()).sum()) / (1 + len(total)))}


def significant_edges(observed: pd.DataFrame, null_w: np.ndarray, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Observed edges with empirical p = (1 + #shuffles with weight >= observed) / (1 + n); sig = p is minimal
    (observed weight above every shuffle). Adds columns null_mean, p, sig."""
    idx = {p: k for k, p in enumerate(pairs)}
    n = null_w.shape[0]
    rows = []
    for r in observed.itertuples():
        col = null_w[:, idx[(r.src, r.dst)]]
        p = (1 + int((col >= r.weight).sum())) / (1 + n)
        rows.append({"src": r.src, "dst": r.dst, "weight": int(r.weight), "null_mean": float(col.mean()),
                     "p": p, "sig": bool(p <= 1 / (1 + n) + 1e-12)})
    cols = ["src", "dst", "weight", "null_mean", "p", "sig"]
    return pd.DataFrame(rows, columns=cols)


def chance_sig_edges(null_w: np.ndarray) -> float:
    """Leave-one-out: mean number of pairs a shuffle would call beyond chance against the other n-1 shuffles."""
    n = null_w.shape[0]
    counts = []
    for k in range(n):
        others = np.delete(null_w, k, axis=0)
        counts.append(int(((null_w[k] > 0) & (null_w[k] > others.max(axis=0))).sum()))
    return float(np.mean(counts))


# ---- partitions and comparisons -----------------------------------------------------------------------------------
def groups_of(labels: dict) -> list[list[str]]:
    """{machine: label} -> sorted list of sorted member lists (largest first)."""
    by: dict = {}
    for m, lab in labels.items():
        by.setdefault(lab, []).append(m)
    return sorted((sorted(v) for v in by.values()), key=lambda v: (-len(v), v))


def same_partition(a: dict, b: dict) -> bool:
    return set(a) == set(b) and groups_of(a) == groups_of(b)


def ari(a: dict, b: dict) -> float:
    from sklearn.metrics import adjusted_rand_score
    keys = sorted(set(a) & set(b))
    return float(adjusted_rand_score([str(a[k]) for k in keys], [str(b[k]) for k in keys]))


def top_k(scores: dict, k: int = TOP_K) -> list[str]:
    return [m for m, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def rank_compare(a: dict, b: dict, k: int = TOP_K) -> dict:
    from scipy.stats import spearmanr
    keys = sorted(set(a) & set(b))
    rho = spearmanr([a[m] for m in keys], [b[m] for m in keys]).statistic if len(keys) > 2 else float("nan")
    return {"top_a": top_k(a, k), "top_b": top_k(b, k), "top_same_set": set(top_k(a, k)) == set(top_k(b, k)),
            "top_same_order": top_k(a, k) == top_k(b, k), "spearman": float(rho),
            "max_abs_diff": float(max(abs(a[m] - b[m]) for m in keys)) if keys else float("nan")}


def group_mixing(edges: pd.DataFrame, machines) -> dict:
    """Share of edges (and weight) inside one SMD group vs the share of machine pairs that are inside one group."""
    pairs = all_pairs(machines)
    base = float(np.mean([smd_group(a) == smd_group(b) for a, b in pairs])) if pairs else float("nan")
    if edges.empty:
        return {"edges": 0, "within_edges": 0, "within_edge_share": None, "within_weight_share": None,
                "within_pair_share": base}
    within = np.array([smd_group(a) == smd_group(b) for a, b in zip(edges["src"], edges["dst"])])
    return {"edges": int(len(edges)), "within_edges": int(within.sum()),
            "within_edge_share": float(within.mean()),
            "within_weight_share": float(edges["weight"][within].sum() / edges["weight"].sum()),
            "within_pair_share": base}


def pagerank_reference(vert: pd.DataFrame, edges: pd.DataFrame, d: float = DAMPING, iters: int = PR_ITERS) -> dict:
    """Plain numpy power iteration of the same formula (for tests and as a third opinion)."""
    ids = list(vert["id"])
    ix = {m: k for k, m in enumerate(ids)}
    w = np.zeros((len(ids), len(ids)))
    for r in edges.itertuples():
        w[ix[r.src], ix[r.dst]] += r.weight
        w[ix[r.dst], ix[r.src]] += r.weight
    deg = w.sum(axis=1)
    share = np.divide(w, deg[:, None], out=np.zeros_like(w), where=deg[:, None] > 0)
    rank = np.full(len(ids), PR_INIT)
    for _ in range(iters):
        rank = (1 - d) + d * share.T @ rank
    return dict(zip(ids, rank.tolist()))


# ---- Spark / GraphFrames ------------------------------------------------------------------------------------------
def under_spark_submit() -> bool:
    return "PYSPARK_GATEWAY_PORT" in os.environ


def build_spark(master: str | None = None, ckpt: Path = CKPT_DIR, app: str = "e8_graph"):
    from pyspark.sql import SparkSession
    b = (SparkSession.builder.appName(app)
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
         .config("spark.sql.shuffle.partitions", "4")
         # pandas -> Spark through Arrow: the rows go to the JVM directly, no Python worker (see _local_df)
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false"))
    if master:
        b = b.master(master)
    elif not under_spark_submit():
        b = b.master("local[*]")
    if not under_spark_submit():
        # Locally the JVM starts here, so the package can still be requested. slf4j-api 1.7 (a GraphFrames
        # dependency) is excluded: Spark 3.5 ships slf4j 2 and the old one only adds warnings.
        b = (b.config("spark.jars.packages", GRAPHFRAMES_PACKAGE)
             .config("spark.jars.repositories", SPARK_PACKAGES_REPO)
             .config("spark.jars.excludes", "org.slf4j:slf4j-api"))
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    ckpt_uri = str(ckpt) if "://" in str(ckpt) else Path(ckpt).as_posix()
    spark.sparkContext.setCheckpointDir(ckpt_uri)
    return spark


def _local_df(spark, pdf: pd.DataFrame):
    """A Spark DataFrame from a small pandas frame that never needs a Python worker: created through Arrow (JVM-side
    rows) and cut from its lineage with localCheckpoint. Built from Python rows instead, every Pregel iteration
    recomputed it through a Python worker; Windows cannot fork workers, so one iteration on the 28-vertex graph then
    took ~36 s (found while running E8)."""
    return spark.createDataFrame(pdf.reset_index(drop=True)).localCheckpoint(eager=True)


def _jvm_df(df):
    """localCheckpoint for a DataFrame derived from _local_df ones (shortens the plan Pregel starts from)."""
    return df.localCheckpoint(eager=True)


def spark_edges(spark, inc: pd.DataFrame, gap: float) -> pd.DataFrame:
    """The same edges as edge_weights, as a Spark self-join (what runs on the cluster)."""
    from pyspark.sql import functions as F
    df = _local_df(spark, inc[["entity_id", "start", "end"]].astype({"start": "int64", "end": "int64"}))
    a, b = df.alias("a"), df.alias("b")
    g = F.greatest(F.col("a.start"), F.col("b.start")) - F.least(F.col("a.end"), F.col("b.end"))
    cond = (F.col("a.entity_id") < F.col("b.entity_id")) & ((g <= gap) if gap > 0 else (g < 0))
    out = (a.join(b, cond).groupBy(F.col("a.entity_id").alias("src"), F.col("b.entity_id").alias("dst"))
           .agg(F.count(F.lit(1)).alias("weight")).toPandas())
    out["weight"] = out["weight"].astype(np.int64)
    return out.sort_values(["src", "dst"]).reset_index(drop=True)


def _graphframe(spark, vert: pd.DataFrame, edges: pd.DataFrame, both_directions: bool = False):
    from graphframes import GraphFrame
    v = _local_df(spark, vert[["id", "smd_group"]].astype(str))
    e = edges[["src", "dst", "weight"]].astype({"src": str, "dst": str, "weight": "int64"})
    if both_directions:
        e = pd.concat([e, e.rename(columns={"src": "dst", "dst": "src"})[["src", "dst", "weight"]]],
                      ignore_index=True)
    if e.empty:  # Arrow cannot infer types of an empty frame
        from pyspark.sql.types import LongType, StringType, StructField, StructType
        schema = StructType([StructField("src", StringType()), StructField("dst", StringType()),
                             StructField("weight", LongType())])
        return GraphFrame(v, spark.createDataFrame([], schema))
    return GraphFrame(v, _local_df(spark, e))


def pregel_pagerank(spark, vert: pd.DataFrame, edges: pd.DataFrame, d: float = DAMPING,
                    iters: int = PR_ITERS) -> dict:
    """Weighted PageRank with graphframes.lib.Pregel (GDS formula, see module docstring)."""
    from graphframes import GraphFrame
    from graphframes.lib import Pregel
    from pyspark.sql import functions as F
    g = _graphframe(spark, vert, edges, both_directions=True)
    wdeg = g.edges.groupBy(F.col("src").alias("id")).agg(F.sum("weight").cast("double").alias("wdeg"))
    v = _jvm_df(g.vertices.join(wdeg, "id", "left").fillna(0.0, subset=["wdeg"]))
    g = GraphFrame(v, g.edges)
    ranks = (Pregel(g).setMaxIter(iters).setCheckpointInterval(PREGEL_CKPT_EVERY)
             .withVertexColumn("rank", F.lit(PR_INIT),
                               F.lit(1.0 - d) + F.lit(d) * F.coalesce(Pregel.msg(), F.lit(0.0)))
             .sendMsgToDst(Pregel.src("rank") * Pregel.edge("weight") / Pregel.src("wdeg"))
             .aggMsgs(F.sum(Pregel.msg()))
             .run())
    return {r["id"]: float(r["rank"]) for r in ranks.select("id", "rank").collect()}


def run_graph(spark, vert: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """GraphFrames connectedComponents + labelPropagation + Pregel PageRank on one graph; with timings."""
    out, t = {}, time.perf_counter()
    g = _graphframe(spark, vert, edges)
    cc = g.connectedComponents()
    out["components"] = {r["id"]: int(r["component"]) for r in cc.select("id", "component").collect()}
    out["cc_s"] = time.perf_counter() - t
    t = time.perf_counter()
    lpa = g.labelPropagation(maxIter=LPA_ITERS)
    out["lpa"] = {r["id"]: int(r["label"]) for r in lpa.select("id", "label").collect()}
    out["lpa_s"] = time.perf_counter() - t
    t = time.perf_counter()
    out["pagerank"] = pregel_pagerank(spark, vert, edges)
    out["pagerank_s"] = time.perf_counter() - t
    return out


# ---- step 1: pregel -----------------------------------------------------------------------------------------------
def write_graph_files(out_dir: Path, vert: pd.DataFrame, graphs: dict[str, pd.DataFrame]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vert.to_csv(out_dir / "vertices.csv", index=False)
    for name, e in graphs.items():
        e.to_csv(out_dir / f"edges_{name}.csv", index=False)


def build_graphs(inc: pd.DataFrame, spans: dict, n_shuffles: int = N_SHUFFLES, seed: int = SEED):
    """(vertices, {graph: edges}, null info). gap10 edges carry null_mean / p / sig columns."""
    vert = vertices(inc["entity_id"])
    pairs = all_pairs(vert["id"])
    obs = {GAP_MAIN: edge_weights(inc, GAP_MAIN), GAP_SENS: edge_weights(inc, GAP_SENS)}
    null_w = null_model(inc, spans, (GAP_MAIN, GAP_SENS), n_shuffles, seed)
    sig10 = significant_edges(obs[GAP_MAIN], null_w[GAP_MAIN], pairs)
    sig0 = significant_edges(obs[GAP_SENS], null_w[GAP_SENS], pairs)
    graphs = {"gap10": sig10, "gap0": sig0,
              "sig": sig10[sig10["sig"]].reset_index(drop=True)}
    null = {"n_shuffles": n_shuffles, "seed": seed, "p_threshold": 1 / (1 + n_shuffles),
            "gap10": {**null_summary(obs[GAP_MAIN], null_w[GAP_MAIN]),
                      "sig_edges": int(sig10["sig"].sum()), "chance_sig_edges": chance_sig_edges(null_w[GAP_MAIN])},
            "gap0": {**null_summary(obs[GAP_SENS], null_w[GAP_SENS]),
                     "sig_edges": int(sig0["sig"].sum()), "chance_sig_edges": chance_sig_edges(null_w[GAP_SENS])}}
    return vert, graphs, null


def cmd_pregel(args) -> int:
    t_all = time.perf_counter()
    timings = {}
    t = time.perf_counter()
    inc = load_incidents(args.incidents)
    spans = load_spans(args.scores, inc)
    vert, graphs, null = build_graphs(inc, spans, args.shuffles, args.seed)
    timings["graphs_and_null_s"] = time.perf_counter() - t
    print(f"{len(inc)} incidents on {len(vert)} machines; edges gap10={len(graphs['gap10'])} "
          f"gap0={len(graphs['gap0'])} sig={len(graphs['sig'])}; chance edges gap10="
          f"{null['gap10']['chance_edges_mean']:.1f}")
    write_graph_files(args.out_dir, vert, graphs)

    t = time.perf_counter()
    spark = build_spark(args.master, args.checkpoint_dir)
    timings["spark_start_s"] = time.perf_counter() - t
    try:
        sc = spark.sparkContext
        env = {"master": sc.master, "default_parallelism": sc.defaultParallelism,
               "spark": spark.version, "graphframes": GRAPHFRAMES_PACKAGE}
        t = time.perf_counter()
        spark_check = {}
        for name, gap in (("gap10", GAP_MAIN), ("gap0", GAP_SENS)):
            se = spark_edges(spark, inc, gap)
            same = se.equals(graphs[name][["src", "dst", "weight"]].reset_index(drop=True))
            spark_check[name] = same
            if not same:
                raise RuntimeError(f"Spark self-join edges differ from the pandas edges for {name}")
        timings["spark_edges_s"] = time.perf_counter() - t
        results = {}
        for name in GRAPHS:
            t = time.perf_counter()
            results[name] = run_graph(spark, vert, graphs[name][["src", "dst", "weight"]])
            results[name]["pagerank_numpy_max_abs_diff"] = max(
                abs(results[name]["pagerank"][m] - v)
                for m, v in pagerank_reference(vert, graphs[name]).items())
            timings[f"{name}_s"] = time.perf_counter() - t
            print(f"  {name}: {len(groups_of(results[name]['components']))} components, "
                  f"{len(groups_of(results[name]['lpa']))} LPA labels, top PageRank "
                  f"{top_k(results[name]['pagerank'], 3)} ({timings[f'{name}_s']:.1f} s)")
    finally:
        spark.stop()
    timings["total_s"] = time.perf_counter() - t_all
    out = {"incidents": Path(args.incidents).as_posix(), "n_incidents": int(len(inc)),
           "n_machines": int(len(vert)), "env": env, "params": {"gap_main": GAP_MAIN, "gap_sens": GAP_SENS,
           "damping": DAMPING, "pr_iters": PR_ITERS, "lpa_iters": LPA_ITERS},
           "spark_edges_equal_pandas": spark_check, "null": null, "results": results, "timings": timings}
    path = args.out_dir / args.out_name
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path}; "
          f"Pregel part runtime {timings['total_s']:.1f} s")
    return 0


# ---- step 3: report -----------------------------------------------------------------------------------------------
def compare(pregel: dict, neo: dict) -> dict:
    """Pregel vs Neo4j per graph, and both vs the SMD machine groups."""
    out = {}
    for name in GRAPHS:
        p, n = pregel["results"][name], neo["results"][name]
        smd = {m: smd_group(m) for m in p["components"]}
        pr = rank_compare(p["pagerank"], n["pagerank"])
        out[name] = {
            "components_same": same_partition(p["components"], n["wcc"]),
            "n_components_pregel": len(groups_of(p["components"])), "n_components_neo4j": len(groups_of(n["wcc"])),
            "pagerank": pr,
            "ari_lpa_gf_vs_louvain": ari(p["lpa"], n["louvain"]),
            "ari_lpa_gf_vs_lpa_gds": ari(p["lpa"], n["lpa"]),
            "ari_vs_smd_group": {"components": ari(p["components"], smd), "lpa_graphframes": ari(p["lpa"], smd),
                                 "louvain": ari(n["louvain"], smd), "lpa_gds": ari(n["lpa"], smd)},
            "louvain_modularity": n.get("louvain_modularity"),
        }
    return out


def _fmt_groups(labels: dict, max_groups: int = 8) -> str:
    gs = [g for g in groups_of(labels)]
    multi = [g for g in gs if len(g) > 1]
    single = sum(1 for g in gs if len(g) == 1)
    txt = "; ".join("{" + ", ".join(m.replace("machine-", "") for m in g) + "}" for g in multi[:max_groups])
    if len(multi) > max_groups:
        txt += f"; ... ({len(multi) - max_groups} more)"
    return (txt or "none") + (f" + {single} singleton(s)" if single else "")


def plot(path: Path, vert: pd.DataFrame, edges10: pd.DataFrame, neo: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    colors = {"1": "#2a78c9", "2": "#d9822b", "3": "#3a9a5b"}
    g = nx.Graph()
    g.add_nodes_from(vert["id"])
    for r in edges10.itertuples():
        g.add_edge(r.src, r.dst, weight=r.weight, sig=bool(r.sig))
    pos = nx.spring_layout(g, weight="weight", seed=SEED, k=0.9)

    def sizes(gg, graph_name):
        pr = neo["results"][graph_name]["pagerank"]
        lo, hi = min(pr[m] for m in gg.nodes), max(pr[m] for m in gg.nodes)
        return [200 + 900 * (pr[m] - lo) / (hi - lo + 1e-12) for m in gg.nodes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=130)
    # Right panel: only machines with a beyond-chance edge, with their own layout (shared positions made long edges
    # pass behind unrelated nodes) and the PageRank of the sig graph.
    sig_g = g.edge_subgraph([(u, v) for u, v, a in g.edges(data=True) if a["sig"]]).copy()
    pos_sig = {}
    comps = sorted(nx.connected_components(sig_g), key=lambda c: (-len(c), sorted(c)))
    for k, comp in enumerate(comps):  # largest component on the left, the others stacked in a column on the right
        if k == 0:
            pos_sig.update(nx.spring_layout(sig_g.subgraph(comp), weight=None, seed=SEED,
                                            k=2.0 / max(len(comp), 1) ** 0.5))
            continue
        y = 0.9 - 1.8 * (k - 1) / max(len(comps) - 2, 1)
        for j, m in enumerate(sorted(comp)):  # small components: members side by side, one row each
            pos_sig[m] = np.array([1.7 + 0.6 * j, y])
    n_alone = g.number_of_nodes() - sig_g.number_of_nodes()
    for ax, gg, pp, gname, title in (
            (axes[0], g, pos, "gap10", f"gap10 co-failure graph ({g.number_of_edges()} edges)"),
            (axes[1], sig_g, pos_sig, "sig", f"beyond-chance edges only ({sig_g.number_of_edges()} edges; "
                                             f"{n_alone} machines without one not shown)")):
        if not gg.number_of_nodes():
            ax.axis("off")
            continue
        ws = [a["weight"] for _, _, a in gg.edges(data=True)]
        wmax = max(ws) if ws else 1
        nx.draw_networkx_edges(gg, pp, ax=ax, width=[0.4 + 3.6 * w / wmax for w in ws],
                               edge_color=["#444444" if a["sig"] else "#bbbbbb" for _, _, a in gg.edges(data=True)],
                               alpha=0.8)
        if gname == "sig":
            nx.draw_networkx_edge_labels(gg, pp, ax=ax, font_size=7, rotate=False,
                                         edge_labels={(u, v): a["weight"] for u, v, a in gg.edges(data=True)})
        nx.draw_networkx_nodes(gg, pp, ax=ax, node_size=sizes(gg, gname), linewidths=0.8, edgecolors="white",
                               node_color=[colors.get(smd_group(m), "#888888") for m in gg.nodes])
        nx.draw_networkx_labels(gg, pp, ax=ax, labels={m: m.replace("machine-", "") for m in gg.nodes},
                                font_size=7)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9, label=f"SMD group {k}")
               for k, c in colors.items()]
    handles += [plt.Line2D([], [], color="#444444", lw=2, label="beyond chance (p = 1/101)"),
                plt.Line2D([], [], color="#bbbbbb", lw=2, label="not beyond chance")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("E8: SMD machines whose incidents co-occur (10-min gap); edge width = overlapping incident pairs, "
                 "node size = PageRank of that graph", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def summary_lines(pregel: dict, neo: dict, cmp: dict, sig_edges: pd.DataFrame) -> list[str]:
    """The headline findings, worded from the numbers (so a rerun on other incidents stays truthful)."""
    n10 = pregel["null"]["gap10"]
    comps_same = all(cmp[g]["components_same"] for g in GRAPHS)
    top_same = all(cmp[g]["pagerank"]["top_same_order"] for g in GRAPHS)
    max_diff = max(cmp[g]["pagerank"]["max_abs_diff"] for g in GRAPHS)
    sig_groups = [g for g in groups_of(pregel["results"]["sig"]["components"]) if len(g) > 1]
    largest = sig_groups[0] if sig_groups else []
    mix = cmp["sig"]["mixing"]
    ari_max = max(max(cmp[g]["ari_vs_smd_group"].values()) for g in GRAPHS)
    short = lambda ms: ", ".join(m.replace("machine-", "") for m in ms)  # noqa: E731
    excess = n10["observed_weight"] / n10["chance_weight_mean"] if n10["chance_weight_mean"] else float("nan")
    L = ["## Summary", "",
         f"- **Machines do fail together more than chance.** With a 10-minute gap the graph has "
         f"{n10['observed_edges']} edges and {n10['observed_weight']} co-occurring incident pairs; shuffling each "
         f"machine's incident times gives {n10['chance_edges_mean']:.0f} ± {n10['chance_edges_sd']:.0f} edges and "
         f"{n10['chance_weight_mean']:.0f} ± {n10['chance_weight_sd']:.0f} pairs (observed = {excess:.2f} x chance, "
         f"above all 100 shuffles). But most of the {n10['observed_edges']} edges are what chance alone would give "
         f"between 28 machines; only {n10['sig_edges']} pairs are individually beyond chance, against "
         f"{n10['chance_sig_edges']:.1f} expected by luck.",
         f"- **Groups of machines that fail together** (components of the beyond-chance graph): "
         + ("; ".join("{" + short(g) + "}" for g in sig_groups) if sig_groups else "none")
         + (f". The largest has {len(largest)} machines and spans SMD groups "
            f"{', '.join(sorted({smd_group(m) for m in largest}))}." if largest else "."),
         f"- **Pregel and Neo4j agree.** Connected components = GDS WCC on every graph: "
         f"{'yes' if comps_same else 'NO'}. Weighted PageRank, same formula: top-5 identical in order "
         f"{'on every graph' if top_same else 'NOT on every graph'}, largest difference {max_diff:.1e}. "
         "Communities differ by algorithm more than by system: unweighted LPA finds "
         f"{len(groups_of(pregel['results']['gap10']['lpa']))} label(s) on gap10 (GraphFrames) vs "
         f"{len(groups_of(neo['results']['gap10']['lpa']))} (GDS), weighted Louvain "
         f"{len(groups_of(neo['results']['gap10']['louvain']))}; on the beyond-chance graph Louvain "
         + ("returns exactly its connected components." if same_partition(neo["results"]["sig"]["louvain"],
                                                                          pregel["results"]["sig"]["components"])
            else "splits some connected components."),
         f"- **Co-failure groups do not follow the SMD machine groups.** Best ARI against the 1-x/2-x/3-x label "
         f"is {ari_max:.2f} (0 = chance); {mix['within_edges']} of {mix['edges']} beyond-chance edges join "
         f"machines of the same SMD group (pair baseline {mix['within_pair_share']:.0%}), a small excess at best."]
    return L


def render_md(pregel: dict, neo: dict, cmp: dict, sig_edges: pd.DataFrame, cluster_cmd: str) -> str:
    n10, n0 = pregel["null"]["gap10"], pregel["null"]["gap0"]
    P, N = pregel["results"], neo["results"]
    L = ["# E8: graph processing on incidents -- which SMD machines fail together?", "",
         "Generated by `python experiments/e8_graph.py report` from `data/e8/pregel.json` "
         "(`experiments/e8_graph.py pregel`, Spark GraphFrames) and `data/e8/neo4j.json` "
         "(`experiments/e8_neo4j.py`, Neo4j + GDS). Pipeline defaults are unchanged.", "",
         *summary_lines(pregel, neo, cmp, sig_edges), "",
         "## Input and graph", "",
         f"- Incidents: `{pregel['incidents']}`, {pregel['n_incidents']} incidents on {pregel['n_machines']} "
         "machines (incidents layer, `src/incidents/group.py`).",
         *([f"- Built for E8 because `data/incidents/source=smd` on this laptop held only the fake fixture: the AE "
            "batch test scores (`data/scores_ae/source=smd`, 28 machines, is_anomaly = ae_flag) were copied to a "
            "staging root and grouped there, leaving the pipeline layers untouched: "
            "`mkdir -p data/e8/root/data/scores/source=smd && cp data/scores_ae/source=smd/scores.parquet "
            "data/e8/root/data/scores/source=smd/ && python src/incidents/group.py --source smd --root data/e8/root`."]
           if "data/e8/root" in pregel["incidents"] else []),
         "- One vertex per machine. Edge when two incidents of different machines co-occur: the later one starts at "
         "most G minutes after the earlier one ends. Weight = number of co-occurring incident pairs.",
         "- `gap10`: G = 10 min (main; equals the incident session gap). `gap0`: intervals must truly overlap "
         "(sensitivity). `sig`: `gap10` edges whose weight is above all 100 shuffles (below).", "",
         "| graph | edges | total weight | components (Pregel / GDS) |", "| --- | ---: | ---: | --- |"]
    for name in GRAPHS:
        e = sig_edges if name == "sig" else None
        n_e = len(e) if e is not None else (n10["observed_edges"] if name == "gap10" else n0["observed_edges"])
        w = int(e["weight"].sum()) if e is not None else (n10["observed_weight"] if name == "gap10"
                                                            else n0["observed_weight"])
        L.append(f"| {name} | {n_e} | {w} | {cmp[name]['n_components_pregel']} / {cmp[name]['n_components_neo4j']} |")
    L += ["", "## Null check: edges expected by chance", "",
          "Each machine's incident times were shuffled 100 times (seed 8): its incident durations and idle gaps are "
          "permuted inside its own time span, so the number of incidents, their lengths and the span are kept and "
          "only the alignment with other machines is destroyed.", "",
          "| gap | observed edges | chance edges, mean ± sd (95th pct) | observed weight | chance weight, mean ± sd "
          "| p (edges / weight) | edges beyond chance | expected beyond chance by luck |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lab, n in (("10 min", n10), ("0 (overlap)", n0)):
        L.append(f"| {lab} | {n['observed_edges']} | {n['chance_edges_mean']:.1f} ± {n['chance_edges_sd']:.1f} "
                 f"({n['chance_edges_p95']:.0f}) | {n['observed_weight']} | {n['chance_weight_mean']:.1f} ± "
                 f"{n['chance_weight_sd']:.1f} | {n['p_edges']:.3f} / {n['p_weight']:.3f} | {n['sig_edges']} | "
                 f"{n['chance_sig_edges']:.1f} |")
    L += ["", "## Pregel (GraphFrames) vs Neo4j (GDS)", "",
          "| graph | same components? | PageRank top 5, Pregel | PageRank top 5, GDS | same top-5 set / order | "
          "Spearman | max abs diff | ARI LPA(GF) vs Louvain | ARI LPA(GF) vs LPA(GDS) |",
          "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    short = lambda ms: ", ".join(m.replace("machine-", "") for m in ms)  # noqa: E731
    for name in GRAPHS:
        c = cmp[name]
        pr = c["pagerank"]
        L.append(f"| {name} | {'yes' if c['components_same'] else 'NO'} | {short(pr['top_a'])} | "
                 f"{short(pr['top_b'])} | {'yes' if pr['top_same_set'] else 'no'} / "
                 f"{'yes' if pr['top_same_order'] else 'no'} | {pr['spearman']:.3f} | {pr['max_abs_diff']:.2e} | "
                 f"{c['ari_lpa_gf_vs_louvain']:.2f} | {c['ari_lpa_gf_vs_lpa_gds']:.2f} |")
    L += ["", "Groups found (machine ids without `machine-`):", ""]
    for name in GRAPHS:
        L += [f"- **{name}** -- connected components: {_fmt_groups(P[name]['components'])}",
              f"  - LPA (GraphFrames): {_fmt_groups(P[name]['lpa'])}",
              f"  - Louvain (GDS, modularity {cmp[name]['louvain_modularity']:.3f}): "
              f"{_fmt_groups(N[name]['louvain'])}" if cmp[name]["louvain_modularity"] is not None else
              f"  - Louvain (GDS): {_fmt_groups(N[name]['louvain'])}"]
    L += ["", "## Do the groups follow the SMD machine groups (1-x, 2-x, 3-x)?", "",
          "ARI = adjusted Rand index against the SMD group label (1 = identical partition, 0 = chance). "
          "Within-group share: fraction of edges joining two machines of the same SMD group, against the fraction "
          "of all machine pairs that are within one group.", "",
          "| graph | ARI components | ARI LPA (GF) | ARI Louvain | ARI LPA (GDS) | within-group edges | "
          "within-group weight | within-group pairs (baseline) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in GRAPHS:
        a, mix = cmp[name]["ari_vs_smd_group"], cmp[name]["mixing"]
        fmt = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
        L.append(f"| {name} | {a['components']:.2f} | {a['lpa_graphframes']:.2f} | {a['louvain']:.2f} | "
                 f"{a['lpa_gds']:.2f} | {mix['within_edges']}/{mix['edges']} ({fmt(mix['within_edge_share'])}) | "
                 f"{fmt(mix['within_weight_share'])} | {fmt(mix['within_pair_share'])} |")
    L += ["", "## Operator queries (Cypher, Neo4j)", ""]
    for q in neo["queries"]:
        L += [f"**{q['title']}**", "", "```cypher", q["cypher"].strip(), "```", ""]
        rows = q["rows"]
        if rows:
            cols = list(rows[0])
            L += ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
            for r in rows:
                L.append("| " + " | ".join(", ".join(map(str, v)) if isinstance(v, list) else
                                           (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r.values()) + " |")
        else:
            L.append("(no rows)")
        L.append("")
    L += ["![E8 co-failure graph](e8_graph.png)", "", "## Runtime", "",
          f"- Pregel part ({pregel['env']['master']}, parallelism {pregel['env']['default_parallelism']}, Spark "
          f"{pregel['env']['spark']}, `{pregel['env']['graphframes']}`): total "
          f"{pregel['timings']['total_s']:.1f} s; graphs + 100-shuffle null model "
          f"{pregel['timings']['graphs_and_null_s']:.1f} s; Spark start {pregel['timings']['spark_start_s']:.1f} s; "
          + "; ".join(f"{g} {pregel['timings'][g + '_s']:.1f} s" for g in GRAPHS) + ".",
          f"- Neo4j part (Neo4j {neo['env'].get('neo4j')}, GDS {neo['env'].get('gds')}): total "
          f"{neo['timings']['total_s']:.1f} s (load {neo['timings']['load_s']:.1f} s, algorithms "
          f"{neo['timings']['algorithms_s']:.1f} s, queries {neo['timings']['queries_s']:.2f} s).",
          f"- Spark self-join edges equal the pandas edges: {pregel['spark_edges_equal_pandas']}.",
          "", "Rerun the Pregel part on the Spark cluster (2 executors x 2 cores), on spark-master:", "",
          "```bash", cluster_cmd, "```", "",
          "## Caveats", "",
          "- SMD has no real timestamps: row i of a machine is 2026-01-01 + i minutes (train rows first). Two "
          "machines' incidents line up only through this synthetic index, so 'at the same time' means 'at the same "
          "row offset'. Machines with longer training data start their test period later (9 machines start on "
          "2026-01-20, the rest on 2026-01-17), which shifts their alignment. The null model keeps each machine's "
          "own span, so it controls for that, but a co-failure edge is still not proof of a shared physical cause.",
          "- Incidents are AE alarms on the test split (is_anomaly = ae_flag), not ground-truth labels; false alarms "
          "are part of the graph.",
          "- 378 machine pairs are tested; 'beyond chance' is an empirical p of 1/101 per pair with no further "
          "multiple-testing correction. The last column of the null table estimates how many such edges luck alone "
          "would produce.",
          "- LPA (GraphFrames and GDS) is unweighted and its tie-breaking differs between the two systems; Louvain is "
          "weighted. Communities are therefore compared by ARI, not for identity; components and PageRank are "
          "deterministic and are compared exactly."]
    return "\n".join(L) + "\n"


def cluster_command() -> str:
    return ("cd /srv/spark-shared/Anomaly-Detection && source .venv/bin/activate\n"
            "# only if data/incidents/source=smd is missing there: python src/incidents/group.py --source smd\n"
            "spark-submit --master spark://spark-master:7077 --executor-cores 2 --total-executor-cores 4 \\\n"
            f"  --packages {GRAPHFRAMES_PACKAGE} --repositories {SPARK_PACKAGES_REPO} \\\n"
            "  --exclude-packages org.slf4j:slf4j-api \\\n"
            "  experiments/e8_graph.py pregel --incidents data/incidents/source=smd \\\n"
            "  --scores data/scores/source=smd --out-name pregel_cluster.json")


def cmd_report(args) -> int:
    pregel = json.loads((args.out_dir / "pregel.json").read_text(encoding="utf-8"))
    neo = json.loads((args.out_dir / "neo4j.json").read_text(encoding="utf-8"))
    vert = pd.read_csv(args.out_dir / "vertices.csv", dtype=str)
    edges = {g: pd.read_csv(args.out_dir / f"edges_{g}.csv", dtype={"src": str, "dst": str}) for g in GRAPHS}
    cmp = compare(pregel, neo)
    for name in GRAPHS:
        cmp[name]["mixing"] = group_mixing(edges[name], vert["id"])
    md = render_md(pregel, neo, cmp, edges["sig"], cluster_command())
    args.md.write_text(md, encoding="utf-8")
    doc = {"pregel": pregel, "neo4j": neo, "compare": cmp,
           "sig_edges": edges["sig"].to_dict(orient="records"), "cluster_command": cluster_command()}
    args.json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    plot(args.png, vert, edges["gap10"], neo)
    for p in (args.md, args.json, args.png):
        print(f"wrote {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pregel", help="graphs, null model, GraphFrames CC / LPA / Pregel PageRank")
    p.add_argument("--incidents", type=Path, default=INCIDENTS_DIR)
    p.add_argument("--scores", type=Path, default=SCORES_DIR, help="scores the incidents came from (time spans)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--out-name", default="pregel.json")
    p.add_argument("--master", default=None, help="Spark master (default local[*], or spark-submit's)")
    p.add_argument("--checkpoint-dir", default=CKPT_DIR)
    p.add_argument("--shuffles", type=int, default=N_SHUFFLES)
    p.add_argument("--seed", type=int, default=SEED)
    r = sub.add_parser("report", help="docs/results/e8_graph.md + .json + .png")
    r.add_argument("--out-dir", type=Path, default=OUT_DIR)
    r.add_argument("--md", type=Path, default=DOC_MD)
    r.add_argument("--json", type=Path, default=DOC_JSON)
    r.add_argument("--png", type=Path, default=DOC_PNG)
    args = ap.parse_args(argv)
    return cmd_pregel(args) if args.cmd == "pregel" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
