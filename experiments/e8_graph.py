"""E8: graph processing on incidents -- which machines fail together? Pregel part (Spark GraphFrames) + report.

Two sources: SMD (method check: its timestamps are synthetic row offsets) and Alibaba v2018 (one shared clock; the
real co-failure analysis). Run from the repo root (venv active); this file does 1 and 3, experiments/e8_neo4j.py 2:
    python experiments/e8_alibaba_input.py                    # 0. Alibaba only: AE scores + incidents, hours 0-72
    python experiments/e8_graph.py pregel --source smd        # 1. graphs + null model + GraphFrames CC / LPA /
    python experiments/e8_graph.py pregel --source alibaba    #    Pregel PageRank
    python experiments/e8_neo4j.py --source smd               # 2. same graphs in Neo4j: GDS WCC / Louvain /
    python experiments/e8_neo4j.py --source alibaba           #    PageRank + Cypher
    python experiments/e8_graph.py report                     # 3. docs/results/e8_graph.md + .json + .png files
E8c (injected cascades, experiments/e8_cascade.py stream / pregel / neo4j) is added to the same report as its own
section, with docs/results/e8_graph_cascade.png, whenever data/e8/cascade_pregel.json exists.

Input (defaults per source in SOURCES; --incidents / --scores override): the incidents layer (columns of
LAYERS["incidents"]) and the scores it was built from. The scores give every machine's time span for the null model
and the vertex set (a machine without incidents is an isolated vertex); a machine missing there spans its own
first..last incident.

Graph: one vertex per machine (property group = SMD machine group 1, 2 or 3 from machine-<g>-<k>; '-' for
Alibaba). Two incidents a, b of
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

Outputs (suffix '' for smd, '_alibaba' for alibaba): data/e8/vertices<suffix>.csv, data/e8/edges<suffix>_<graph>.csv
(read by e8_neo4j.py: the same graph), data/e8/pregel<suffix>.json. The pipeline (scores, incidents, models) is not
changed.
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
WAVE_SIZES = (3, 5, 8)  # alarm waves are counted from these numbers of distinct machines
T0_MINUTES = int(pd.Timestamp(S.TIME_BASE_UTC).timestamp() // 60)  # 2026-01-01T00:00Z, start of both time axes
GRAPHFRAMES_PACKAGE = "graphframes:graphframes:0.8.3-spark3.5-s_2.12"
SPARK_PACKAGES_REPO = "https://repos.spark-packages.org"

OUT_DIR = ROOT / "data" / "e8"
ALIBABA_ROOT = Path("data") / "e8" / "alibaba_root"
# Per source: default inputs (relative to the repo root), output suffix, Q1 machine (None = the machine with the most
# beyond-chance partners), display name.
SOURCES = {
    "smd": {"incidents": Path(S.PATHS["incidents"]) / "source=smd", "scores": Path(S.PATHS["scores"]) / "source=smd",
            "suffix": "", "focus": "machine-1-1", "name": "SMD"},
    "alibaba": {"incidents": ALIBABA_ROOT / S.PATHS["incidents"] / "source=alibaba",
                "scores": ALIBABA_ROOT / S.PATHS["scores"] / "source=alibaba",
                "suffix": "_alibaba", "focus": None, "name": "Alibaba v2018"},
}
CKPT_DIR = ROOT / "checkpoints" / "e8_graph"
DOC_MD = ROOT / "docs" / "results" / "e8_graph.md"
DOC_JSON = ROOT / "docs" / "results" / "e8_graph.json"
DOC_PNG = ROOT / "docs" / "results" / "e8_graph.png"   # SMD; Alibaba gets e8_graph_alibaba.png


def out_file(out_dir: Path, base: str, source: str, ext: str) -> Path:
    """data/e8/<base><suffix>.<ext>, e.g. pregel.json (smd) or pregel_alibaba.json."""
    return Path(out_dir) / f"{base}{SOURCES[source]['suffix']}.{ext}"


def edges_file(out_dir: Path, source: str, graph: str) -> Path:
    return Path(out_dir) / f"edges{SOURCES[source]['suffix']}_{graph}.csv"


def short_id(m: str) -> str:
    return m.replace("machine-", "")


# ---- graph construction (pandas / numpy; Spark builds the same edges in spark_edges) ----------------------------
def smd_group(entity_id: str) -> str:
    """'machine-2-7' -> '2' (the SMD machine group); '-' for anything else (Alibaba 'm_12')."""
    parts = entity_id.split("-")
    return parts[1] if len(parts) == 3 and parts[0] == "machine" else "-"


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
    """Per machine (first window_start, last window_end) in minutes, for every machine in the scores and in the
    incidents; a machine without scores spans its own first..last incident."""
    spans = {m: (int(g["start"].min()), int(g["end"].max())) for m, g in inc.groupby("entity_id")}
    if scores_path is not None and Path(scores_path).exists():
        sc = pd.read_parquet(scores_path, columns=["entity_id", "window_start", "window_end"])
        for m, g in sc.groupby("entity_id"):
            lo, hi = int(_minutes(g["window_start"]).min()), int(_minutes(g["window_end"]).max())
            old = spans.get(m, (lo, hi))
            spans[m] = (min(lo, old[0]), max(hi, old[1]))
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
    return pd.DataFrame({"id": ms, "group": [smd_group(m) for m in ms]})


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
               seed: int = SEED, pairs: list[tuple[str, str]] | None = None) -> dict:
    """{gap: (n, n_pairs) array of pair weights over the shuffles}; one shuffle feeds every gap. pairs defaults to
    all pairs of machines with incidents."""
    rng = np.random.default_rng(seed)
    pairs = all_pairs(inc["entity_id"]) if pairs is None else pairs
    out = {g: np.zeros((n, len(pairs)), dtype=np.int64) for g in gaps}
    # per shuffle: number of alarm waves with >= k machines for k in WAVE_SIZES, then the largest wave
    out["waves"] = np.zeros((n, len(WAVE_SIZES) + 1), dtype=np.int64)
    for k in range(n):
        sh = shuffle_incidents(inc, spans, rng)
        for g in gaps:
            out[g][k] = pair_vector(edge_weights(sh, g), pairs)
        w = alarm_waves(sh, GAP_MAIN)
        out["waves"][k] = [int((w["machines"] >= size).sum()) for size in WAVE_SIZES] +             [int(w["machines"].max()) if len(w) else 0]
    return out


def alarm_waves(inc: pd.DataFrame, gap: float = GAP_MAIN) -> pd.DataFrame:
    """Incident-level view: incidents of any machines chained in time order, a new wave starting when an incident
    starts more than `gap` minutes after every earlier incident ended (gap 0: when it does not overlap them).
    Columns start, end (minutes), incidents, machines (distinct), members (sorted machine ids); time order."""
    cols = ["start", "end", "incidents", "machines", "members"]
    if inc.empty:
        return pd.DataFrame(columns=cols)
    d = inc.sort_values(["start", "end", "entity_id"]).reset_index(drop=True)
    wave, cur, end = [], -1, None
    for s, e in zip(d["start"].to_numpy(), d["end"].to_numpy()):
        joins = end is not None and ((s - end <= gap) if gap > 0 else (s - end < 0))  # same rule as co_occur
        if not joins:
            cur += 1
            end = e
        else:
            end = max(end, e)
        wave.append(cur)
    d["wave"] = wave
    w = d.groupby("wave").agg(start=("start", "min"), end=("end", "max"), incidents=("entity_id", "size"),
                              machines=("entity_id", "nunique"),
                              members=("entity_id", lambda x: sorted(set(x)))).reset_index(drop=True)
    return w[cols]


def wave_summary(inc: pd.DataFrame, null_waves: np.ndarray, t0_minutes: int, top: int = 8) -> dict:
    """Observed alarm waves vs the shuffles: per size threshold k in WAVE_SIZES, the number of waves with >= k
    machines (and the incidents in them), and the largest wave; empirical p = (1 + #shuffles >= observed) / (1 + n)."""
    w = alarm_waves(inc, GAP_MAIN)
    n = null_waves.shape[0]
    p_ge = lambda col, obs: float((1 + (col >= obs).sum()) / (1 + n))  # noqa: E731
    by_size = []
    for j, k in enumerate(WAVE_SIZES):
        obs = int((w["machines"] >= k).sum())
        col = null_waves[:, j]
        by_size.append({"min_machines": k, "observed": obs,
                        "incidents": int(w.loc[w["machines"] >= k, "incidents"].sum()),
                        "chance_mean": float(col.mean()), "chance_sd": float(col.std(ddof=1)), "p": p_ge(col, obs)})
    largest, nl = (int(w["machines"].max()) if len(w) else 0), null_waves[:, -1]
    big = w[w["machines"] >= WAVE_SIZES[0]].sort_values(["machines", "start"], ascending=[False, True])
    return {"waves": int(len(w)), "multi_machine_waves": int((w["machines"] >= 2).sum()), "by_size": by_size,
            "largest": largest, "chance_largest_mean": float(nl.mean()), "chance_largest_max": int(nl.max()),
            "p_largest": p_ge(nl, largest),
            "top": [{"start_h": (int(r.start) - t0_minutes) / 60, "minutes": int(r.end - r.start),
                     "machines": int(r.machines), "incidents": int(r.incidents), "members": list(r.members)}
                    for r in big.head(top).itertuples()]}


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
    v = _local_df(spark, vert[["id", "group"]].astype(str))
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
def write_graph_files(out_dir: Path, vert: pd.DataFrame, graphs: dict[str, pd.DataFrame], source: str = "smd") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vert.to_csv(out_file(out_dir, "vertices", source, "csv"), index=False)
    for name, e in graphs.items():
        e.to_csv(edges_file(out_dir, source, name), index=False)


def read_graph_files(out_dir: Path, source: str = "smd") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    vert = pd.read_csv(out_file(out_dir, "vertices", source, "csv"), dtype=str, keep_default_na=False)
    edges = {g: pd.read_csv(edges_file(out_dir, source, g), dtype={"src": str, "dst": str}) for g in GRAPHS}
    return vert, edges


def build_graphs(inc: pd.DataFrame, spans: dict, n_shuffles: int = N_SHUFFLES, seed: int = SEED):
    """(vertices, {graph: edges}, null info). Vertices = machines with incidents or a span (scores). gap10 / gap0
    edges carry null_mean / p / sig columns."""
    vert = vertices(set(inc["entity_id"]) | set(spans))
    pairs = all_pairs(vert["id"])
    obs = {GAP_MAIN: edge_weights(inc, GAP_MAIN), GAP_SENS: edge_weights(inc, GAP_SENS)}
    null_w = null_model(inc, spans, (GAP_MAIN, GAP_SENS), n_shuffles, seed, pairs)
    sig10 = significant_edges(obs[GAP_MAIN], null_w[GAP_MAIN], pairs)
    sig0 = significant_edges(obs[GAP_SENS], null_w[GAP_SENS], pairs)
    graphs = {"gap10": sig10, "gap0": sig0,
              "sig": sig10[sig10["sig"]].reset_index(drop=True)}
    null = {"n_shuffles": n_shuffles, "seed": seed, "p_threshold": 1 / (1 + n_shuffles), "n_pairs": len(pairs),
            "gap10": {**null_summary(obs[GAP_MAIN], null_w[GAP_MAIN]),
                      "sig_edges": int(sig10["sig"].sum()), "chance_sig_edges": chance_sig_edges(null_w[GAP_MAIN])},
            "gap0": {**null_summary(obs[GAP_SENS], null_w[GAP_SENS]),
                     "sig_edges": int(sig0["sig"].sum()), "chance_sig_edges": chance_sig_edges(null_w[GAP_SENS])},
            "waves": wave_summary(inc, null_w["waves"], T0_MINUTES)}
    return vert, graphs, null


def rel_path(q) -> str:
    q = Path(q)
    full = q if q.is_absolute() else (Path.cwd() / q)
    full = full.resolve()
    return full.relative_to(ROOT).as_posix() if full.is_relative_to(ROOT) else q.as_posix()


def cmd_pregel(args) -> int:
    t_all = time.perf_counter()
    timings = {}
    cfg = SOURCES[args.source]
    args.incidents = args.incidents or ROOT / cfg["incidents"]
    args.scores = args.scores or ROOT / cfg["scores"]
    t = time.perf_counter()
    inc = load_incidents(args.incidents)
    spans = load_spans(args.scores, inc)
    vert, graphs, null = build_graphs(inc, spans, args.shuffles, args.seed)
    timings["graphs_and_null_s"] = time.perf_counter() - t
    print(f"{cfg['name']}: {len(inc)} incidents, {len(vert)} machines; edges gap10={len(graphs['gap10'])} "
          f"gap0={len(graphs['gap0'])} sig={len(graphs['sig'])}; chance edges gap10="
          f"{null['gap10']['chance_edges_mean']:.1f}")
    write_graph_files(args.out_dir, vert, graphs, args.source)

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
    out = {"source": args.source, "incidents": rel_path(args.incidents), "scores": rel_path(args.scores),
           "n_incidents": int(len(inc)), "n_machines": int(len(vert)),
           "machines_with_incidents": int(inc["entity_id"].nunique()),
           "span_minutes": {"min": int(min(b - a for a, b in spans.values())),
                            "max": int(max(b - a for a, b in spans.values()))},
           "incident_minutes": {"median": float((inc["end"] - inc["start"]).median()),
                                "mean": float((inc["end"] - inc["start"]).mean())},
           "env": env, "params": {"gap_main": GAP_MAIN, "gap_sens": GAP_SENS,
           "damping": DAMPING, "pr_iters": PR_ITERS, "lpa_iters": LPA_ITERS},
           "spark_edges_equal_pandas": spark_check, "null": null, "results": results, "timings": timings}
    path = Path(args.out_dir) / (args.out_name or out_file(args.out_dir, "pregel", args.source, "json").name)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path}; "
          f"Pregel part runtime {timings['total_s']:.1f} s")
    return 0


# ---- step 3: report -----------------------------------------------------------------------------------------------
def compare(pregel: dict, neo: dict, source: str = "smd") -> dict:
    """Pregel vs Neo4j per graph; for SMD also every partition vs the SMD machine groups."""
    out = {}
    for name in GRAPHS:
        p, n = pregel["results"][name], neo["results"][name]
        out[name] = {
            "components_same": same_partition(p["components"], n["wcc"]),
            "n_components_pregel": len(groups_of(p["components"])), "n_components_neo4j": len(groups_of(n["wcc"])),
            "pagerank": rank_compare(p["pagerank"], n["pagerank"]),
            "ari_lpa_gf_vs_louvain": ari(p["lpa"], n["louvain"]),
            "ari_lpa_gf_vs_lpa_gds": ari(p["lpa"], n["lpa"]),
            "louvain_equals_components": same_partition(n["louvain"], p["components"]),
            "louvain_modularity": n.get("louvain_modularity"),
        }
        if source == "smd":
            smd = {m: smd_group(m) for m in p["components"]}
            out[name]["ari_vs_smd_group"] = {"components": ari(p["components"], smd),
                                             "lpa_graphframes": ari(p["lpa"], smd),
                                             "louvain": ari(n["louvain"], smd), "lpa_gds": ari(n["lpa"], smd)}
    return out


def headline(pregel: dict, neo: dict, cmp: dict, edges: dict[str, pd.DataFrame]) -> dict:
    """The numbers the summary and the SMD-vs-Alibaba table use, for one source."""
    n10 = pregel["null"]["gap10"]
    groups = [g for g in groups_of(pregel["results"]["sig"]["components"]) if len(g) > 1]
    days = None
    if pregel.get("span_minutes"):
        days = pregel["span_minutes"]["max"] / 1440
    return {
        "machines": pregel["n_machines"], "machines_with_incidents": pregel.get("machines_with_incidents"),
        "incidents": pregel["n_incidents"], "days": days,
        "incidents_per_machine_day": (pregel["n_incidents"] / pregel["n_machines"] / days) if days else None,
        "median_incident_min": (pregel.get("incident_minutes") or {}).get("median"),
        "pairs": pregel["null"].get("n_pairs"),
        "edges": n10["observed_edges"], "chance_edges": n10["chance_edges_mean"], "chance_edges_sd": n10["chance_edges_sd"],
        "weight": n10["observed_weight"], "chance_weight": n10["chance_weight_mean"],
        "weight_ratio": n10["observed_weight"] / n10["chance_weight_mean"] if n10["chance_weight_mean"] else None,
        "p_weight": n10["p_weight"], "sig_edges": n10["sig_edges"], "chance_sig_edges": n10["chance_sig_edges"],
        "sig_share_of_edges": n10["sig_edges"] / n10["observed_edges"] if n10["observed_edges"] else None,
        "sig_share_of_pairs": n10["sig_edges"] / pregel["null"]["n_pairs"] if pregel["null"].get("n_pairs") else None,
        "groups": groups, "group_sizes": [len(g) for g in groups],
        "machines_in_groups": sum(len(g) for g in groups),
        "components_same_all": all(cmp[g]["components_same"] for g in GRAPHS),
        "pagerank_top_same_all": all(cmp[g]["pagerank"]["top_same_order"] for g in GRAPHS),
        "pagerank_max_diff": max(cmp[g]["pagerank"]["max_abs_diff"] for g in GRAPHS),
        "louvain_equals_components_sig": cmp["sig"]["louvain_equals_components"],
        "gap0_edges": pregel["null"]["gap0"]["observed_edges"],
        "gap0_chance_edges": pregel["null"]["gap0"]["chance_edges_mean"],
        "incidents_per_machine": pregel["n_incidents"] / max(pregel.get("machines_with_incidents") or 1, 1),
        "waves": pregel["null"].get("waves"),
    }


def _fmt_groups(labels: dict, max_groups: int = 8) -> str:
    gs = groups_of(labels)
    multi = [g for g in gs if len(g) > 1]
    single = sum(1 for g in gs if len(g) == 1)
    txt = "; ".join("{" + ", ".join(short_id(m) for m in g) + "}" for g in multi[:max_groups])
    if len(multi) > max_groups:
        txt += f"; ... ({len(multi) - max_groups} more)"
    return (txt or "none") + (f" + {single} singleton(s)" if single else "")


def _braces(groups) -> str:
    return "; ".join("{" + ", ".join(short_id(m) for m in g) + "}" for g in groups) or "none"


def plot(path: Path, vert: pd.DataFrame, edges10: pd.DataFrame, neo: dict, source: str = "smd",
         incidents: pd.DataFrame | None = None, waves: dict | None = None) -> None:
    """Left: the gap10 graph. Right: the beyond-chance edges, or (when incidents are given) an incident timeline
    per machine with the alarm waves of >= WAVE_SIZES[-1] machines marked -- used for Alibaba, whose pairwise
    beyond-chance graph is nearly empty while its waves carry the signal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    colors = {"1": "#2a78c9", "2": "#d9822b", "3": "#3a9a5b"}
    single = "#8fa6cf"
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
    big = [c for c in comps if len(c) > 2]
    small = [c for c in comps if len(c) <= 2]
    for k, comp in enumerate(big):  # components of 3+ side by side on the left, pairs in a column on the right
        sub = nx.spring_layout(sig_g.subgraph(comp), weight=None, seed=SEED, k=2.0 / max(len(comp), 1) ** 0.5)
        pos_sig.update({m: xy + np.array([2.4 * k, 0.0]) for m, xy in sub.items()})
    x0 = 2.4 * max(len(big) - 1, 0) + 1.7
    for k, comp in enumerate(small):
        y = 0.9 - 1.8 * k / max(len(small) - 1, 1)
        for j, m in enumerate(sorted(comp)):
            pos_sig[m] = np.array([x0 + 0.6 * j, y])
    n_alone = g.number_of_nodes() - sig_g.number_of_nodes()
    for ax, gg, pp, gname, title in (
            (axes[0], g, pos, "gap10", f"gap10 co-failure graph ({g.number_of_edges()} edges, "
                                       f"{g.number_of_nodes()} machines)"),
            (axes[1], sig_g, pos_sig, "sig", f"beyond-chance edges only ({sig_g.number_of_edges()} edges; "
                                             f"{n_alone} machines without one not shown)")):
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        if gname == "sig" and incidents is not None:
            _timeline(ax, incidents, waves)
            continue
        if not gg.number_of_nodes():
            ax.text(0.5, 0.5, "no beyond-chance edge", ha="center", va="center", transform=ax.transAxes)
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
                               node_color=[colors.get(smd_group(m), single) for m in gg.nodes])
        nx.draw_networkx_labels(gg, pp, ax=ax, labels={m: short_id(m) for m in gg.nodes}, font_size=7,
                                font_color="black")
    handles = ([plt.Line2D([], [], marker="o", ls="", color=c, markersize=9, label=f"SMD group {k}")
                for k, c in colors.items()] if source == "smd" else
               [plt.Line2D([], [], marker="o", ls="", color=single, markersize=9, label="Alibaba machine")])
    handles += [plt.Line2D([], [], color="#444444", lw=2, label="beyond chance (p = 1/101)"),
                plt.Line2D([], [], color="#bbbbbb", lw=2, label="not beyond chance")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    what = ("SMD machines (synthetic row-offset time)" if source == "smd"
            else "Alibaba machines, hours 0-72 (shared clock)")
    fig.suptitle(f"E8: {what} whose incidents co-occur (10-min gap); edge width = overlapping incident pairs, "
                 "node size = PageRank of that graph", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def _timeline(ax, incidents: pd.DataFrame, waves: dict | None) -> None:
    """Incidents as short bars, one row per machine (ordered by first incident), hours of the trace on x; the
    alarm waves with >= WAVE_SIZES[-1] machines shaded."""
    inc = incidents.copy()
    inc["h0"], inc["h1"] = (inc["start"] - T0_MINUTES) / 60, (inc["end"] - T0_MINUTES) / 60
    order = inc.groupby("entity_id")["h0"].min().sort_values().index.tolist()
    row = {m: k for k, m in enumerate(order)}
    big = [t for t in (waves or {}).get("top", []) if t["machines"] >= WAVE_SIZES[-1]]
    in_big = set()
    for t in big:
        ax.axvspan(t["start_h"] - 0.4, t["start_h"] + t["minutes"] / 60 + 0.4, color="#f2c14e", alpha=0.35, lw=0)
        ax.text(t["start_h"], len(order) + 0.3, f"{t['machines']} machines", fontsize=8, ha="center")
        in_big |= {(m, round(t["start_h"], 3)) for m in t["members"]}
    for r in inc.itertuples():
        hit = any(t["start_h"] - 1e-9 <= r.h0 <= t["start_h"] + t["minutes"] / 60 and r.entity_id in t["members"]
                  for t in big)
        ax.plot([r.h0, max(r.h1, r.h0 + 0.25)], [row[r.entity_id]] * 2, lw=4, solid_capstyle="butt",
                color="#c0392b" if hit else "#5b6b8c")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    ax.set_ylim(-1, len(order) + 1.2)
    ax.set_xlabel("hour of the trace (shared clock)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axis("on")
    ax.set_title(f"incidents of {len(order)} machines; shaded = waves of >= {WAVE_SIZES[-1]} machines (red)",
                 fontsize=10)


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _wave_row(waves: dict | None, size: int) -> dict | None:
    return next((x for x in waves["by_size"] if x["min_machines"] == size), None) if waves else None


def _wave_cell(waves: dict | None, size: int) -> str:
    r = _wave_row(waves, size)
    return "-" if r is None else f"{r['observed']} / {r['chance_mean']:.2f} (p = {r['p']:.3f})"


def comparison_lines(h: dict[str, dict]) -> list[str]:
    s, a = h.get("smd"), h.get("alibaba")
    cols = [k for k in ("smd", "alibaba") if k in h]
    head = "| | " + " | ".join(SOURCES[k]["name"] for k in cols) + " |"
    rows = [
        ("time base", {"smd": "synthetic: row i = T0 + i min, per machine", "alibaba": "real, one clock for all"}),
        ("machines (with incidents)", {k: f"{h[k]['machines']} ({h[k]['machines_with_incidents']})" for k in cols}),
        ("observed span", {k: f"{h[k]['days']:.1f} days" if h[k]["days"] else "-" for k in cols}),
        ("incidents (per machine-day)", {k: f"{h[k]['incidents']} ({h[k]['incidents_per_machine_day']:.2f})"
                                         if h[k]["incidents_per_machine_day"] else str(h[k]["incidents"])
                                         for k in cols}),
        ("machine pairs", {k: str(h[k]["pairs"]) for k in cols}),
        ("gap10 edges: observed / chance", {k: f"{h[k]['edges']} / {h[k]['chance_edges']:.1f} ± "
                                               f"{h[k]['chance_edges_sd']:.1f}" for k in cols}),
        ("co-occurring pairs: observed / chance", {k: f"{h[k]['weight']} / {h[k]['chance_weight']:.1f} "
                                                      f"(x{h[k]['weight_ratio']:.2f}, p = {h[k]['p_weight']:.3f})"
                                                      if h[k]["weight_ratio"] else "-" for k in cols}),
        ("edges beyond chance (expected by luck)", {k: f"{h[k]['sig_edges']} ({h[k]['chance_sig_edges']:.1f})"
                                                    for k in cols}),
        ("share of edges beyond chance", {k: _pct(h[k]["sig_share_of_edges"]) for k in cols}),
        ("share of machine pairs beyond chance", {k: f"{h[k]['sig_share_of_pairs']:.1%}" for k in cols}),
        ("co-failure groups (sizes)", {k: f"{len(h[k]['group_sizes'])} ({', '.join(map(str, h[k]['group_sizes']))})"
                                       if h[k]["group_sizes"] else "0" for k in cols}),
        ("machines in a group", {k: f"{h[k]['machines_in_groups']} of {h[k]['machines']}" for k in cols}),
        *[(f"alarm waves with >= {sz} machines: observed / chance (p)",
           {k: _wave_cell(h[k]["waves"], sz) for k in cols}) for sz in WAVE_SIZES],
        ("largest alarm wave: machines / chance mean (p)",
         {k: (f"{h[k]['waves']['largest']} / {h[k]['waves']['chance_largest_mean']:.1f} "
              f"(p = {h[k]['waves']['p_largest']:.3f})") if h[k]["waves"] else "-" for k in cols}),
        ("Pregel CC = GDS WCC (3 graphs)", {k: "yes" if h[k]["components_same_all"] else "NO" for k in cols}),
        ("PageRank top 5 same order (3 graphs)", {k: "yes" if h[k]["pagerank_top_same_all"] else "NO" for k in cols}),
        ("PageRank max abs difference", {k: f"{h[k]['pagerank_max_diff']:.1e}" for k in cols}),
        ("Louvain on sig = its components", {k: "yes" if h[k]["louvain_equals_components_sig"] else "no"
                                             for k in cols}),
    ]
    L = [head, "| --- |" + " --- |" * len(cols)]
    L += ["| " + label + " | " + " | ".join(vals.get(k, "-") for k in cols) + " |" for label, vals in rows]
    return L


def summary_lines(h: dict[str, dict]) -> list[str]:
    L = ["## Summary", ""]
    s, a = h.get("smd"), h.get("alibaba")
    if a:
        w = a["waves"] or {}
        big = [t for t in (w.get("top") or []) if t["machines"] >= WAVE_SIZES[-1]]
        r8 = _wave_row(w, WAVE_SIZES[-1])
        common = (sorted(set.intersection(*(set(t["members"]) for t in big)), key=lambda m: (len(m), m))
                  if len(big) >= 2 else [])
        expect_common = float(np.prod([t["machines"] for t in big]) / a["machines"] ** (len(big) - 1)) if big else 0.0
        pairs_txt = (f"{a['sig_edges']} machine pair is" if a["sig_edges"] == 1
                     else f"{a['sig_edges']} machine pairs are")
        L.append(f"- **Alibaba (shared clock), the real analysis: co-failure comes as cluster-wide alarm waves.** "
                 f"{a['machines']} machines, {a['incidents']} AE incidents in {a['days']:.0f} days (hours 0-72, "
                 f"never seen in training). Machines alarm together far more than chance: {a['edges']} gap10 edges "
                 f"vs {a['chance_edges']:.1f} ± {a['chance_edges_sd']:.1f} when each machine's incident times are "
                 f"shuffled (p = {a['p_weight']:.3f}), and {a['gap0_edges']} truly overlapping pairs vs "
                 f"{a['gap0_chance_edges']:.1f}. "
                 + (f"The excess sits in {len(big)} wave(s) of >= {WAVE_SIZES[-1]} machines ("
                    + "; ".join(f"hour {t['start_h']:.1f}: {t['machines']} machines" for t in big)
                    + f"), against {r8['chance_mean']:.2f} such waves by chance (p = {r8['p']:.3f})"
                    + (f"; {len(common)} machines are in every one of them ({', '.join(common)}; "
                       f"{expect_common:.1f} expected if the waves drew their machines at random), the closest "
                       "thing to a group that fails together on a real clock" if common else "")
                    + ". " if big and r8 else "")
                 + f"Pairwise, only {pairs_txt} beyond chance ({a['chance_sig_edges']:.1f} by luck): with "
                 f"{a['incidents_per_machine']:.1f} incidents per machine a pair co-occurs once or twice, which "
                 "shuffles also produce, so the per-pair test has little power here and the wave test is the "
                 "informative one. No labels: these are machines that alarm together, not confirmed faults.")
    if s:
        L.append(f"- **SMD (synthetic row-offset time), the method check.** {s['edges']} edges vs "
                 f"{s['chance_edges']:.0f} by chance, x{s['weight_ratio']:.2f} the chance co-occurring pairs; "
                 f"{s['sig_edges']} pairs beyond chance ({s['chance_sig_edges']:.1f} by luck, "
                 f"{_pct(s['sig_share_of_edges'])} of the edges), groups {_braces(s['groups'])}. SMD rows of "
                 "different machines are not known to be simultaneous, so part 1 shows that the method works "
                 "(both engines, the null check, the queries), not that SMD machines failed together.")
    if s and a:
        r5s, r5a = _wave_row(s["waves"], 5), _wave_row(a["waves"], 5)
        L.append(f"- **SMD vs Alibaba.** Share of edges beyond chance: {_pct(s['sig_share_of_edges'])} (SMD) vs "
                 f"{_pct(a['sig_share_of_edges'])} (Alibaba); observed / chance edges x"
                 f"{s['edges'] / s['chance_edges']:.2f} vs x{a['edges'] / a['chance_edges']:.2f}; machines in a "
                 f"pairwise co-failure group {s['machines_in_groups']}/{s['machines']} vs "
                 f"{a['machines_in_groups']}/{a['machines']} (largest {max(s['group_sizes'] or [0])} vs "
                 f"{max(a['group_sizes'] or [0])}); largest alarm wave {s['waves']['largest']} vs "
                 f"{a['waves']['largest']} machines"
                 + (f"; waves of >= 5 machines {r5s['observed']} (chance {r5s['chance_mean']:.1f}) vs "
                    f"{r5a['observed']} (chance {r5a['chance_mean']:.1f})" if r5s and r5a else "")
                 + ". SMD has many incidents per machine (dense graph, many testable pairs); Alibaba few per "
                 "machine but sharp synchronized waves on a real clock.")
    both = [x for x in (s, a) if x]
    L.append("- **Pregel (GraphFrames) and Neo4j (GDS) agree** on "
             + " and ".join(SOURCES[k]["name"] for k in h) + ": connected components = WCC on every graph: "
             + ("yes" if all(x["components_same_all"] for x in both) else "NO")
             + "; weighted PageRank top 5 in the same order: "
             + ("yes" if all(x["pagerank_top_same_all"] for x in both) else "NO")
             + f", largest difference {max(x['pagerank_max_diff'] for x in both):.1e}. Communities (LPA, Louvain) "
             "differ by algorithm, not by system.")
    return L


def part_lines(source: str, pregel: dict, neo: dict, cmp: dict, edges: dict[str, pd.DataFrame]) -> list[str]:
    n10, n0 = pregel["null"]["gap10"], pregel["null"]["gap0"]
    P, N = pregel["results"], neo["results"]
    sfx = SOURCES[source]["suffix"]
    L = ["### Input and graph", "",
         f"- Incidents: `{pregel['incidents']}`, {pregel['n_incidents']} incidents on "
         f"{pregel.get('machines_with_incidents', pregel['n_machines'])} of {pregel['n_machines']} machines "
         f"(scores: `{pregel.get('scores', '-')}`). Machines without an incident are isolated vertices."]
    if source == "smd" and "data/e8/root" in pregel["incidents"]:
        L.append("- Built for E8 because `data/incidents/source=smd` on this laptop held only the fake fixture: the "
                 "AE batch test scores (`data/scores_ae/source=smd`, is_anomaly = ae_flag) were copied to a staging "
                 "root and grouped there: `mkdir -p data/e8/root/data/scores/source=smd && cp "
                 "data/scores_ae/source=smd/scores.parquet data/e8/root/data/scores/source=smd/ && python "
                 "src/incidents/group.py --source smd --root data/e8/root`.")
    if source == "alibaba":
        L.append("- Built by `python experiments/e8_alibaba_input.py`: the original trace (no injected faults) on "
                 "hours 0-72, which `models/ae_alibaba` never saw (trained on hours 72-192, E5b), scored with "
                 "`autoencoder.score_layer` (is_anomaly = ae_flag), grouped with `src/incidents/group.py --source "
                 "alibaba --root data/e8/alibaba_root`. Pipeline layers untouched.")
    L += ["", "| graph | edges | total weight | components (Pregel / GDS) |", "| --- | ---: | ---: | --- |"]
    for name in GRAPHS:
        e = edges[name]
        L.append(f"| {name} | {len(e)} | {int(e['weight'].sum()) if len(e) else 0} | "
                 f"{cmp[name]['n_components_pregel']} / {cmp[name]['n_components_neo4j']} |")
    L += ["", "### Null check: edges expected by chance", "",
          "| gap | observed edges | chance edges, mean ± sd (95th pct) | observed weight | chance weight, mean ± sd "
          "| p (edges / weight) | edges beyond chance | expected beyond chance by luck |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for lab, n in (("10 min", n10), ("0 (overlap)", n0)):
        L.append(f"| {lab} | {n['observed_edges']} | {n['chance_edges_mean']:.1f} ± {n['chance_edges_sd']:.1f} "
                 f"({n['chance_edges_p95']:.0f}) | {n['observed_weight']} | {n['chance_weight_mean']:.1f} ± "
                 f"{n['chance_weight_sd']:.1f} | {n['p_edges']:.3f} / {n['p_weight']:.3f} | {n['sig_edges']} | "
                 f"{n['chance_sig_edges']:.1f} |")
    w = pregel["null"].get("waves")
    if w:
        when = ("hour of the trace" if source == "alibaba" else "row offset in hours, synthetic")
        L += ["", "### Alarm waves (incident level)", "",
              "Incidents of all machines chained in time order with the same 10-minute gap; a wave is a chain. "
              "Same 100 shuffles as above.", "",
              "| waves with >= k machines | observed | incidents in them | chance, mean ± sd | p |",
              "| --- | ---: | ---: | ---: | ---: |"]
        for r in w["by_size"]:
            L.append(f"| k = {r['min_machines']} | {r['observed']} | {r['incidents']} | {r['chance_mean']:.2f} ± "
                     f"{r['chance_sd']:.2f} | {r['p']:.3f} |")
        L += [f"| largest wave (machines) | {w['largest']} | | {w['chance_largest_mean']:.1f} (max "
              f"{w['chance_largest_max']}) | {w['p_largest']:.3f} |", "",
              f"Largest waves ({w['waves']} waves in all, {w['multi_machine_waves']} with 2+ machines):", "",
              f"| start ({when}) | minutes | machines | incidents | members |", "| ---: | ---: | ---: | ---: | --- |"]
        for t in w["top"]:
            L.append(f"| {t['start_h']:.2f} | {t['minutes']} | {t['machines']} | {t['incidents']} | "
                     f"{', '.join(short_id(m) for m in t['members'])} |")
    L += ["", "### Pregel (GraphFrames) vs Neo4j (GDS)", "",
          "| graph | same components? | PageRank top 5, Pregel | PageRank top 5, GDS | same top-5 set / order | "
          "Spearman | max abs diff | ARI LPA(GF) vs Louvain | ARI LPA(GF) vs LPA(GDS) |",
          "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    short = lambda ms: ", ".join(short_id(m) for m in ms)  # noqa: E731
    for name in GRAPHS:
        c = cmp[name]
        pr = c["pagerank"]
        L.append(f"| {name} | {'yes' if c['components_same'] else 'NO'} | {short(pr['top_a'])} | "
                 f"{short(pr['top_b'])} | {'yes' if pr['top_same_set'] else 'no'} / "
                 f"{'yes' if pr['top_same_order'] else 'no'} | {pr['spearman']:.3f} | {pr['max_abs_diff']:.2e} | "
                 f"{c['ari_lpa_gf_vs_louvain']:.2f} | {c['ari_lpa_gf_vs_lpa_gds']:.2f} |")
    L += ["", "Spearman below 1 with a difference near 1e-16 only reorders tied scores (isolated machines all have "
          "1 - d = 0.15).", "", "Groups found" + (" (machine ids without `machine-`)" if source == "smd" else "")
          + ":", ""]
    for name in GRAPHS:
        mod = cmp[name]["louvain_modularity"]
        L += [f"- **{name}** -- connected components: {_fmt_groups(P[name]['components'])}",
              f"  - LPA (GraphFrames): {_fmt_groups(P[name]['lpa'])}",
              f"  - Louvain (GDS{'' if mod is None else f', modularity {mod:.3f}'}): {_fmt_groups(N[name]['louvain'])}"]
    if source == "smd":
        vert_ids = sorted(P["gap10"]["components"])
        L += ["", "### Do the groups follow the SMD machine groups (1-x, 2-x, 3-x)?", "",
              "ARI = adjusted Rand index against the SMD group label (1 = identical partition, 0 = chance). "
              "Within-group share: fraction of edges joining two machines of the same SMD group, against the "
              "fraction of all machine pairs that are within one group.", "",
              "| graph | ARI components | ARI LPA (GF) | ARI Louvain | ARI LPA (GDS) | within-group edges | "
              "within-group weight | within-group pairs (baseline) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name in GRAPHS:
            a, mix = cmp[name]["ari_vs_smd_group"], group_mixing(edges[name], vert_ids)
            cmp[name]["mixing"] = mix
            L.append(f"| {name} | {a['components']:.2f} | {a['lpa_graphframes']:.2f} | {a['louvain']:.2f} | "
                     f"{a['lpa_gds']:.2f} | {mix['within_edges']}/{mix['edges']} ({_pct(mix['within_edge_share'])}) | "
                     f"{_pct(mix['within_weight_share'])} | {_pct(mix['within_pair_share'])} |")
    L += ["", "### Operator queries (Cypher, Neo4j)", ""]
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
    L += [f"![E8 {SOURCES[source]['name']} co-failure graph](e8_graph{sfx}.png)", ""]
    return L


def runtime_lines(source: str, pregel: dict, neo: dict) -> list[str]:
    t, e = pregel["timings"], pregel["env"]
    return [f"- **{SOURCES[source]['name']}** Pregel part ({e['master']}, parallelism {e['default_parallelism']}, "
            f"Spark {e['spark']}, `{e['graphframes']}`): total {t['total_s']:.1f} s (graphs + 100-shuffle null "
            f"model {t['graphs_and_null_s']:.1f} s; Spark start {t['spark_start_s']:.1f} s; "
            + "; ".join(f"{g} {t[g + '_s']:.1f} s" for g in GRAPHS)
            + f"). Neo4j part (Neo4j {neo['env'].get('neo4j')}, GDS {neo['env'].get('gds')}): total "
            f"{neo['timings']['total_s']:.1f} s (load {neo['timings']['load_s']:.1f} s, algorithms "
            f"{neo['timings']['algorithms_s']:.1f} s, queries {neo['timings']['queries_s']:.2f} s). Spark self-join "
            f"edges = pandas edges: {all(pregel['spark_edges_equal_pandas'].values())}."]


def render_md(parts: dict[str, dict], extra: dict | None = None) -> str:
    """parts[source] = {pregel, neo, cmp, edges, headline}."""
    h = {k: v["headline"] for k, v in parts.items()}
    L = ["# E8: graph processing on incidents -- which machines fail together?", "",
         "Generated by `python experiments/e8_graph.py report` from `data/e8/pregel[_alibaba].json` "
         "(`experiments/e8_graph.py pregel`, Spark GraphFrames) and `data/e8/neo4j[_alibaba].json` "
         "(`experiments/e8_neo4j.py`, Neo4j + GDS). Pipeline defaults are unchanged.", "",
         "Two datasets with the same method. **SMD** (part 1) has no timestamps: its 'times' are row offsets "
         "(row i of a machine = 2026-01-01 + i minutes), so it is only a method check. **Alibaba v2018** (part 2) "
         "has one shared clock for all machines, so co-occurring incidents there really happened at the same time; "
         "it is the co-failure analysis.", "",
         *summary_lines(h), "",
         "## Method (both datasets)", "",
         "- One vertex per machine. Edge when two incidents of different machines co-occur: the later one starts at "
         "most G minutes after the earlier one ends. Weight = number of co-occurring incident pairs.",
         "- `gap10`: G = 10 min (main; equals the incident session gap). `gap0`: the intervals must truly overlap "
         "(sensitivity). `sig`: `gap10` edges whose weight is above all 100 shuffles.",
         "- Null check: each machine's incident times are shuffled 100 times (seed 8): incident durations and idle "
         "gaps are permuted inside the machine's own time span, keeping its number of incidents, their lengths and "
         "the span, destroying only the alignment with other machines. 'Expected by luck' = leave-one-out count of "
         "pairs a shuffle would call beyond chance.",
         "- Alarm waves: the incidents of all machines chained in time order with the same 10-minute gap; the "
         "number of waves with >= 3 / 5 / 8 machines and the largest wave are compared with the same shuffles.",
         "- Pregel part: GraphFrames connectedComponents and labelPropagation, and weighted PageRank written with the "
         "GraphFrames Pregel API (GDS formula, d = 0.85, 30 iterations from 0). Neo4j part: the same CSV graph "
         "loaded with MERGE; GDS WCC, Louvain (weighted), LPA, PageRank (same formula); 3 operator queries.", "",
         "## SMD vs Alibaba", "", *comparison_lines(h), ""]
    if "smd" in parts:
        p = parts["smd"]
        L += ["## Part 1 -- SMD: method check (synthetic row-offset timestamps)", "",
              "**SMD timestamps are synthetic.** The dataset has no clock; row i of each machine was given "
              "2026-01-01T00:00Z + i minutes (train rows first, then test rows). Two machines' incidents 'co-occur' "
              "only in this row index; nothing says the rows were recorded at the same moment, and machines with "
              "longer training data start their test rows later (9 machines on 2026-01-20, the rest on "
              "2026-01-17). Read part 1 as a test of the method on a labelled dataset, not as evidence that "
              "machines failed together.", "",
              *part_lines("smd", p["pregel"], p["neo"], p["cmp"], p["edges"])]
    if "alibaba" in parts:
        p = parts["alibaba"]
        L += ["## Part 2 -- Alibaba v2018: co-failure on a shared clock", "",
              "All 40 machines share the trace clock (2026-01-01T00:00Z = start of the trace), so incidents that "
              "co-occur here happened at the same time. **Alibaba has no labels:** an incident is a window run the "
              "AE flagged, so the groups below are machines that *alarm together*, not confirmed faults.", "",
              *part_lines("alibaba", p["pregel"], p["neo"], p["cmp"], p["edges"])]
    L += ["## Runtime", ""]
    for k, p in parts.items():
        L += runtime_lines(k, p["pregel"], p["neo"])
    if extra and extra.get("alibaba_input"):
        ai = extra["alibaba_input"]
        L.append(f"- Alibaba input (`experiments/e8_alibaba_input.py`): {ai['total_s']:.1f} s (raw slice "
                 f"{ai['raw_s']:.1f} s, AE scoring {ai['score_s']:.1f} s, incidents {ai['incidents_s']:.1f} s); "
                 f"{ai['scores']['windows']} windows, {ai['scores']['flagged']} flagged.")
    L += ["", "Rerun the Pregel part on the Spark cluster (2 executors x 2 cores), on spark-master:", ""]
    for k in parts:
        L += ["```bash", cluster_command(k), "```", ""]
    L += ["## Caveats", "",
          "- SMD: synthetic row-offset timestamps (part 1); the null model keeps each machine's own span, which "
          "controls for the two start dates, but an SMD edge is not evidence of a shared physical cause.",
          "- Alibaba: no labels; incidents are AE alarms (is_anomaly = ae_flag) on 3 days, so a group means machines "
          "that alarm together. A shared alarm can also come from a cluster-wide event (deployment, batch job "
          "wave) that is normal for the operator.",
          "- Incidents are AE alarms in both datasets; false alarms are part of the graph.",
          "- The per-pair test needs several co-occurrences per pair: with ~3 incidents per Alibaba machine a pair "
          "co-occurs at most a few times, so few pairs can pass it even when the graph as a whole is far above "
          "chance. The alarm-wave test (count of waves with >= k machines) is the better test there.",
          "- Each machine pair is tested at an empirical p of 1/101 with no further multiple-testing correction "
          "(378 SMD pairs, 780 Alibaba pairs). 'Expected by luck' estimates how many beyond-chance edges luck "
          "alone gives.",
          "- LPA (GraphFrames and GDS) is unweighted and breaks ties differently in the two systems; Louvain is "
          "weighted. Communities are compared by ARI; components and PageRank are deterministic and compared "
          "exactly."]
    return "\n".join(L) + "\n"


def cluster_command(source: str = "smd") -> str:
    cfg = SOURCES[source]
    pre = ("# only if data/incidents/source=smd is missing there: python src/incidents/group.py --source smd\n"
           if source == "smd" else
           "# builds the Alibaba incidents on the cluster first (needs PyTorch and models/ae_alibaba):\n"
           "python experiments/e8_alibaba_input.py\n")
    return ("cd /srv/spark-shared/Anomaly-Detection && source .venv/bin/activate\n" + pre +
            "spark-submit --master spark://spark-master:7077 --executor-cores 2 --total-executor-cores 4 \\\n"
            f"  --packages {GRAPHFRAMES_PACKAGE} --repositories {SPARK_PACKAGES_REPO} \\\n"
            "  --exclude-packages org.slf4j:slf4j-api \\\n"
            f"  experiments/e8_graph.py pregel --source {source} --incidents {cfg['incidents'].as_posix()} \\\n"
            f"  --scores {cfg['scores'].as_posix()} --out-name pregel{cfg['suffix']}_cluster.json")


def load_part(out_dir: Path, source: str) -> dict | None:
    pf, nf = out_file(out_dir, "pregel", source, "json"), out_file(out_dir, "neo4j", source, "json")
    if not (pf.exists() and nf.exists()):
        return None
    pregel = json.loads(pf.read_text(encoding="utf-8"))
    neo = json.loads(nf.read_text(encoding="utf-8"))
    vert, edges = read_graph_files(out_dir, source)
    cmp = compare(pregel, neo, source)
    return {"pregel": pregel, "neo": neo, "cmp": cmp, "edges": edges, "vert": vert,
            "headline": headline(pregel, neo, cmp, edges)}


def cmd_report(args) -> int:
    parts = {k: p for k in SOURCES if (p := load_part(args.out_dir, k)) is not None}
    if not parts:
        print(f"no pregel*.json + neo4j*.json pairs under {args.out_dir}", file=sys.stderr)
        return 2
    extra = {}
    ai = ROOT / ALIBABA_ROOT / "input.json" if args.alibaba_input is None else Path(args.alibaba_input)
    if "alibaba" in parts and ai.exists():
        extra["alibaba_input"] = json.loads(ai.read_text(encoding="utf-8"))
    md = render_md(parts, extra)  # also fills cmp[...]["mixing"] for SMD
    from experiments import e8_cascade as C
    cascade = C.load_results()
    cascade_png = args.png.with_name(f"{args.png.stem}_cascade.png")
    if cascade is not None:
        md = md.rstrip("\n") + "\n" + "\n".join(C.section_lines(cascade, cascade_png.name)) + "\n"
    args.md.write_text(md, encoding="utf-8")
    doc = {"sources": list(parts), "comparison": {k: p["headline"] for k, p in parts.items()}, **extra}
    if cascade is not None:
        pg = cascade["pregel"]
        doc["cascade"] = {"rules": pg["rules"], "over_seeds": pg["over_seeds"],
                          "seeds": [{"seed": r["seed"], "truth": r["truth"], "score": r["score"],
                                     "agreement": r["agreement"], "edges": r["edges"], "roots": r["roots"]}
                                    for r in pg["seeds"]],
                          "baseline": {k: pg["baseline"][k] for k in ("false_groups", "agreement", "n_edges",
                                                                      "group_sizes")},
                          "stream": cascade["stream"], "neo4j": cascade["neo4j"], "pregel_total_s": pg["total_s"]}
    for k, p in parts.items():
        doc[k] = {"pregel": p["pregel"], "neo4j": p["neo"], "compare": p["cmp"],
                  "sig_edges": p["edges"]["sig"].to_dict(orient="records"), "cluster_command": cluster_command(k)}
    args.json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    pngs = []
    for k, p in parts.items():
        png = args.png if k == "smd" else args.png.with_name(f"{args.png.stem}{SOURCES[k]['suffix']}.png")
        inc = None
        if k == "alibaba":
            f = ROOT / p["pregel"]["incidents"]
            inc = load_incidents(f) if f.exists() else None
        plot(png, p["vert"], p["edges"]["gap10"], p["neo"], k, inc, p["pregel"]["null"].get("waves"))
        pngs.append(png)
    if cascade is not None:
        C.plot(cascade_png, cascade)
        pngs.append(cascade_png)
    for f in (args.md, args.json, *pngs):
        print(f"wrote {rel_path(f)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pregel", help="graphs, null model, GraphFrames CC / LPA / Pregel PageRank")
    p.add_argument("--source", choices=sorted(SOURCES), default="smd")
    p.add_argument("--incidents", type=Path, default=None, help="default: SOURCES[source]['incidents']")
    p.add_argument("--scores", type=Path, default=None, help="scores the incidents came from (spans, vertex set)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--out-name", default=None, help="default pregel<suffix>.json")
    p.add_argument("--master", default=None, help="Spark master (default local[*], or spark-submit's)")
    p.add_argument("--checkpoint-dir", default=CKPT_DIR)
    p.add_argument("--shuffles", type=int, default=N_SHUFFLES)
    p.add_argument("--seed", type=int, default=SEED)
    r = sub.add_parser("report", help="docs/results/e8_graph.md + .json + .png (one PNG per source)")
    r.add_argument("--out-dir", type=Path, default=OUT_DIR)
    r.add_argument("--md", type=Path, default=DOC_MD)
    r.add_argument("--json", type=Path, default=DOC_JSON)
    r.add_argument("--png", type=Path, default=DOC_PNG)
    r.add_argument("--alibaba-input", type=Path, default=None, help="default data/e8/alibaba_root/input.json")
    args = ap.parse_args(argv)
    return cmd_pregel(args) if args.cmd == "pregel" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
