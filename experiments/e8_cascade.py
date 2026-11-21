"""E8c: injected cascading faults on Alibaba -- does the stream catch them, and does the graph recover the group and root?

Run from the repo root (venv active; Kafka up: docker compose -f infra/docker-compose.yml up -d; PyTorch from
requirements-ml.txt; Neo4j for step 3: docker compose -f infra/docker-compose.neo4j.yml up -d, NEO4J_PASSWORD set):
    python experiments/e8_cascade.py stream             # inject + Kafka + score_job + incidents, seeds 7 8 9
    python experiments/e8_cascade.py pregel             # directed graphs, GraphFrames components / Pregel roots / reverse PageRank, scoring
    python experiments/e8_cascade.py neo4j              # the same graphs in Neo4j: GDS WCC + reverse PageRank, Cypher queries
    python experiments/e8_graph.py report               # adds the E8c section and e8_graph_cascade.png to docs/results

Rules, fixed in this file before any result was looked at (the report repeats them):
1. Injection (src/producer/inject_faults.py --mode cascade): the E5b slice (first 40 machines, hours 0-48). Per seed
   6 cascades, each = 1 random root + 4 random dependents, root fault at T (5-minute aligned), each dependent at
   T + random 2-8 min, every fault 15 min of net_in x 0.1 ("break"). Starts in hours 24-47.5 (the last 30 min are left
   out so that every cascade window, with the 10-minute edge rule, is closed by the stream's watermark before the
   replay ends), at least 2 h apart, at least 1 h from the two clean-data alarm waves of E8 (hour 23.5 for 55 min,
   hour 62.0 for 50 min), disjoint machines, not inside a gap of the trace (at least 5 machines with data over
   the whole cascade). Seeds 7, 8, 9. Truth -> seed_<n>/inject/cascade_truth.json.
2. Stream: the injected raw layer is replayed with src/producer/replay.py into Kafka telemetry.alibaba (speedup 0),
   and src/streaming/score_job.py --source alibaba --input kafka --once reads exactly those records (starting offsets
   recorded before the replay) and scores them with models/ae_alibaba; is_anomaly = ae_flag. Incidents:
   src/incidents/group.py run_batch into data/e8/cascade_root/seed_<n>. The first 24 hours are the AE's history
   (recentring, adaptive limit), as in E5b.
3. Graph: 40 vertices (all machines of the slice); incidents with onset in hours 24-48 only. Directed edge u -> v
   (u != v) for each incident pair where v's incident starts strictly after u's starts and no later than 10 min after
   u's ends; weight = number of such pairs. Onset of a machine = its earliest incident onset in that range.
4. Pregel (GraphFrames): connectedComponents (weak); root of a component = Pregel message passing of the earliest
   onset along edges in both directions, then the vertex whose onset equals the component minimum and whose
   in-degree is 0 (several such vertices = a tie; none = no root). Reverse PageRank (edges reversed, weighted, the
   GDS formula of E8, 30 iterations) as a secondary ranking. Checked against a pandas reference and against Neo4j GDS.
5. Scoring of a cascade, window W = [T, end of its last fault + 10 min]:
   - detected: at least one member has an incident overlapping W; members alarmed = members with one.
   - recovered component: of the components that contain an alarmed member, the one with the most members
     (ties: the one whose alarmed member alarmed first). Member recall = members in it / 5; precision = members in
     it / its size (undetected cascade: recall 0, precision not defined).
   - root correct: the Pregel root of the recovered component is exactly the true root (a tie that includes it is
     counted separately, not as correct). Reverse PageRank: the top vertex of the component is the true root.
   - false groups: components of >= 3 machines that are no cascade's recovered component; and, on the clean hours
     0-72 baseline of E8 (data/e8/alibaba_root, offline scores), every component of >= 3 machines.
   Secondary, also fixed now: a time-resolved variant on the same edges, components over incidents instead of
   machines (alarm chains), so unrelated incidents of one machine at different times cannot join two groups; its
   root is the machine of the chain's earliest incident; false groups = chains of >= 3 machines with no incident
   overlapping any cascade window. Reported per seed and as mean (min-max) over seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from contracts import schemas as S  # noqa: E402
from experiments import e8_graph as G  # noqa: E402

SOURCE = "alibaba"
SEEDS = (7, 8, 9)
MACHINES, HOURS = 40, 48
WINDOW_H = (24.0, 47.5)                 # cascade starts and ends (see rule 1)
SCOPE_H = (24.0, 48.0)                  # incidents in the graph (rule 3)
AVOID = ((23.5, 55.0), (62.0, 50.0))    # E8 clean-data alarm waves: start hour, minutes
N_CASCADES, CASCADE_TYPE = 6, "break"
GAP_MIN = 10
MIN_GROUP = 3
DEMO_SEED, DEMO_CASCADE = 7, "c1"
SENTINEL = 1e15                         # onset of a machine without incidents (no edges, never a root)
MODEL_DIR = Path("models/ae_alibaba")
CASCADE_ROOT = Path("data/e8/cascade_root")
BASELINE_INCIDENTS = G.ALIBABA_ROOT / S.PATHS["incidents"] / f"source={SOURCE}"
OUT_DIR = Path("data/e8")
PREGEL_JSON, NEO4J_JSON, STREAM_JSON = "cascade_pregel.json", "neo4j_cascade.json", "cascade_stream.json"
CKPT_ROOT = Path("checkpoints/e8_cascade")
T0_MIN = G.T0_MINUTES


def seed_dir(seed: int, root: Path = CASCADE_ROOT) -> Path:
    return ROOT / root / f"seed_{seed}"


def minutes_of(iso: str) -> int:
    return int(pd.Timestamp(iso).timestamp() // 60)


def hours_of(minute: float) -> float:
    return (minute - T0_MIN) / 60


# ---- step 1: inject, Kafka, score_job, incidents ------------------------------------------------------------------
def topic_end_offsets(topic: str = S.TOPIC_ALIBABA, bootstrap: str = S.KAFKA_BOOTSTRAP) -> dict[int, int]:
    from confluent_kafka import Consumer, TopicPartition
    from confluent_kafka.admin import AdminClient
    md = AdminClient({"bootstrap.servers": bootstrap}).list_topics(topic, timeout=10)
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": "e8c-offsets", "enable.auto.commit": False})
    try:
        return {p: c.get_watermark_offsets(TopicPartition(topic, p), timeout=10)[1]
                for p in sorted(md.topics[topic].partitions)}
    finally:
        c.close()


def starting_offsets_json(offsets: dict[int, int], topic: str = S.TOPIC_ALIBABA) -> str:
    return json.dumps({topic: {str(p): int(o) for p, o in sorted(offsets.items())}})


def score_job_cmd(out_dir: Path, ckpt: Path, offsets: dict[int, int]) -> list[str]:
    return [sys.executable, "src/streaming/score_job.py", "--input", "kafka", "--source", SOURCE, "--once",
            "--out-dir", Path(out_dir).as_posix(), "--checkpoint-dir", Path(ckpt).as_posix(),
            "--starting-offsets", starting_offsets_json(offsets)]


def stream_vs_offline(model_dir: Path, raw_dir: Path, scores_dir: Path) -> dict:
    """The stream's ae_flag against autoencoder.score_layer on the same injected raw layer (windows in both)."""
    from src.detection import autoencoder as M
    meta, _, flag, _ = M.score_layer(model_dir, raw_dir)
    off = pd.DataFrame({"entity_id": meta["entity_id"], "ws": meta["window_start"].astype(np.int64), "off": flag})
    st = pd.read_parquet(scores_dir, columns=["entity_id", "window_start", "ae_flag"])
    st = pd.DataFrame({"entity_id": st["entity_id"].astype(str),
                       "ws": ((pd.to_datetime(st["window_start"], utc=True) - pd.Timestamp(0, tz="UTC"))
                              // pd.Timedelta(microseconds=1)).astype(np.int64),
                       "stream": st["ae_flag"].fillna(False).astype(bool)})
    both = off.merge(st, on=["entity_id", "ws"], how="inner")
    return {"offline_windows": int(len(off)), "stream_windows": int(len(st)), "common_windows": int(len(both)),
            "same_flag": int((both["off"] == both["stream"]).sum()), "offline_flags": int(both["off"].sum()),
            "stream_flags": int(both["stream"].sum())}


def run_seed_stream(seed: int, spark, root: Path = CASCADE_ROOT) -> dict:
    from src.incidents import group as GR
    from src.producer import inject_faults as I
    from src.producer import replay as R
    d = seed_dir(seed, root)
    out, t = {"seed": seed}, time.perf_counter()
    truth = I.inject_cascades(ROOT, d / "inject", machines=MACHINES, hours=HOURS, seed=seed, n=N_CASCADES,
                              window_h=WINDOW_H, avoid=list(AVOID), kind=CASCADE_TYPE)
    out["inject_s"] = time.perf_counter() - t
    t = time.perf_counter()
    before = topic_end_offsets()
    if R.main(["--source", SOURCE, "--root", str(d / "inject"), "--speedup", "0"]) != 0:
        raise RuntimeError("replay failed")
    after = topic_end_offsets()
    out["kafka_records"] = int(sum(after.values()) - sum(before.values()))
    out["replay_s"] = time.perf_counter() - t
    if out["kafka_records"] != truth["rows"]:
        raise RuntimeError(f"seed {seed}: {out['kafka_records']} records in Kafka, {truth['rows']} rows injected")
    scores = GR.scores_dir(d, SOURCE)
    ckpt = ROOT / CKPT_ROOT / f"seed_{seed}"
    for x in (scores, ckpt, GR.incidents_dir(d, SOURCE)):
        shutil.rmtree(x, ignore_errors=True)
    cmd = score_job_cmd(scores.relative_to(ROOT), ckpt.relative_to(ROOT), before)   # score_job runs in ROOT
    out["score_job_cmd"] = " ".join(c if " " not in c and "{" not in c else f"'{c}'" for c in
                                    ["python", *cmd[1:]])
    t = time.perf_counter()
    with (d / "score_job.log").open("w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=3600).returncode
    out["score_job_s"] = time.perf_counter() - t
    if rc != 0:
        raise RuntimeError(f"score_job failed (exit {rc}); see {d / 'score_job.log'}")
    t = time.perf_counter()
    out["parity"] = stream_vs_offline(ROOT / MODEL_DIR, I.raw_dir(d / "inject"), scores)
    out["parity_s"] = time.perf_counter() - t
    t = time.perf_counter()
    out["incidents"] = GR.run_batch(SOURCE, d, spark)
    out["incidents_s"] = time.perf_counter() - t
    print(f"seed {seed}: {out['kafka_records']} records, stream windows {out['parity']['stream_windows']}, "
          f"flags {out['parity']['stream_flags']}, same flag as offline {out['parity']['same_flag']}/"
          f"{out['parity']['common_windows']}, {out['incidents']} incidents", flush=True)
    return out


def cmd_stream(args) -> int:
    from src.incidents import group as GR
    t0 = time.perf_counter()
    spark = GR.build_spark("e8_cascade_incidents")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        seeds = [run_seed_stream(s, spark, args.root) for s in args.seeds]
    finally:
        spark.stop()
    doc = {"seeds": seeds, "total_s": time.perf_counter() - t0}
    (ROOT / args.out_dir).mkdir(parents=True, exist_ok=True)
    (ROOT / args.out_dir / STREAM_JSON).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"stream part done in {doc['total_s']:.1f} s -> {(args.out_dir / STREAM_JSON).as_posix()}")
    return 0


# ---- graph (pandas reference) -------------------------------------------------------------------------------------
def in_scope(inc: pd.DataFrame, scope_h: tuple[float, float] | None) -> pd.DataFrame:
    if scope_h is None:
        return inc.reset_index(drop=True)
    lo, hi = T0_MIN + scope_h[0] * 60, T0_MIN + scope_h[1] * 60
    return inc[(inc["start"] >= lo) & (inc["start"] < hi)].reset_index(drop=True)


def directed_pairs(inc: pd.DataFrame, gap: float = GAP_MIN) -> pd.DataFrame:
    """Incident pairs (i -> j): different machines, start_j > start_i and start_j <= end_i + gap."""
    e = inc["entity_id"].to_numpy()
    s, en = inc["start"].to_numpy(), inc["end"].to_numpy()
    ok = (s[None, :] > s[:, None]) & (s[None, :] <= en[:, None] + gap) & (e[None, :] != e[:, None])
    i, j = np.nonzero(ok)
    return pd.DataFrame({"i": i, "j": j, "src": e[i], "dst": e[j]})


def directed_edges(inc: pd.DataFrame, gap: float = GAP_MIN) -> pd.DataFrame:
    p = directed_pairs(inc, gap)
    if p.empty:
        return pd.DataFrame({"src": pd.Series(dtype=str), "dst": pd.Series(dtype=str),
                             "weight": pd.Series(dtype=np.int64)})
    w = p.groupby(["src", "dst"]).size().rename("weight").reset_index()
    w["weight"] = w["weight"].astype(np.int64)
    return w.sort_values(["src", "dst"]).reset_index(drop=True)


def onsets(inc: pd.DataFrame) -> dict[str, int]:
    return {m: int(v) for m, v in inc.groupby("entity_id")["start"].min().items()}


def _union_find(items, pairs):
    parent = {x: x for x in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {x: find(x) for x in items}


def components(machines, edges: pd.DataFrame) -> dict[str, str]:
    """Weak components; label = smallest machine id of the component."""
    return _union_find(sorted(machines), zip(edges["src"], edges["dst"]))


def members_of(comp: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for m, c in comp.items():
        out.setdefault(c, []).append(m)
    return {c: sorted(v) for c, v in out.items()}


def roots(comp: dict, onset: dict, edges: pd.DataFrame) -> dict[str, list[str]]:
    """Per component: machines whose onset equals the component's earliest and that have in-degree 0."""
    indeg = edges.groupby("dst").size().to_dict() if len(edges) else {}
    out = {}
    for c, ms in members_of(comp).items():
        on = [onset[m] for m in ms if m in onset]
        first = min(on) if on else None
        out[c] = sorted(m for m in ms if m in onset and onset[m] == first and indeg.get(m, 0) == 0)
    return out


def reverse_pagerank_reference(machines, edges: pd.DataFrame, d: float = G.DAMPING, iters: int = G.PR_ITERS) -> dict:
    """PageRank on the reversed edges, the formula of e8_graph.pregel_pagerank (init PR_INIT, no dangling share)."""
    rank = {m: G.PR_INIT for m in machines}
    rev = edges.rename(columns={"src": "dst", "dst": "src"})
    wout = rev.groupby("src")["weight"].sum().to_dict()
    for _ in range(iters):
        msg = {m: 0.0 for m in machines}
        for a, b, w in zip(rev["src"], rev["dst"], rev["weight"]):
            msg[b] += rank[a] * w / wout[a]
        rank = {m: (1 - d) + d * msg[m] for m in machines}
    return rank


def incident_chains(inc: pd.DataFrame, gap: float = GAP_MIN) -> list[dict]:
    """Time-resolved variant: components over incidents (same pair rule). Each chain: machines, start, end, roots
    (machine(s) of the earliest incident), incident indices."""
    p = directed_pairs(inc, gap)
    lab = _union_find(range(len(inc)), zip(p["i"], p["j"]))
    out = []
    for c in sorted(set(lab.values())):
        idx = [i for i, v in lab.items() if v == c]
        part = inc.iloc[idx]
        first = part["start"].min()
        out.append({"machines": sorted(set(part["entity_id"])), "start": int(first), "end": int(part["end"].max()),
                    "roots": sorted(set(part.loc[part["start"] == first, "entity_id"])), "incidents": idx})
    return out


def reference(machines, inc: pd.DataFrame, gap: float = GAP_MIN) -> dict:
    edges = directed_edges(inc, gap)
    comp = components(machines, edges)
    on = onsets(inc)
    return {"edges": edges, "components": comp, "onsets": on, "roots": roots(comp, on, edges),
            "revpr": reverse_pagerank_reference(sorted(machines), edges), "chains": incident_chains(inc, gap)}


# ---- scoring (rule 5) ---------------------------------------------------------------------------------------------
def cascade_window(c: dict, gap: float = GAP_MIN) -> tuple[int, int]:
    return minutes_of(c["start_time"]), minutes_of(c["end_time"]) + int(gap)


def overlaps(inc: pd.DataFrame, w: tuple[int, int]) -> pd.Series:
    return (inc["start"] < w[1]) & (inc["end"] > w[0])


def score_cascade(c: dict, inc: pd.DataFrame, comp: dict, root_by_comp: dict, revpr: dict, chains: list[dict],
                  gap: float = GAP_MIN) -> dict:
    members, true_root = list(c["members"]), c["root"]
    w = cascade_window(c, gap)
    hit = inc[overlaps(inc, w) & inc["entity_id"].isin(members)]
    first_alarm = hit.groupby("entity_id")["start"].min().to_dict()
    out = {"cascade_id": c["cascade_id"], "root": true_root, "members": members, "start_h": hours_of(w[0]),
           "members_alarmed": sorted(first_alarm), "detected": bool(first_alarm),
           "root_alarmed": true_root in first_alarm,
           "root_delay_min": (first_alarm[true_root] - w[0]) if true_root in first_alarm else None}
    # descriptive, not a rule: can the stream's 5-minute onsets order the root before its dependents at all?
    if true_root in first_alarm:
        rt = first_alarm[true_root]
        others = [v for m, v in first_alarm.items() if m != true_root]
        out["root_onset"] = ("first" if all(v > rt for v in others) else
                             "tied" if all(v >= rt for v in others) else "later")
        out["tied_offsets_min"] = sorted(f["offset_min"] for f in c.get("faults", []) if f.get("role") == "dependent"
                                         and first_alarm.get(f["entity_id"]) == rt)
    else:
        out["root_onset"], out["tied_offsets_min"] = None, []
    groups = members_of(comp)
    if first_alarm:
        cands = {comp[m] for m in first_alarm}
        best = min(cands, key=lambda k: (-len(set(groups[k]) & set(members)),
                                         min(first_alarm[m] for m in groups[k] if m in first_alarm), k))
        g = groups[best]
        r = root_by_comp.get(best, [])
        top = max(sorted(g), key=lambda m: revpr.get(m, 0.0))
        k = len(set(g) & set(members))
        out["main"] = {"component": g, "recall": k / len(members), "precision": k / len(g), "roots": r,
                       "root_correct": r == [true_root], "root_in_tie": len(r) > 1 and true_root in r,
                       "revpr_top": top, "revpr_correct": top == true_root, "component_id": best}
        idx = set(hit.index)
        cands = [ch for ch in chains if idx & set(ch["incidents"])]
        best_ch = min(cands, key=lambda ch: (-len(set(ch["machines"]) & set(members)), ch["start"]))
        k = len(set(best_ch["machines"]) & set(members))
        out["time"] = {"component": best_ch["machines"], "recall": k / len(members),
                       "precision": k / len(best_ch["machines"]), "roots": best_ch["roots"],
                       "root_correct": best_ch["roots"] == [true_root],
                       "root_in_tie": len(best_ch["roots"]) > 1 and true_root in best_ch["roots"],
                       "start_h": hours_of(best_ch["start"])}
    else:
        miss = {"component": [], "recall": 0.0, "precision": None, "roots": [], "root_correct": False,
                "root_in_tie": False}
        out["main"] = {**miss, "revpr_top": None, "revpr_correct": False, "component_id": None}
        out["time"] = dict(miss)
    return out


def false_groups(comp: dict, matched: set, chains: list[dict], inc: pd.DataFrame, windows: list[tuple[int, int]],
                 min_size: int = MIN_GROUP) -> dict:
    main = [g for k, g in members_of(comp).items() if len(g) >= min_size and k not in matched]
    in_w = set()
    for w in windows:
        in_w |= set(inc.index[overlaps(inc, w)])
    time_ = [ch for ch in chains if len(ch["machines"]) >= min_size and not (set(ch["incidents"]) & in_w)]
    return {"main": len(main), "main_groups": main, "time": len(time_),
            "time_groups": [{"start_h": hours_of(ch["start"]), "machines": ch["machines"]} for ch in time_]}


def score_run(truth: dict, inc: pd.DataFrame, comp: dict, root_by_comp: dict, revpr: dict, chains: list[dict],
              gap: float = GAP_MIN) -> dict:
    per = [score_cascade(c, inc, comp, root_by_comp, revpr, chains, gap) for c in truth["cascades"]]
    matched = {p["main"]["component_id"] for p in per if p["main"]["component_id"] is not None}
    fg = false_groups(comp, matched, chains, inc, [cascade_window(c, gap) for c in truth["cascades"]])
    return {"cascades": per, "false_groups": fg, "summary": summarise_cascades(per, fg)}


def summarise_cascades(per: list[dict], fg: dict) -> dict:
    n = len(per)
    det = [p for p in per if p["detected"]]
    out = {"cascades": n, "detected": len(det), "members": sum(len(p["members"]) for p in per),
           "members_alarmed": sum(len(p["members_alarmed"]) for p in per),
           "roots_alarmed": sum(p["root_alarmed"] for p in per)}
    delays = [p["root_delay_min"] for p in per if p["root_delay_min"] is not None]
    out["root_delay_min_mean"] = float(np.mean(delays)) if delays else None
    out["root_onset"] = {k: sum(p.get("root_onset") == k for p in per) for k in ("first", "tied", "later")}
    out["tied_offsets_min"] = sorted(o for p in per for o in p.get("tied_offsets_min", []))
    for v in ("main", "time"):
        prec = [p[v]["precision"] for p in det]
        out[v] = {"recall_mean": float(np.mean([p[v]["recall"] for p in per])) if per else None,
                  "precision_mean": float(np.mean(prec)) if prec else None,
                  "root_correct": sum(p[v]["root_correct"] for p in per),
                  "root_in_tie": sum(p[v]["root_in_tie"] for p in per),
                  "full_recovery": sum(p[v]["recall"] == 1.0 and p[v]["precision"] == 1.0 for p in det),
                  "false_groups": fg[v]}
        if v == "main":
            out[v]["revpr_correct"] = sum(p[v]["revpr_correct"] for p in per)
    return out


def mean_min_max(vals) -> dict | None:
    vals = [v for v in vals if v is not None]
    return {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals))} if vals else None


def over_seeds(summaries: list[dict]) -> dict:
    out = {k: mean_min_max([s[k] for s in summaries]) for k in
           ("detected", "members_alarmed", "roots_alarmed", "root_delay_min_mean")}
    for v in ("main", "time"):
        out[v] = {k: mean_min_max([s[v].get(k) for s in summaries]) for k in
                  ("recall_mean", "precision_mean", "root_correct", "root_in_tie", "full_recovery", "false_groups",
                   "revpr_correct") if k in summaries[0][v]}
    return out


# ---- step 2: Pregel (GraphFrames) ---------------------------------------------------------------------------------
def spark_directed_edges(spark, inc: pd.DataFrame, gap: float = GAP_MIN) -> pd.DataFrame:
    """directed_edges as a Spark self-join (what runs on a cluster)."""
    from pyspark.sql import functions as F
    df = G._local_df(spark, inc[["entity_id", "start", "end"]].astype({"start": "int64", "end": "int64"}))
    a, b = df.alias("a"), df.alias("b")
    cond = ((F.col("a.entity_id") != F.col("b.entity_id")) & (F.col("b.start") > F.col("a.start"))
            & (F.col("b.start") <= F.col("a.end") + F.lit(gap)))
    out = (a.join(b, cond).groupBy(F.col("a.entity_id").alias("src"), F.col("b.entity_id").alias("dst"))
           .agg(F.count(F.lit(1)).alias("weight")).toPandas())
    out["weight"] = out["weight"].astype(np.int64)
    return out.sort_values(["src", "dst"]).reset_index(drop=True)


def pregel_cascade(spark, machines, onset: dict, edges: pd.DataFrame) -> dict:
    """GraphFrames connectedComponents; Pregel min-onset propagation (both directions) -> roots; Pregel reverse
    PageRank. Returns components {id: component}, minon, roots {component: [ids]}, revpr, iterations, timings."""
    from graphframes import GraphFrame
    from graphframes.lib import Pregel
    from pyspark.sql import functions as F
    vert = pd.DataFrame({"id": sorted(machines)})
    vert["onset"] = [float(onset.get(m, SENTINEL)) for m in vert["id"]]
    out, t = {}, time.perf_counter()
    v = G._local_df(spark, vert)
    e = G._graphframe(spark, G.vertices(machines), edges).edges
    g = GraphFrame(v, e)
    cc = g.connectedComponents()
    out["components"] = {r["id"]: int(r["component"]) for r in cc.select("id", "component").collect()}
    out["cc_s"] = time.perf_counter() - t
    t = time.perf_counter()
    sizes = pd.Series(out["components"]).value_counts()
    iters = max(1, int(sizes.max()) - 1)   # min onset crosses any component in at most size-1 steps
    if len(edges):
        res = (Pregel(g).setMaxIter(iters).setCheckpointInterval(G.PREGEL_CKPT_EVERY)
               .withVertexColumn("minon", F.col("onset"),
                                 F.least(F.col("minon"), F.coalesce(Pregel.msg(), F.col("minon"))))
               .sendMsgToDst(Pregel.src("minon")).sendMsgToSrc(Pregel.dst("minon"))
               .aggMsgs(F.min(Pregel.msg())).run())
        minon = {r["id"]: float(r["minon"]) for r in res.select("id", "minon").collect()}
    else:
        minon = dict(zip(vert["id"], vert["onset"]))
    indeg = {r["id"]: int(r["inDegree"]) for r in g.inDegrees.collect()} if len(edges) else {}
    out["pregel_iterations"] = iters if len(edges) else 0
    out["minon"] = minon
    by_comp: dict[int, list[str]] = {}
    for m, c in out["components"].items():
        if onset.get(m) is not None and float(onset[m]) == minon[m] and indeg.get(m, 0) == 0:
            by_comp.setdefault(c, []).append(m)
    out["roots"] = {str(c): sorted(by_comp.get(c, [])) for c in sorted(set(out["components"].values()))}
    out["roots_s"] = time.perf_counter() - t
    t = time.perf_counter()
    out["revpr"] = pregel_reverse_pagerank(spark, machines, edges)
    out["revpr_s"] = time.perf_counter() - t
    return out


def pregel_reverse_pagerank(spark, machines, edges: pd.DataFrame, d: float = G.DAMPING,
                            iters: int = G.PR_ITERS) -> dict:
    from graphframes import GraphFrame
    from graphframes.lib import Pregel
    from pyspark.sql import functions as F
    rev = edges.rename(columns={"src": "dst", "dst": "src"})[["src", "dst", "weight"]]
    if rev.empty:
        return {m: 1.0 - d for m in machines}
    g = G._graphframe(spark, G.vertices(machines), rev)
    wout = g.edges.groupBy(F.col("src").alias("id")).agg(F.sum("weight").cast("double").alias("wout"))
    v = G._jvm_df(g.vertices.join(wout, "id", "left").fillna(0.0, subset=["wout"]))
    ranks = (Pregel(GraphFrame(v, g.edges)).setMaxIter(iters).setCheckpointInterval(G.PREGEL_CKPT_EVERY)
             .withVertexColumn("rank", F.lit(G.PR_INIT),
                               F.lit(1.0 - d) + F.lit(d) * F.coalesce(Pregel.msg(), F.lit(0.0)))
             .sendMsgToDst(Pregel.src("rank") * Pregel.edge("weight") / Pregel.src("wout"))
             .aggMsgs(F.sum(Pregel.msg())).run())
    return {r["id"]: float(r["rank"]) for r in ranks.select("id", "rank").collect()}


def relabel(pregel_comp: dict) -> dict:
    """GraphFrames component ids -> the smallest member id, so they compare with the pandas labels."""
    first = {}
    for m in sorted(pregel_comp):
        first.setdefault(pregel_comp[m], m)
    return {m: first[c] for m, c in pregel_comp.items()}


def analyse_graph(spark, name: str, machines, inc: pd.DataFrame, truth: dict | None) -> dict:
    """One graph: Spark edges, Pregel, the pandas reference, agreement, and (with truth) the cascade scoring."""
    t = time.perf_counter()
    ref = reference(machines, inc)
    sp_edges = spark_directed_edges(spark, inc) if len(inc) else ref["edges"]
    edges_s = time.perf_counter() - t
    pg = pregel_cascade(spark, machines, ref["onsets"], sp_edges)
    comp = relabel(pg["components"])
    root_by_comp = {}
    for c, rs in pg["roots"].items():
        if rs:
            root_by_comp[comp[rs[0]]] = rs
    for c in set(comp.values()):
        root_by_comp.setdefault(c, [])
    ref_roots = {c: r for c, r in ref["roots"].items()}
    agree = {"spark_edges_equal_pandas": bool(sp_edges.reset_index(drop=True).equals(ref["edges"].reset_index(drop=True))),
             "components_equal": comp == ref["components"],
             "roots_equal": {c: r for c, r in root_by_comp.items()} == ref_roots,
             "revpr_max_abs_diff": float(max(abs(pg["revpr"][m] - ref["revpr"][m]) for m in machines))}
    groups = members_of(comp)
    out = {"name": name, "incidents": int(len(inc)), "machines_with_incidents": int(inc["entity_id"].nunique()),
           "edges": ref["edges"].to_dict(orient="records"), "n_edges": int(len(ref["edges"])),
           "edge_pairs": int(ref["edges"]["weight"].sum()) if len(ref["edges"]) else 0,
           "onsets": ref["onsets"], "components": comp, "group_sizes": sorted((len(g) for g in groups.values()
                                                                                 if len(g) > 1), reverse=True),
           "roots": root_by_comp, "revpr": pg["revpr"], "chains": [{k: v for k, v in ch.items() if k != "incidents"}
                                                                   for ch in ref["chains"] if len(ch["machines"]) > 1],
           "agreement": agree, "pregel_iterations": pg["pregel_iterations"],
           "timings": {"edges_s": edges_s, "cc_s": pg["cc_s"], "roots_s": pg["roots_s"], "revpr_s": pg["revpr_s"]}}
    if truth is not None:
        out["score"] = score_run(truth, inc, comp, root_by_comp, pg["revpr"], ref["chains"])
    else:
        fg = false_groups(comp, set(), ref["chains"], inc, [])
        out["false_groups"] = fg
    return out


def cmd_pregel(args) -> int:
    t0 = time.perf_counter()
    spark = G.build_spark(args.master, ROOT / "checkpoints" / "e8_cascade_graph", app="e8_cascade")
    t_spark = time.perf_counter() - t0
    try:
        runs = []
        for seed in args.seeds:
            d = seed_dir(seed, args.root)
            truth = json.loads((d / "inject" / "cascade_truth.json").read_text(encoding="utf-8"))
            inc = in_scope(G.load_incidents(d / S.PATHS["incidents"] / f"source={SOURCE}"), SCOPE_H)
            r = analyse_graph(spark, f"seed_{seed}", truth["machines"], inc, truth)
            r["seed"], r["truth"] = seed, truth
            runs.append(r)
            s = r["score"]["summary"]
            print(f"seed {seed}: detected {s['detected']}/{s['cascades']}, main recall {s['main']['recall_mean']:.2f} "
                  f"precision {s['main']['precision_mean']}, roots {s['main']['root_correct']}, time-resolved recall "
                  f"{s['time']['recall_mean']:.2f} roots {s['time']['root_correct']}; agreement {r['agreement']}",
                  flush=True)
        base_inc = G.load_incidents(ROOT / BASELINE_INCIDENTS)
        base_machines = runs[0]["truth"]["machines"] if runs else sorted(base_inc["entity_id"].unique())
        base = analyse_graph(spark, "clean_0_72", base_machines, base_inc, None)
        base["hours"] = 72
        print(f"clean baseline: false groups main {base['false_groups']['main']}, time "
              f"{base['false_groups']['time']}; agreement {base['agreement']}", flush=True)
    finally:
        spark.stop()
    doc = {"rules": rules(), "seeds": runs, "baseline": base,
           "over_seeds": over_seeds([r["score"]["summary"] for r in runs]) if runs else None,
           "env": {"graphframes": G.GRAPHFRAMES_PACKAGE, "spark": "3.5.1"},
           "spark_start_s": t_spark, "total_s": time.perf_counter() - t0}
    (ROOT / args.out_dir).mkdir(parents=True, exist_ok=True)
    (ROOT / args.out_dir / PREGEL_JSON).write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(f"pregel part done in {doc['total_s']:.1f} s -> {(args.out_dir / PREGEL_JSON).as_posix()}")
    return 0


def rules() -> dict:
    return {"seeds": list(SEEDS), "machines": MACHINES, "slice_hours": HOURS, "cascades_per_seed": N_CASCADES,
            "dependents": 4, "fault": f"{CASCADE_TYPE}: net_in x 0.1 for 15 min", "dependent_delay_min": [2, 8],
            "cascade_window_hours": list(WINDOW_H), "graph_scope_hours": list(SCOPE_H), "avoid_waves": AVOID,
            "avoid_margin_h": 1.0, "min_gap_h": 2.0, "edge_gap_min": GAP_MIN, "false_group_min_machines": MIN_GROUP,
            "demo": f"seed {DEMO_SEED}, cascade {DEMO_CASCADE}"}


# ---- step 3: Neo4j ------------------------------------------------------------------------------------------------
LABEL, REL = "CascadeMachine", "CASCADE_NEXT"
Q_MEMBERS = (f"MATCH (m:{LABEL} {{id: $machine}})\n"
             f"MATCH (x:{LABEL}) WHERE x.component = m.component AND x.onset IS NOT NULL\n"
             "RETURN x.id AS machine, x.onset_time AS onset ORDER BY x.onset, x.id")
Q_PATHS = (f"MATCH (m:{LABEL} {{id: $machine}})\n"
           f"MATCH (c:{LABEL}) WHERE c.component = m.component\n"
           "WITH m, min(c.onset) AS first\n"
           f"MATCH (r:{LABEL}) WHERE r.component = m.component AND r.onset = first\n"
           f"  AND NOT EXISTS {{ MATCH (:{LABEL})-[:{REL}]->(r) }}\n"
           f"MATCH (x:{LABEL}) WHERE x.component = m.component AND x <> r\n"
           f"OPTIONAL MATCH p = shortestPath((r)-[:{REL}*..15]->(x))\n"
           "RETURN r.id AS root, x.id AS member, [n IN nodes(p) | n.id] AS path, length(p) AS hops\n"
           "ORDER BY root, x.onset, member")
Q_ROOTS = (f"MATCH (c:{LABEL}) WHERE c.onset IS NOT NULL\n"
           "WITH c.component AS comp, min(c.onset) AS first\n"
           f"MATCH (r:{LABEL}) WHERE r.component = comp AND r.onset = first\n"
           f"  AND NOT EXISTS {{ MATCH (:{LABEL})-[:{REL}]->(r) }}\n"
           "RETURN r.id AS root ORDER BY root")


def neo4j_load(session, run: dict) -> dict:
    session.run(f"MATCH (m:{LABEL}) DETACH DELETE m").consume()
    rows = [{"id": m, "onset": run["onsets"].get(m),
             "onset_time": (pd.Timestamp(run["onsets"][m] * 60, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
                            if m in run["onsets"] else None)} for m in sorted(run["components"])]
    session.run(f"UNWIND $rows AS r MERGE (m:{LABEL} {{id: r.id}}) SET m.onset = r.onset, m.onset_time = r.onset_time",
                rows=rows).consume()
    session.run(f"UNWIND $rows AS r MATCH (a:{LABEL} {{id: r.src}}), (b:{LABEL} {{id: r.dst}}) "
                f"MERGE (a)-[e:{REL}]->(b) SET e.weight = r.weight", rows=run["edges"]).consume()
    n = session.run(f"MATCH (m:{LABEL}) RETURN count(m) AS n").single()["n"]
    e = session.run(f"MATCH ()-[r:{REL}]->() RETURN count(r) AS n").single()["n"]
    return {"nodes": int(n), "edges": int(e)}


def neo4j_algorithms(session, name: str) -> dict:
    cfg = {"concurrency": 1}
    out = {}
    for g, orient in ((f"e8c_{name}", "NATURAL"), (f"e8c_{name}_rev", "REVERSE")):
        session.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=g).consume()
        session.run("CALL gds.graph.project($g, $label, $rels) YIELD graphName RETURN graphName", g=g, label=LABEL,
                    rels={REL: {"orientation": orient, "properties": "weight"}}).consume()
    session.run("CALL gds.wcc.write($g, $c) YIELD componentCount RETURN componentCount", g=f"e8c_{name}",
                c={**cfg, "writeProperty": "component"}).consume()
    out["wcc"] = {r["id"]: int(r["v"]) for r in session.run(
        f"MATCH (m:{LABEL}) RETURN m.id AS id, m.component AS v")}
    out["revpr"] = {r["id"]: float(r["v"]) for r in session.run(
        "CALL gds.pageRank.stream($g, $c) YIELD nodeId, score RETURN gds.util.asNode(nodeId).id AS id, score AS v",
        g=f"e8c_{name}_rev", c={**cfg, "relationshipWeightProperty": "weight", "dampingFactor": G.DAMPING,
                                "maxIterations": G.PR_ITERS, "tolerance": 0.0})}
    out["roots"] = [r["root"] for r in session.run(Q_ROOTS)]
    for g in (f"e8c_{name}", f"e8c_{name}_rev"):
        session.run("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=g).consume()
    return out


def demo_machine(run: dict, cascade_id: str = DEMO_CASCADE) -> tuple[dict, str]:
    """The demo cascade and the alarming machine the queries start from: its member that alarmed last (else root)."""
    c = next(x for x in run["score"]["cascades"] if x["cascade_id"] == cascade_id)
    alarmed = [m for m in c["members_alarmed"] if m in run["onsets"]]
    return c, (max(alarmed, key=lambda m: (run["onsets"][m], m)) if alarmed else c["root"])


def neo4j_compare(run: dict, neo: dict) -> dict:
    pg_roots = sorted(m for rs in run["roots"].values() for m in rs)
    return {"components_equal": G.same_partition(neo["wcc"], run["components"]),
            "roots_equal": sorted(neo["roots"]) == pg_roots,
            "revpr_max_abs_diff": float(max(abs(neo["revpr"][m] - run["revpr"][m]) for m in run["revpr"]))}


def neo4j_run(driver, pregel_doc: dict) -> dict:
    out = {"seeds": {}}
    runs = sorted(pregel_doc["seeds"], key=lambda r: r["seed"] == DEMO_SEED)   # demo seed last: stays loaded
    with driver.session() as session:
        from experiments import e8_neo4j as N
        out["env"] = N.versions(session)
        for run in runs:
            t = time.perf_counter()
            loaded = neo4j_load(session, run)
            t_load = time.perf_counter() - t
            t = time.perf_counter()
            neo = neo4j_algorithms(session, f"seed_{run['seed']}")
            out["seeds"][str(run["seed"])] = {"loaded": loaded, "load_s": t_load,
                                              "algorithms_s": time.perf_counter() - t,
                                              "compare": neo4j_compare(run, neo), "roots": neo["roots"]}
            if run["seed"] == DEMO_SEED:
                c, machine = demo_machine(run)
                t = time.perf_counter()
                out["demo"] = {"seed": DEMO_SEED, "cascade": c["cascade_id"], "true_root": c["root"],
                               "members": c["members"], "machine": machine, "queries": [
                                   {"key": "members", "title": f"(a) cascade members of alarming machine {machine}, "
                                                               "ordered by onset", "cypher": Q_MEMBERS,
                                    "rows": [dict(r) for r in session.run(Q_MEMBERS, machine=machine)]},
                                   {"key": "paths", "title": f"(b) path from the root to each member ({machine}'s "
                                                             "component)", "cypher": Q_PATHS,
                                    "rows": [dict(r) for r in session.run(Q_PATHS, machine=machine)]}]}
                out["demo"]["queries_s"] = time.perf_counter() - t
    return out


def cmd_neo4j(args) -> int:
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        print("set NEO4J_PASSWORD (the password of infra/docker-compose.neo4j.yml)", file=sys.stderr)
        return 2
    from neo4j import GraphDatabase
    from experiments import e8_neo4j as N
    doc = json.loads((ROOT / args.out_dir / PREGEL_JSON).read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    with GraphDatabase.driver(os.environ.get("NEO4J_URI", N.URI), auth=(N.USER, password)) as driver:
        driver.verify_connectivity()
        out = neo4j_run(driver, doc)
    out["total_s"] = time.perf_counter() - t0
    (ROOT / args.out_dir / NEO4J_JSON).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for s, r in out["seeds"].items():
        print(f"seed {s}: {r['compare']}")
    print(f"neo4j part done in {out['total_s']:.1f} s -> {(args.out_dir / NEO4J_JSON).as_posix()}")
    return 0


# ---- report (called by e8_graph.py report) ------------------------------------------------------------------------
def load_results(out_dir: Path = OUT_DIR) -> dict | None:
    f = ROOT / out_dir / PREGEL_JSON
    if not f.exists():
        return None
    res = {"pregel": json.loads(f.read_text(encoding="utf-8"))}
    for key, name in (("neo4j", NEO4J_JSON), ("stream", STREAM_JSON)):
        g = ROOT / out_dir / name
        res[key] = json.loads(g.read_text(encoding="utf-8")) if g.exists() else None
    return res


def _mmm(x, fmt="{:.2f}", scale=1.0) -> str:
    if x is None:
        return "-"
    f = (lambda v: fmt.format(v * scale))
    return f"{f(x['mean'])} ({f(x['min'])}-{f(x['max'])})"


def _seed_row(r: dict, v: str) -> str:
    s = r["score"]["summary"]
    m = s[v]
    prec = "-" if m["precision_mean"] is None else f"{m['precision_mean']:.2f}"
    extra = f" | {m['revpr_correct']}/{s['cascades']}" if v == "main" else ""
    return (f"| {r['seed']} | {s['detected']}/{s['cascades']} | {s['members_alarmed']}/{s['members']} | "
            f"{m['recall_mean']:.2f} | {prec} | {m['full_recovery']} | {m['root_correct']}/{s['cascades']} | "
            f"{m['root_in_tie']}{extra} | {m['false_groups']} |")


def section_lines(res: dict, png_name: str) -> list[str]:
    p, neo, st = res["pregel"], res.get("neo4j"), res.get("stream")
    runs, base, o = p["seeds"], p["baseline"], p["over_seeds"]
    L = ["", "## E8c: injected cascades on Alibaba (stream detection, group and root recovery)", "",
         "> **These are injected cascades**, not observed ones: a root fault and four dependent faults placed on random "
         "machines by `src/producer/inject_faults.py --mode cascade`. They test whether the pipeline and the graph "
         "method *can* recover a cascade of this shape; they say nothing about how often real Alibaba faults cascade, "
         "and real dependents need not start 2-8 minutes after their root.", ""]
    if o:
        L += ["**Result.** " + summary_text(p, neo, st), ""]
    L += ["### Rules, fixed before any result was looked at", "",
          f"1. **Injection.** E5b slice (first {MACHINES} machines, hours 0-{HOURS}), seeds "
          f"{', '.join(map(str, SEEDS))}. Per seed {N_CASCADES} cascades: one random root + 4 random dependents, root "
          "fault at T (5-minute aligned), each dependent at T + random 2-8 min, every fault 15 min of net_in x 0.1 "
          "(`break`, which the Alibaba AE catches reliably in E5b; `--cascade-type spike` exists but was not run). "
          f"Cascades start and end in hours {WINDOW_H[0]:g}-{WINDOW_H[1]:g} (the last 30 min of the 24-48 h half are "
          "left out so the stream's 10-minute watermark closes every cascade window before the replay ends), at "
          "least 2 h apart, at least 1 h from the two clean-data alarm waves of E8 (hour 23.5 for 55 min, hour 62.0 "
          "for 50 min), on disjoint machines, never inside a gap of the trace (at least 5 machines with data over the "
          "whole cascade; added when the first planning attempt hit such a gap, before anything was scored). Ground truth: `data/e8/cascade_root/seed_<n>/inject/cascade_truth.json` "
          "(also in `e8_graph.json` under `cascade`).",
          "2. **Stream.** The injected slice is replayed into Kafka `telemetry.alibaba` (`src/producer/replay.py "
          "--speedup 0`); `src/streaming/score_job.py --source alibaba --input kafka --once` reads exactly those "
          "records (starting offsets taken before the replay) and scores them with `models/ae_alibaba`, "
          "is_anomaly = ae_flag. Incidents: `src/incidents/group.py` into `data/e8/cascade_root/seed_<n>`. Hours 0-24 "
          "are the AE's history (recentring, adaptive limit), as in E5b.",
          f"3. **Graph.** {MACHINES} vertices; incidents with onset in hours {SCOPE_H[0]:g}-{SCOPE_H[1]:g}. Directed "
          f"edge u -> v when v's incident starts strictly after u's and no later than {GAP_MIN} min after u's ends; "
          "weight = number of such incident pairs. A machine's onset = its earliest incident onset in that range.",
          "4. **Pregel (GraphFrames).** `connectedComponents` (weak); root = Pregel message passing of the earliest "
          "onset along the edges (both directions), then the vertex whose onset equals the component's earliest and "
          "whose in-degree is 0 (several = a tie, counted apart; none = no root). Reverse PageRank (reversed, "
          "weighted, E8's GDS formula, 30 iterations) as a secondary ranking.",
          "5. **Scoring.** Cascade window W = [T, end of its last fault + 10 min]. *Detected*: a member has an "
          "incident overlapping W. *Recovered component*: of the components holding an alarmed member, the one with "
          "the most members (ties: earliest alarm). *Member recall* = members in it / 5, *precision* = members in it / "
          "its size (undetected: recall 0, precision undefined). *Root correct*: the Pregel root is exactly the true "
          "root. *False groups*: components of >= 3 machines that are no cascade's recovered component; on the clean "
          "hours 0-72 of E8, every component of >= 3 machines.",
          "6. **Secondary, also fixed beforehand: time-resolved components.** Same edges, but components over "
          "incidents (alarm chains) instead of machines, so two unrelated incidents of one machine at different "
          "times cannot join two groups; root = machine of the chain's earliest incident; false groups = chains of "
          ">= 3 machines with no incident overlapping a cascade window.", ""]
    if st:
        L += ["### Stream", "", "| seed | records via Kafka | windows scored | AE flags | same flag as offline "
              "`score_layer` | incidents | score_job s |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for s in st["seeds"]:
            q = s["parity"]
            L.append(f"| {s['seed']} | {s['kafka_records']} | {q['stream_windows']} | {q['stream_flags']} | "
                     f"{q['same_flag']}/{q['common_windows']} | {s['incidents']} | {s['score_job_s']:.0f} |")
        L += ["", "The offline column re-scores the same injected raw layer with `autoencoder.score_layer` (the E5b "
              "path) as a check that the streaming job gives the same AE flags.", ""]
    L += ["### Per seed (main rule: machine components)", "",
          "| seed | detected | members alarmed | member recall | precision | exact group | root correct | root in a tie "
          "| reverse PR top = root | false groups |",
          "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    L += [_seed_row(r, "main") for r in runs]
    L += ["", "### Per seed (secondary: time-resolved components)", "",
          "| seed | detected | members alarmed | member recall | precision | exact group | root correct | root in a tie "
          "| false groups |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    L += [_seed_row(r, "time") for r in runs]
    if o:
        n = runs[0]["score"]["summary"]["cascades"]
        L += ["", "### Mean (min-max) over seeds", "",
              "| | main (machine components) | time-resolved |", "| --- | ---: | ---: |",
              f"| cascades detected (of {n}) | {_mmm(o['detected'], '{:.1f}')} | same |",
              f"| members alarmed (of {n * 5}) | {_mmm(o['members_alarmed'], '{:.1f}')} | same |",
              f"| root fault detection delay, min | {_mmm(o['root_delay_min_mean'], '{:.1f}')} | same |",
              f"| member recall | {_mmm(o['main']['recall_mean'])} | {_mmm(o['time']['recall_mean'])} |",
              f"| precision | {_mmm(o['main']['precision_mean'])} | {_mmm(o['time']['precision_mean'])} |",
              f"| exact group (recall = precision = 1) | {_mmm(o['main']['full_recovery'], '{:.1f}')} | "
              f"{_mmm(o['time']['full_recovery'], '{:.1f}')} |",
              f"| root correct (of {n}) | {_mmm(o['main']['root_correct'], '{:.1f}')} | "
              f"{_mmm(o['time']['root_correct'], '{:.1f}')} |",
              f"| root in a tie | {_mmm(o['main']['root_in_tie'], '{:.1f}')} | "
              f"{_mmm(o['time']['root_in_tie'], '{:.1f}')} |",
              f"| reverse PageRank top = root | {_mmm(o['main']['revpr_correct'], '{:.1f}')} | - |",
              f"| false groups (>= {MIN_GROUP} machines) per 24 h | {_mmm(o['main']['false_groups'], '{:.1f}')} | "
              f"{_mmm(o['time']['false_groups'], '{:.1f}')} |",
              f"| false groups on clean hours 0-72 (per 24 h) | {base['false_groups']['main']} "
              f"({base['false_groups']['main'] / 3:.1f}) | {base['false_groups']['time']} "
              f"({base['false_groups']['time'] / 3:.1f}) |"]
    L += ["", "### Every cascade", "",
          "| seed | cascade | start (h) | root | alarmed | main: component (roots) | time-resolved: component (roots) |",
          "| ---: | --- | ---: | --- | --- | --- | --- |"]
    for r in runs:
        for c in r["score"]["cascades"]:
            def cell(v):
                x = c[v]
                if not x["component"]:
                    return "not detected"
                return (f"{len(x['component'])} machines, {round(x['recall'] * 5)} members "
                        f"(root {', '.join(x['roots']) or 'none'}{' ✓' if x['root_correct'] else ''})")
            L.append(f"| {r['seed']} | {c['cascade_id']} | {c['start_h']:.2f} | {c['root']} | "
                     f"{len(c['members_alarmed'])}/5 | {cell('main')} | {cell('time')} |")
    ro = {k: sum(r["score"]["summary"]["root_onset"][k] for r in runs) for k in ("first", "tied", "later")}
    tied = sorted(o for r in runs for o in r["score"]["summary"]["tied_offsets_min"])
    n_all = sum(r["score"]["summary"]["cascades"] for r in runs)
    L += ["", f"**Why the root is rarely named (diagnostic, looked at after the results, not a rule).** Within the "
          f"cascade window the true root's first alarm came strictly before every dependent's in {ro['first']} of "
          f"{n_all} cascades, tied with at least one dependent in {ro['tied']} and came later in {ro['later']}. Every "
          f"tied dependent had started {min(tied) if tied else '-'}-{max(tied) if tied else '-'} min after the root, "
          "so its fault fell in the root's 5-minute window and both incidents start at the same window: with "
          "window-level onsets, no rule can order them. On the main (machine) graph the cascades are moreover joined "
          "to background alarms elsewhere in the 24 hours through machines that alarm twice, so the component's "
          "earliest machine is usually not in the cascade at all."]
    L += ["", "### Pregel (GraphFrames) vs pandas vs Neo4j (GDS)", "",
          "| graph | edges | Spark self-join = pandas | components = pandas | roots = pandas | reverse PR max diff "
          "| Neo4j WCC = Pregel | Neo4j roots (Cypher) = Pregel | Neo4j reverse PR max diff |",
          "| --- | ---: | --- | --- | --- | ---: | --- | --- | ---: |"]
    for r in runs + [base]:
        a = r["agreement"]
        nc = (neo or {}).get("seeds", {}).get(str(r.get("seed")), {}).get("compare") if r.get("seed") else None
        L.append(f"| {r['name']} | {r['n_edges']} | {a['spark_edges_equal_pandas']} | {a['components_equal']} | "
                 f"{a['roots_equal']} | {a['revpr_max_abs_diff']:.1e} | "
                 + (f"{nc['components_equal']} | {nc['roots_equal']} | {nc['revpr_max_abs_diff']:.1e} |" if nc
                    else "- | - | - |"))
    if neo and neo.get("demo"):
        dm = neo["demo"]
        L += ["", f"### Neo4j operator queries (seed {dm['seed']}, cascade {dm['cascade']}: true root {dm['true_root']}, "
              f"members {', '.join(dm['members'])})", ""]
        for q in dm["queries"]:
            L += [f"**{q['title']}**", "", "```cypher", q["cypher"], "```", ""]
            if q["rows"]:
                cols = list(q["rows"][0])
                L += ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
                for row in q["rows"]:
                    L.append("| " + " | ".join(" -> ".join(v) if isinstance(v, list) else ("-" if v is None else str(v))
                                               for v in row.values()) + " |")
            else:
                L.append("(no rows)")
            L.append("")
    L += [f"![E8c cascade]({png_name})", "",
          f"Left: timeline of the demo cascade (seed {DEMO_SEED}, {DEMO_CASCADE}): shaded = injected fault, bar = "
          "incident, every machine with an incident near it. Right: its recovered machine component with the directed "
          "edges (label = incident pairs).", "", "### E8c runtime", ""]
    if st:
        L.append(f"- Stream part (inject + Kafka replay + score_job + offline check + incidents, 3 seeds): "
                 f"{st['total_s']:.0f} s; score_job {sum(s['score_job_s'] for s in st['seeds']):.0f} s of it.")
    L.append(f"- Pregel part (3 seeds + clean baseline, local[*]): {p['total_s']:.0f} s (Spark start "
             f"{p['spark_start_s']:.0f} s).")
    if neo:
        L.append(f"- Neo4j part: {neo['total_s']:.1f} s.")
    L += ["", "### E8c caveats", "",
          "- Injected cascades with a fixed shape (5 machines, 2-8 min delays, 15-min faults, one fault type); real "
          "cascades vary in size, delay and symptom, and no labelled Alibaba cascade exists to check against.",
          "- Onsets have the stream's 5-minute window resolution: a dependent starting 2-4 min after its root is often "
          "flagged in the root's window, so both get the same onset; the strict 'starts after' rule then gives no edge "
          "between them and the root can tie with that dependent.",
          "- The main rule builds one machine graph over 24 hours, so a machine's unrelated alarm at another time can "
          "join a cascade to background alarms (lower precision, wrong onset); the time-resolved variant avoids this "
          "but is not the graph the Pregel/Neo4j jobs run.",
          "- A low false-group count on the machine graph is not good news by itself: background alarm groups that "
          "touch a cascade member are merged into that cascade's component (counted in its precision, not as false "
          "groups). The time-resolved count is the cleaner false-group measure.",
          "- The clean baseline is E8's offline-scored hours 0-72 (no injection), not a stream run; its false groups "
          "include the two real alarm waves, which are unlabelled and may be real cluster events."]
    return L


def summary_text(p: dict, neo: dict | None, st: dict | None) -> str:
    o, runs = p["over_seeds"], p["seeds"]
    n = runs[0]["score"]["summary"]["cascades"]
    agree = all(r["agreement"]["components_equal"] and r["agreement"]["roots_equal"] for r in runs + [p["baseline"]])
    neo_ok = neo and all(v["compare"]["components_equal"] and v["compare"]["roots_equal"]
                         for v in neo["seeds"].values())
    return (f"Over 3 seeds x {n} injected cascades, the stream detected {o['detected']['mean']:.1f} of {n} per seed "
            f"(members alarmed {o['members_alarmed']['mean']:.1f} of {n * 5}). The machine graph recovered the group "
            f"with member recall {o['main']['recall_mean']['mean']:.2f} and precision "
            f"{_mmm(o['main']['precision_mean']).split(' ')[0]}, and named the true root in "
            f"{o['main']['root_correct']['mean']:.1f} of {n} per seed ({o['main']['root_in_tie']['mean']:.1f} more in "
            f"a tie); reverse PageRank put the root on top in {o['main']['revpr_correct']['mean']:.1f}. The "
            f"time-resolved variant: recall {o['time']['recall_mean']['mean']:.2f}, precision "
            f"{_mmm(o['time']['precision_mean']).split(' ')[0]}, root correct {o['time']['root_correct']['mean']:.1f} "
            f"of {n}. False groups of >= {MIN_GROUP} machines with injection: "
            f"{o['main']['false_groups']['mean']:.1f} per 24 h on the machine graph (background alarms are absorbed "
            f"into the cascades' component and show up as the low precision instead) and "
            f"{o['time']['false_groups']['mean']:.1f} time-resolved; on the clean hours 0-72 "
            f"{p['baseline']['false_groups']['main'] / 3:.1f} and {p['baseline']['false_groups']['time'] / 3:.1f} "
            "per 24 h. "
            f"Pregel = pandas reference on components and roots: {agree}; Neo4j (GDS WCC, Cypher roots) = Pregel: "
            f"{bool(neo_ok) if neo else 'not run'}.")


def plot(path: Path, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    p = res["pregel"]
    run = next(r for r in p["seeds"] if r["seed"] == DEMO_SEED)
    truth = next(c for c in run["truth"]["cascades"] if c["cascade_id"] == DEMO_CASCADE)
    sc = next(c for c in run["score"]["cascades"] if c["cascade_id"] == DEMO_CASCADE)
    inc = G.load_incidents(seed_dir(DEMO_SEED) / S.PATHS["incidents"] / f"source={SOURCE}")
    w0, w1 = cascade_window(truth)
    lo, hi = w0 - 45, w1 + 45
    near = inc[(inc["start"] < hi) & (inc["end"] > lo)]
    machines = list(truth["members"]) + sorted(set(near["entity_id"]) - set(truth["members"]))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    for k, m in enumerate(machines):
        y = len(machines) - 1 - k
        f = next((f for f in truth["faults"] if f["entity_id"] == m), None)
        if f:
            a, b = minutes_of(f["start_time"]) - w0, minutes_of(f["end_time"]) - w0
            ax.axvspan(a, b, ymin=(y + 0.1) / len(machines), ymax=(y + 0.9) / len(machines),
                       color="#f2c14e", alpha=0.5, lw=0)
        for r in near[near["entity_id"] == m].itertuples():
            ax.plot([r.start - w0, r.end - w0], [y + 0.5] * 2, lw=6, solid_capstyle="butt",
                    color="#c0392b" if m in truth["members"] else "#5b6b8c")
    ax.set_yticks([len(machines) - 1 - k + 0.5 for k in range(len(machines))])
    ax.set_yticklabels([f"{m} ({'root' if m == truth['root'] else 'member' if m in truth['members'] else 'other'})"
                        for m in machines], fontsize=8)
    ax.set_ylim(0, len(machines))
    ax.set_xlim(lo - w0, hi - w0)
    ax.set_xlabel(f"minutes after the root fault (T = hour {hours_of(w0):.2f} of the trace)")
    ax.set_title(f"seed {DEMO_SEED}, cascade {DEMO_CASCADE}: injected faults (shaded) and stream incidents (bars)",
                 fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    comp = sc["main"]["component"]
    bx.axis("off")
    if comp:
        g = nx.DiGraph()
        g.add_nodes_from(comp)
        for e in run["edges"]:
            if e["src"] in comp and e["dst"] in comp:
                g.add_edge(e["src"], e["dst"], weight=e["weight"])
        pos = nx.spring_layout(g, seed=3, k=1.2) if len(comp) > 1 else {comp[0]: np.array([0.0, 0.0])}
        colors = ["#c0392b" if m == truth["root"] else "#f2c14e" if m in truth["members"] else "#c8d1e0" for m in comp]
        nx.draw_networkx_nodes(g, pos, ax=bx, node_color=colors, node_size=900, edgecolors="black", linewidths=0.8)
        nx.draw_networkx_labels(g, pos, ax=bx, font_size=8)
        nx.draw_networkx_edges(g, pos, ax=bx, arrows=True, arrowsize=14, node_size=900, edge_color="#444444")
        nx.draw_networkx_edge_labels(g, pos, ax=bx, font_size=7, edge_labels={(u, v): a["weight"]
                                                                             for u, v, a in g.edges(data=True)})
        bx.set_title(f"recovered component: {len(comp)} machines, recall {sc['main']['recall']:.1f}, precision "
                     f"{sc['main']['precision']:.2f}\nPregel root(s) {', '.join(sc['main']['roots']) or 'none'} "
                     f"(true root {truth['root']}); edge label = incident pairs", fontsize=10)
        from matplotlib.patches import Patch
        bx.legend(handles=[Patch(color="#c0392b", label="true root"), Patch(color="#f2c14e", label="true member"),
                           Patch(color="#c8d1e0", label="not in the cascade")], loc="lower center", ncol=3,
                  fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    else:
        bx.text(0.5, 0.5, "cascade not detected", ha="center", va="center", transform=bx.transAxes)
    fig.suptitle("E8c: an injected cascade (root + 4 dependents, net_in x 0.1) through Kafka + score_job, and the "
                 "recovered co-failure component", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---- CLI ----------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stream", "pregel", "neo4j"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
        p.add_argument("--root", type=Path, default=CASCADE_ROOT)
        p.add_argument("--out-dir", type=Path, default=OUT_DIR)
        if name == "pregel":
            p.add_argument("--master", default=None)
    args = ap.parse_args(argv)
    return {"stream": cmd_stream, "pregel": cmd_pregel, "neo4j": cmd_neo4j}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
