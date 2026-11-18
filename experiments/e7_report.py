"""E7 helper (called by experiments/e7_hdfs.sh): HDFS as the Spark checkpoint store.

Sub-commands
    event    --file F --name N [key=value ...]     append one timestamped event (JSON line) to F
    batches  --progress FILE [--since-ms MS]       number of micro-batches with input rows in score_job's progress log
    sample   FILE --field live|under|missing|corrupt|status
                                                   one value of a saved 'dfsadmin -report -live' + 'fsck' sample
    compare  --clean DIR --fault DIR --out F       clean vs fault run: Parquet scores and anomalies topic, with the
                                                   definitions of experiments/e3_compare.py; exit 0 = identical
    report   --out-dir D [--baseline F] [--expected-datanodes N] [--md F] [--json F]
                                                   docs/results/e7_hdfs.md + .json from everything under D

Layout of D (written by e7_hdfs.sh)
    runs/<run_id>.json      drain_test.sh results; run ids end in _clean, _fault, _hdfs_rep<k>, _nfs_rep<k>
    clean/, fault/          scores/ (copy of data/scores_stream/source=smd) + anomalies.jsonl (topic dump)
    fault/samples/*.txt     '# e7-sample ts_ms=<ms> phase=<before|outage|restarted>' + dfsadmin output, then a line
                            '==== fsck' + 'hdfs fsck <checkpoint>' output
    compare.json            output of 'compare'; events.jsonl  output of 'event'
PASS = (a) drained, (b) drained AND scores/topic identical to (a) AND fsck never reported a missing or corrupt
block AND reported under-replicated blocks while the DataNode was down AND was HEALTHY with 0 under-replicated
blocks and all DataNodes live after the restart. (c) is a measurement, not a pass/fail criterion.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "infra" / "cluster"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments import e3_compare as E3  # noqa: E402

BASELINE = ROOT / "docs" / "results" / "e2_cluster_5w.json"
OUT_MD = ROOT / "docs" / "results" / "e7_hdfs.md"
OUT_JSON = ROOT / "docs" / "results" / "e7_hdfs.json"
METRICS = ("records_per_s", "steady_records_per_s", "drain_s", "batch_mean_s", "batch_p95_s", "first_batch_s")
CAVEATS = [
    "The NameNode is still a single point of failure: HDFS removes the single NFS server from the DataNode side "
    "(blocks are replicated twice over three VMs), but every checkpoint read and write needs the one NameNode on "
    "spark-master. An HA NameNode (two NameNodes, three JournalNodes, ZooKeeper failover) is future work; the "
    "cluster has no memory left for it on the 8 GB workers.",
    "spark-master is also the Spark master, the driver host, the Kafka broker and the NFS server, so losing that VM "
    "still stops the pipeline; E7 tests the loss of a storage node, not of the master.",
    "Only a DataNode process was stopped (systemctl stop), not a VM or a disk, and only one of three; with "
    "replication 2 a second simultaneous DataNode loss can make blocks missing.",
    "The NameNode counts a stopped DataNode's replicas as live until it declares the node dead (2 x recheck "
    "interval + 10 heartbeats = 90 s here; Hadoop's default is 10.5 min), so under-replication appears in fsck only "
    "from then on, and the NameNode then re-replicates onto the two remaining DataNodes.",
    "The NFS throughput baseline is the E2 sweep (docs/results/e2_cluster_5w.json) from an earlier day; since then "
    "Cassandra (E6) and the HDFS daemons run on spark-w2, spark-w5 and spark-master.",
]


# ---- small pieces -------------------------------------------------------------------------------------
def append_event(path: Path, name: str, pairs: list[str], now_ms: int | None = None) -> dict:
    ev = {"name": name, "ts_ms": int(time.time() * 1000) if now_ms is None else now_ms}
    for kv in pairs:
        k, sep, v = kv.partition("=")
        if not sep:
            raise ValueError(f"event field '{kv}' is not key=value")
        ev[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")
    return ev


def read_events(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def batches_with_rows(progress: Path, since_ms: int = 0) -> int:
    from drain_monitor import read_progress
    return sum(1 for b in read_progress(Path(progress), since_ms) if int(b.get("numInputRows", 0)) > 0)


def _first_int(text: str, label: str) -> int | None:
    m = re.search(rf"^\s*{re.escape(label)}:\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def parse_fsck(text: str) -> dict:
    """Counts of an 'hdfs fsck' report (the first, replicated-blocks section)."""
    status = re.search(r"^Status:\s*(\w+)", text, re.MULTILINE)
    return {"status": status.group(1) if status else None,
            "total_blocks": _first_int(text, "Total blocks (validated)"),
            "under_replicated": _first_int(text, "Under-replicated blocks"),
            "missing": _first_int(text, "Missing blocks"),
            "corrupt": _first_int(text, "Corrupt blocks"),
            "missing_replicas": _first_int(text, "Missing replicas"),
            "datanodes": _first_int(text, "Number of data-nodes")}


def parse_live(text: str) -> int | None:
    m = re.search(r"^Live datanodes \((\d+)\)", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def parse_sample(text: str) -> dict:
    head = re.search(r"^# e7-sample ts_ms=(\d+) phase=(\S+)", text, re.MULTILINE)
    report, _, fsck = text.partition("==== fsck")
    return {"ts_ms": int(head.group(1)) if head else None, "phase": head.group(2) if head else None,
            "live": parse_live(report), **parse_fsck(fsck)}


def load_samples(directory: Path) -> list[dict]:
    rows = [parse_sample(f.read_text(encoding="utf-8")) for f in sorted(Path(directory).glob("*.txt"))]
    return sorted((r for r in rows if r["ts_ms"] is not None), key=lambda r: r["ts_ms"])


def sample_field(path: Path, field: str) -> str:
    s = parse_sample(Path(path).read_text(encoding="utf-8"))
    key = {"under": "under_replicated"}.get(field, field)
    v = s.get(key)
    return "" if v is None else str(v)


# ---- compare --------------------------------------------------------------------------------------------
def compare_runs(clean: Path, fault: Path) -> dict:
    cs, fs = E3.load_scores(Path(clean) / "scores"), E3.load_scores(Path(fault) / "scores")
    ct, ft = E3.topic_rows(Path(clean) / "anomalies.jsonl"), E3.topic_rows(Path(fault) / "anomalies.jsonl")
    parquet, topic = E3.compare(cs, fs), E3.compare(ct, ft)
    return {"parquet": parquet, "topic": topic,
            "clean_anomalies": sum(1 for r in cs if r.get("is_anomaly")),
            "fault_anomalies": sum(1 for r in fs if r.get("is_anomaly")),
            "pass": E3.verdict(parquet, topic)}


# ---- report ---------------------------------------------------------------------------------------------
RUN_KINDS = (("clean", re.compile(r"_clean$")), ("fault", re.compile(r"_fault$")),
             ("hdfs", re.compile(r"_hdfs_rep\d+$")), ("nfs", re.compile(r"_nfs_rep\d+$")))


def load_runs(run_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {k: [] for k, _ in RUN_KINDS}
    for f in sorted(Path(run_dir).glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rid = str(r.get("run_id", f.stem))
        for kind, rx in RUN_KINDS:
            if rx.search(rid):
                out[kind].append(r)
    return out


def drained(run: dict | None) -> bool:
    return bool(run) and run.get("status") == "drained" and run.get("consumed") == run.get("records")


def stats(runs: list[dict]) -> dict:
    ok = [r for r in runs if r.get("status") == "drained"]
    out: dict = {"runs": len(runs), "drained": len(ok), "run_ids": [r.get("run_id") for r in runs]}
    for m in METRICS:
        vals = [float(r[m]) for r in ok if isinstance(r.get(m), (int, float))]
        out[m] = ({"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)} if vals else None)
    return out


def baseline_runs(path: Path, executors: int = 5, cores: int = 2) -> tuple[list[dict], dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = [r for r in d.get("runs", []) if r.get("executors") == executors and r.get("cores") == cores]
    return runs, d.get("meta", {})


def fault_analysis(samples: list[dict], events: list[dict], expected_dn: int) -> dict:
    ev = {e["name"]: e for e in events}
    stop = ev.get("datanode_stopped", {}).get("ts_ms")
    restart = ev.get("datanode_started", {}).get("ts_ms")
    outage = [s for s in samples if s["phase"] == "outage"]
    after = [s for s in samples if s["phase"] == "restarted"]
    rel = (lambda t: None if t is None or stop is None else round((t - stop) / 1000, 1))
    dead = next((s for s in outage if s["live"] is not None and s["live"] < expected_dn), None)
    first_under = next((s for s in outage if (s["under_replicated"] or 0) > 0), None)
    final = samples[-1] if samples else None
    healed = next((s for s in after if s["live"] == expected_dn and s["under_replicated"] == 0
                   and s["missing"] == 0 and s["status"] == "HEALTHY"), None)
    max_under = max((s["under_replicated"] or 0 for s in outage), default=0)
    never_missing = all((s["missing"] or 0) == 0 and (s["corrupt"] or 0) == 0 for s in samples)
    checks = {
        "fsck never reported a missing or corrupt block": bool(samples) and never_missing,
        "fsck reported under-replicated blocks while the DataNode was down": max_under > 0,
        "after the restart: all DataNodes live, 0 under-replicated, HEALTHY": healed is not None,
    }
    return {"stop_host": ev.get("datanode_stopped", {}).get("host"),
            "stopped_after_batches": ev.get("datanode_stopped", {}).get("after_batches"),
            "samples": len(samples), "max_under_replicated": max_under,
            "max_missing": max((s["missing"] or 0 for s in samples), default=0),
            "blocks_at_max": next((s["total_blocks"] for s in outage if (s["under_replicated"] or 0) == max_under
                                   and max_under), None),
            "dead_detected_after_s": rel(dead["ts_ms"]) if dead else None,
            "first_under_replicated_after_s": rel(first_under["ts_ms"]) if first_under else None,
            "restart_after_s": rel(restart),
            "under_replicated_at_restart": next((s["under_replicated"] for s in reversed(outage)), None),
            "healed_after_restart_s": (round((healed["ts_ms"] - restart) / 1000, 1) if healed and restart else None),
            "final": final, "checks": checks,
            "timeline": [{"t_s": rel(s["ts_ms"]), "phase": s["phase"], "live": s["live"], "status": s["status"],
                          "blocks": s["total_blocks"], "under": s["under_replicated"], "missing": s["missing"]}
                         for s in samples]}


def run_summary(run: dict | None) -> dict | None:
    if not run:
        return None
    keys = ("run_id", "status", "records", "consumed", "batches", "drain_s", "records_per_s",
            "steady_records_per_s", "batch_mean_s", "executors", "cores", "checkpoint_dir")
    return {k: run.get(k) for k in keys}


def build_report(out_dir: Path, baseline: Path | None, expected_dn: int = 3) -> dict:
    out_dir = Path(out_dir)
    runs = load_runs(out_dir / "runs")
    events = read_events(out_dir / "events.jsonl")
    comp_path = out_dir / "compare.json"
    comp = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else None
    clean = runs["clean"][-1] if runs["clean"] else None
    fault = runs["fault"][-1] if runs["fault"] else None
    rep: dict = {"meta": {"date": time.strftime("%Y-%m-%d", time.gmtime()), "out_dir": out_dir.name,
                          "expected_datanodes": expected_dn},
                 "a_clean": None, "b_fault": None, "c_throughput": None, "caveats": CAVEATS}
    if clean:
        rep["a_clean"] = {"run": run_summary(clean), "drained": drained(clean),
                          "score_rows": comp["parquet"]["clean_rows"] if comp else None,
                          "anomalies": comp.get("clean_anomalies") if comp else None,
                          "anomalies_topic_distinct": comp["topic"]["clean_distinct"] if comp else None}
    if fault:
        fa = fault_analysis(load_samples(out_dir / "fault" / "samples"), events, expected_dn)
        fa["checks"] = {"fault run drained": drained(fault),
                        "scores and anomalies identical to (a)": bool(comp and comp.get("pass")), **fa["checks"]}
        rep["b_fault"] = {"run": run_summary(fault), "compare": comp, **fa,
                          "pass": all(fa["checks"].values())}
    if runs["hdfs"] or runs["nfs"]:
        c: dict = {"hdfs": stats(runs["hdfs"])}
        if runs["nfs"]:
            c["nfs_rerun"] = stats(runs["nfs"])
        if baseline and Path(baseline).exists():
            b_runs, b_meta = baseline_runs(baseline)
            c["nfs_baseline"] = {**stats(b_runs), "source": Path(baseline).name, "date": b_meta.get("date"),
                                 "records": b_meta.get("records"), "replicate": b_meta.get("replicate")}
        for ref in ("nfs_baseline", "nfs_rerun"):
            h, n = c["hdfs"].get("records_per_s"), (c.get(ref) or {}).get("records_per_s")
            if h and n:
                c[f"hdfs_vs_{ref}"] = round(h["mean"] / n["mean"], 3)
        rep["c_throughput"] = c
    verdicts = ([rep["a_clean"]["drained"]] if clean else []) + ([rep["b_fault"]["pass"]] if fault else [])
    # (c) alone is a measurement: it "passes" when at least one HDFS drain completed
    rep["pass"] = all(verdicts) if verdicts else bool(rep["c_throughput"] and rep["c_throughput"]["hdfs"]["drained"])
    return rep


def thin_timeline(tl: list[dict], limit: int = 30) -> list[dict]:
    """Samples where something changed (phase, live nodes, status, under-replicated, missing), first and last
    always; if still more than limit, every k-th of those (first and last kept)."""
    sig = ("phase", "live", "status", "under", "missing")
    keep = [x for i, x in enumerate(tl) if i in (0, len(tl) - 1) or any(x[k] != tl[i - 1][k] for k in sig)]
    if len(keep) > limit:
        k = -(-len(keep) // limit)
        keep = [x for i, x in enumerate(keep) if i % k == 0 or i == len(keep) - 1]
    return keep


def _f(v, spec=",.0f", none="–") -> str:
    return none if v is None else format(v, spec)


def _rng(s: dict | None, m: str, spec: str) -> str:
    x = (s or {}).get(m)
    return "–" if not x else f"{_f(x['mean'], spec)} ({_f(x['min'], spec)}–{_f(x['max'], spec)})"


def render_md(rep: dict) -> str:
    L = ["# E7: HDFS as the checkpoint store", "",
         "Spark Structured Streaming keeps its offsets, commits and state store in the checkpoint directory. Until "
         "now that was `checkpoints/` on the NFS share served by spark-master: one server, one disk. E7 puts the "
         "checkpoint on HDFS instead (Apache Hadoop 3.3.6: NameNode on spark-master, DataNodes on spark-master, "
         "spark-w2 and spark-w5, `dfs.replication = 2`; `deploy.sh hdfs`) with "
         "`submit.sh / drain_test.sh -k hdfs://spark-master:9000/...`. Without `-k` the pipeline is unchanged (NFS).",
         "", f"Measured {rep['meta']['date']} with `experiments/e7_hdfs.sh` (5 executors × 2 cores, "
         "`maxOffsetsPerTrigger` 200,000, 32 topic partitions).", ""]
    a, b, c = rep["a_clean"], rep["b_fault"], rep["c_throughput"]
    if a:
        r = a["run"]
        L += ["## (a) Clean drain, HDFS checkpoints", "",
              f"- {_f(r['consumed'])} of {_f(r['records'])} records consumed in {r['batches']} micro-batches, "
              f"drain {_f(r['drain_s'], '.1f')} s ({_f(r['records_per_s'])} records/s); status `{r['status']}`.",
              f"- score rows written: {_f(a['score_rows'])}; anomalies (`is_anomaly`): {_f(a['anomalies'])}; "
              f"distinct anomalies on the `anomalies` topic: {_f(a['anomalies_topic_distinct'])}.",
              f"- checkpoint `{r['checkpoint_dir']}`.", ""]
    if b:
        r, comp = b["run"], b.get("compare") or {}
        p, t = comp.get("parquet", {}), comp.get("topic", {})
        L += ["## (b) Same drain, DataNode on " + str(b["stop_host"]) + " stopped", "",
              f"After micro-batch {b['stopped_after_batches']} had committed, `systemctl stop hadoop-datanode` on "
              f"{b['stop_host']}; it stayed down until the drain had finished and fsck had been sampled.", "",
              f"- drain: {_f(r['consumed'])} of {_f(r['records'])} records, {r['batches']} batches, "
              f"{_f(r['drain_s'], '.1f')} s ({_f(r['records_per_s'])} records/s); status `{r['status']}`.",
              f"- vs (a): Parquet rows {_f(p.get('clean_rows'))} / {_f(p.get('fault_rows'))}, lost {p.get('lost')}, "
              f"extra {p.get('extra')}, duplicates {p.get('fault_duplicates')}, value mismatches {p.get('mismatch')}; "
              f"anomalies {_f(comp.get('clean_anomalies'))} / {_f(comp.get('fault_anomalies'))}; topic (de-duplicated) "
              f"lost {t.get('lost')}, extra {t.get('extra')}, mismatches {t.get('mismatch')}.",
              f"- NameNode declared the DataNode dead {_f(b['dead_detected_after_s'], '.1f')} s after the stop; "
              f"fsck first showed under-replicated blocks at {_f(b['first_under_replicated_after_s'], '.1f')} s, "
              f"at most {b['max_under_replicated']} (of {_f(b['blocks_at_max'])} blocks), missing blocks at most "
              f"{b['max_missing']}.",
              f"- DataNode restarted at {_f(b['restart_after_s'], '.1f')} s (under-replicated then: "
              f"{_f(b['under_replicated_at_restart'])}); fsck HEALTHY with 0 under-replicated and all DataNodes live "
              f"{_f(b['healed_after_restart_s'], '.1f')} s after the restart.", "",
              "| t after stop (s) | phase | live DataNodes | fsck status | blocks | under-replicated | missing |",
              "| ---: | --- | ---: | --- | ---: | ---: | ---: |"]
        keep = thin_timeline(b["timeline"])
        L += [f"| {_f(x['t_s'], '.1f')} | {x['phase']} | {_f(x['live'], 'd')} | {x['status'] or '–'} | "
              f"{_f(x['blocks'], 'd')} | {_f(x['under'], 'd')} | {_f(x['missing'], 'd')} |" for x in keep]
        L += [""] + [f"- [{'x' if v else ' '}] {k}" for k, v in b["checks"].items()]
        L += ["", f"**(b): {'PASS' if b['pass'] else 'FAIL'}**", ""]
    if c:
        L += ["## (c) Throughput: HDFS vs NFS checkpoints (5 × 2, 2,000,000-record backlog, replicate 4)", "",
              "| checkpoint store | runs drained | records/s | steady records/s | drain (s) | batch mean (s) | first batch (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key, name in (("hdfs", "HDFS (this run)"), ("nfs_rerun", "NFS (re-run today)"),
                          ("nfs_baseline", "NFS (E2 baseline, e2_cluster_5w.json)")):
            s = c.get(key)
            if s:
                L.append(f"| {name} | {s['drained']}/{s['runs']} | {_rng(s, 'records_per_s', ',.0f')} | "
                         f"{_rng(s, 'steady_records_per_s', ',.0f')} | {_rng(s, 'drain_s', '.1f')} | "
                         f"{_rng(s, 'batch_mean_s', '.2f')} | {_rng(s, 'first_batch_s', '.1f')} |")
        L.append("")
        for ref, name in (("nfs_baseline", "the E2 NFS baseline"), ("nfs_rerun", "today's NFS re-run")):
            if f"hdfs_vs_{ref}" in c:
                L.append(f"- HDFS records/s = {c[f'hdfs_vs_{ref}']:.3f} × {name} (mean of means).")
        L.append("")
    L += ["## Caveats", ""] + [f"- {x}" for x in rep["caveats"]]
    L += ["", f"**Result: {'PASS' if rep['pass'] else 'FAIL'}** ((a) drained; (b) identical output, no block "
          "missing, under-replication seen and healed)"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("event")
    e.add_argument("--file", type=Path, required=True), e.add_argument("--name", required=True)
    e.add_argument("fields", nargs="*")
    bt = sub.add_parser("batches")
    bt.add_argument("--progress", type=Path, required=True), bt.add_argument("--since-ms", type=int, default=0)
    sm = sub.add_parser("sample")
    sm.add_argument("file", type=Path)
    sm.add_argument("--field", required=True, choices=("live", "under", "missing", "corrupt", "status", "total_blocks"))
    cp = sub.add_parser("compare")
    cp.add_argument("--clean", type=Path, required=True), cp.add_argument("--fault", type=Path, required=True)
    cp.add_argument("--out", type=Path, required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--out-dir", type=Path, required=True)
    rp.add_argument("--baseline", type=Path, default=BASELINE)
    rp.add_argument("--expected-datanodes", type=int, default=3)
    rp.add_argument("--md", type=Path, default=OUT_MD), rp.add_argument("--json", type=Path, default=OUT_JSON)
    a = ap.parse_args(argv)
    if a.cmd == "event":
        append_event(a.file, a.name, a.fields)
        return 0
    if a.cmd == "batches":
        print(batches_with_rows(a.progress, a.since_ms) if a.progress.exists() else 0)
        return 0
    if a.cmd == "sample":
        print(sample_field(a.file, a.field))
        return 0
    if a.cmd == "compare":
        res = compare_runs(a.clean, a.fault)
        a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        p = res["parquet"]
        print(f"compare: Parquet {p['clean_rows']} / {p['fault_rows']} rows, lost {p['lost']}, extra {p['extra']}, "
              f"duplicates {p['fault_duplicates']}, mismatch {p['mismatch']}; anomalies {res['clean_anomalies']} / "
              f"{res['fault_anomalies']}; {'PASS' if res['pass'] else 'FAIL'}")
        return 0 if res["pass"] else 1
    rep = build_report(a.out_dir, a.baseline, a.expected_datanodes)
    text = render_md(rep)
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.md.write_text(text, encoding="utf-8")
    a.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(text)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
