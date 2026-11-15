"""E3 helper (called by experiments/e3_faults.sh): compare a fault run with a clean run of score_job.

Sub-commands
    dump-topic  --bootstrap B --topic T --out FILE     every record of the topic, one JSON line each (offset order)
    lag         --checkpoint DIR --bootstrap B --topic T   records of the topic not yet in a committed batch (prints an int)
    report      --clean-scores D --fault-scores D --clean-topic F --fault-topic F [--out-json F] [--timings F]
                                                           loss / duplicate report, exit 0 = PASS, 1 = FAIL

Definitions (key = (entity_id, window_start))
    Parquet duplicates : rows - distinct keys over all batch-*.parquet files of one run.
    lost               : keys of the clean run missing from the fault run.   extra: keys only the fault run has.
    topic duplicates   : records - distinct keys in the anomalies topic. The Kafka sink is at-least-once, so a replayed
                         batch may publish its anomalies twice; consumers de-duplicate on the key (contract), so this
                         is reported but is not a failure. Lost / extra are computed after de-duplication.
    mismatch           : keys in both runs whose ae_score differs by more than 1e-6 or whose is_anomaly differs.
PASS = no lost, no extra, no Parquet duplicates, no topic keys lost or extra.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

KEY = ("entity_id", "window_start")
SCORE_TOL = 1e-6


def load_scores(directory: Path) -> list[dict]:
    """All rows of directory/batch-*.parquet as dicts (entity_id, window_start, ae_score, is_anomaly, file)."""
    rows: list[dict] = []
    for f in sorted(Path(directory).glob("batch-*.parquet")):
        t = pq.read_table(f, columns=["entity_id", "window_start", "ae_score", "is_anomaly"]).to_pylist()
        for r in t:
            r["window_start"] = str(r["window_start"])
            r["file"] = f.name
            rows.append(r)
    return rows


def topic_rows(path: Path) -> list[dict]:
    """Records of a dump-topic file as dicts with entity_id and window_start normalised to str."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        v = json.loads(line)
        rows.append({"entity_id": v["entity_id"], "window_start": _norm_ts(v["window_start"]),
                     "ae_score": v.get("ae_score"), "is_anomaly": v.get("is_anomaly")})
    return rows


def _norm_ts(s) -> str:
    """'2026-01-01T00:05:00.000Z' (topic JSON) and '2026-01-01 00:05:00+00:00' (Parquet) -> same string."""
    import pandas as pd
    return str(pd.Timestamp(s).tz_convert("UTC") if pd.Timestamp(s).tzinfo else pd.Timestamp(s, tz="UTC"))


def keyed(rows: list[dict]) -> dict:
    """key -> first row of that key."""
    out: dict = {}
    for r in rows:
        out.setdefault((r["entity_id"], _norm_ts(r["window_start"])), r)
    return out


def compare(clean: list[dict], fault: list[dict]) -> dict:
    """Counts for one layer (Parquet or topic): rows, distinct, duplicates, lost, extra, mismatch."""
    c, f = keyed(clean), keyed(fault)
    lost, extra = sorted(set(c) - set(f)), sorted(set(f) - set(c))
    mismatch = []
    for k in set(c) & set(f):
        a, b = c[k], f[k]
        if a.get("is_anomaly") != b.get("is_anomaly") or abs((a.get("ae_score") or 0.0) - (b.get("ae_score") or 0.0)) > SCORE_TOL:
            mismatch.append(k)
    return {"clean_rows": len(clean), "clean_distinct": len(c), "fault_rows": len(fault), "fault_distinct": len(f),
            "clean_duplicates": len(clean) - len(c), "fault_duplicates": len(fault) - len(f),
            "lost": len(lost), "extra": len(extra), "mismatch": len(mismatch),
            "lost_sample": [list(k) for k in lost[:5]], "extra_sample": [list(k) for k in extra[:5]]}


def verdict(parquet: dict, topic: dict) -> bool:
    return (parquet["lost"] == 0 and parquet["extra"] == 0 and parquet["fault_duplicates"] == 0
            and topic["lost"] == 0 and topic["extra"] == 0)


# ---- Kafka / checkpoint --------------------------------------------------------------------------------
def _consumer(bootstrap: str):
    from confluent_kafka import Consumer
    return Consumer({"bootstrap.servers": bootstrap, "group.id": "e3-compare-" + str(id(object())),
                     "enable.auto.commit": False, "auto.offset.reset": "earliest"})


def end_offsets(bootstrap: str, topic: str) -> dict[int, int]:
    from confluent_kafka import TopicPartition
    c = _consumer(bootstrap)
    try:
        parts = c.list_topics(topic, timeout=15).topics[topic].partitions
        return {p: c.get_watermark_offsets(TopicPartition(topic, p), timeout=15)[1] for p in parts}
    finally:
        c.close()


def dump_topic(bootstrap: str, topic: str, out: Path) -> int:
    from confluent_kafka import TopicPartition
    ends = end_offsets(bootstrap, topic)
    c = _consumer(bootstrap)
    c.assign([TopicPartition(topic, p, 0) for p in ends])
    pos = {p: 0 for p in ends}
    n = 0
    try:
        with Path(out).open("w", encoding="utf-8") as fh:
            while any(pos[p] < ends[p] for p in ends):
                m = c.poll(5.0)
                if m is None or m.error():
                    if m is None:
                        break
                    continue
                pos[m.partition()] = m.offset() + 1
                fh.write(m.value().decode("utf-8") + "\n")
                n += 1
    finally:
        c.close()
    return n


def committed_end_offsets(checkpoint: Path) -> dict[str, dict[str, int]] | None:
    """Source offsets of the newest COMMITTED batch (offsets/N where commits/N exists), or None before the first one."""
    commits = sorted(int(p.name) for p in (checkpoint / "commits").glob("*") if p.name.isdigit())
    if not commits:
        return None
    lines = (checkpoint / "offsets" / str(commits[-1])).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[2])


def lag(checkpoint: Path, bootstrap: str, topic: str) -> int:
    ends = end_offsets(bootstrap, topic)
    done = (committed_end_offsets(checkpoint) or {}).get(topic, {})
    return sum(max(0, e - int(done.get(str(p), 0))) for p, e in ends.items())


# ---- report --------------------------------------------------------------------------------------------
def render(parquet: dict, topic: dict, timings: dict, ok: bool) -> str:
    def row(name, d):
        return (f"| {name} | {d['clean_rows']} | {d['fault_rows']} | {d['fault_duplicates']} | "
                f"{d['lost']} | {d['extra']} | {d['mismatch']} |")
    lines = ["# E3 fault tolerance", "",
             "| layer | clean rows | fault rows | duplicates in fault run | lost | extra | value mismatch |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
             row("Parquet (scores_stream)", parquet), row("topic anomalies (de-duplicated)", topic), ""]
    lines += [f"- topic `anomalies`: {topic['fault_rows']} records, {topic['fault_distinct']} distinct "
              f"(entity_id, window_start): {topic['fault_duplicates']} duplicate publishes (at-least-once sink; "
              f"consumers de-duplicate on the key). Clean run: {topic['clean_duplicates']}."]
    for k, v in timings.items():
        lines.append(f"- {k}: {v}")
    lines += ["", f"**Result: {'PASS' if ok else 'FAIL'}** (lost 0, extra 0, Parquet duplicates 0)"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-topic")
    d.add_argument("--bootstrap", required=True), d.add_argument("--topic", required=True)
    d.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("lag")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--bootstrap", required=True), g.add_argument("--topic", required=True)
    r = sub.add_parser("report")
    for a in ("--clean-scores", "--fault-scores", "--clean-topic", "--fault-topic"):
        r.add_argument(a, type=Path, required=True)
    r.add_argument("--timings", type=Path, default=None, help="JSON object name -> text, printed in the report")
    r.add_argument("--out-json", type=Path, default=None)
    r.add_argument("--out-md", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "dump-topic":
        print(dump_topic(a.bootstrap, a.topic, a.out))
        return 0
    if a.cmd == "lag":
        print(lag(a.checkpoint, a.bootstrap, a.topic))
        return 0
    parquet = compare(load_scores(a.clean_scores), load_scores(a.fault_scores))
    topic = compare(topic_rows(a.clean_topic), topic_rows(a.fault_topic))
    timings = json.loads(a.timings.read_text(encoding="utf-8")) if a.timings else {}
    ok = verdict(parquet, topic)
    text = render(parquet, topic, timings, ok)
    print(text)
    if a.out_md:
        a.out_md.write_text(text + "\n", encoding="utf-8")
    if a.out_json:
        a.out_json.write_text(json.dumps({"parquet": parquet, "topic": topic, "timings": timings, "pass": ok},
                                         indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
