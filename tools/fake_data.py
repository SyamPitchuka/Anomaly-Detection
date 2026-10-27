"""Generate small, deterministic fake data in the frozen contract shapes.

Run from the repo root:
    python tools/fake_data.py                       # all layers
    python tools/fake_data.py --layers scores incidents

Writes:
    data/sample/telemetry_smd.jsonl      one Kafka value per line (3 machines x 60 rows)
    data/sample/telemetry_alibaba.jsonl  (2 machines x 60 rows)
    data/raw|features|scores|incidents/source=smd/fake.parquet
Machine machine-1-2 has a planted anomaly in minutes 20-34 so scores and incidents are non-empty.
"""
import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SMD_MACHINES = ["machine-1-1", "machine-1-2", "machine-2-1"]
ALI_MACHINES = ["m_1932", "m_2051"]
ROWS, WINDOW_MIN = 60, 5
ANOM_ENTITY, ANOM_START, ANOM_END = "machine-1-2", 20, 34


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def smd_rows(rng):
    out = []
    for e in SMD_MACHINES:
        for i in range(ROWS):
            bad = e == ANOM_ENTITY and ANOM_START <= i <= ANOM_END
            m = {k: round(0.2 + 0.05 * math.sin(i / 5 + j) + rng.random() * 0.02 + (0.6 if bad and j < 3 else 0), 4)
                 for j, k in enumerate(S.SMD_METRICS)}
            out.append({"source": "smd", "entity_id": e, "event_time": BASE + timedelta(minutes=i),
                        "metrics": m, "attrs": {"split": "test"}, "label": int(bad)})
    return out


def ali_rows(rng):
    out = []
    for e in ALI_MACHINES:
        for i in range(ROWS):
            m = {k: round(30 + 10 * rng.random(), 2) for k in S.ALIBABA_METRICS}
            out.append({"source": "alibaba", "entity_id": e, "event_time": BASE + timedelta(seconds=10 * i),
                        "metrics": m, "attrs": {}, "label": None})
    return out


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({**r, "event_time": iso(r["event_time"])}, separators=(",", ":")) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} records)")


def write_layer(layer, rows):
    schema = S.arrow_schema(layer)
    cols = {name: [r[name] for r in rows] for name in schema.names}
    cols = {k: [list(v.items()) if isinstance(v, dict) else v for v in vals] for k, vals in cols.items()}
    table = pa.table(cols, schema=schema)
    path = ROOT / S.PATHS[layer] / "source=smd" / "fake.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"wrote {path.relative_to(ROOT)} ({table.num_rows} rows)")


def windows(rows):
    by = {}
    for r in rows:
        w = r["event_time"].replace(minute=(r["event_time"].minute // WINDOW_MIN) * WINDOW_MIN, second=0)
        by.setdefault((r["entity_id"], w), []).append(r)
    return by


def features(rows):
    out = []
    for (e, w), rs in sorted(windows(rows).items()):
        for k in S.SMD_METRICS[:3]:
            v = [r["metrics"][k] for r in rs]
            mean = sum(v) / len(v)
            std = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
            out.append({"entity_id": e, "window_start": w, "window_end": w + timedelta(minutes=WINDOW_MIN),
                        "metric": k, "mean": mean, "max": max(v), "std": std,
                        "slope": (v[-1] - v[0]) / max(len(v) - 1, 1), "n": len(v)})
    return out


def scores(rows):
    out = []
    for (e, w), rs in sorted(windows(rows).items()):
        bad = any(r["label"] for r in rs)
        t1 = 4.5 if bad else 0.8
        t2 = 3.9 if bad else 0.6
        ae = 5.2 if bad else 0.4
        out.append({"entity_id": e, "window_start": w, "window_end": w + timedelta(minutes=WINDOW_MIN),
                    "tier1_score": t1, "tier1_flag": t1 > 3, "tier2_score": t2, "tier2_flag": t2 > 3,
                    "pca_score": 2.8 if bad else 0.5, "ae_score": ae, "ae_flag": ae > 3,
                    "is_anomaly": bad, "top_metrics": ["m00", "m01", "m02"] if bad else [], "label": int(bad)})
    return out


def incidents(score_rows):
    out, cur = [], None
    for s in score_rows + [None]:
        if s and s["is_anomaly"] and cur and cur["entity_id"] == s["entity_id"] and cur["end_time"] == s["window_start"]:
            cur["end_time"], cur["n_windows"] = s["window_end"], cur["n_windows"] + 1
            cur["max_score"] = max(cur["max_score"], s["tier1_score"])
            continue
        if cur:
            cur["incident_id"] = f"{cur['entity_id']}_{cur['start_time'].strftime('%Y%m%d%H%M')}"
            out.append(cur)
            cur = None
        if s and s["is_anomaly"]:
            cur = {"entity_id": s["entity_id"], "start_time": s["window_start"], "end_time": s["window_end"],
                   "n_windows": 1, "max_score": s["tier1_score"], "top_metrics": s["top_metrics"],
                   "created_at": BASE + timedelta(days=1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="*", default=["jsonl", "raw", "features", "scores", "incidents"])
    a = ap.parse_args()
    rng = random.Random(42)
    smd, ali = smd_rows(rng), ali_rows(rng)
    if "jsonl" in a.layers:
        write_jsonl(smd, ROOT / "data/sample/telemetry_smd.jsonl")
        write_jsonl(ali, ROOT / "data/sample/telemetry_alibaba.jsonl")
    if "raw" in a.layers:
        write_layer("raw", smd)
    if "features" in a.layers:
        write_layer("features", features(smd))
    sc = scores(smd)
    if "scores" in a.layers:
        write_layer("scores", sc)
    if "incidents" in a.layers:
        write_layer("incidents", incidents(sc))


if __name__ == "__main__":
    main()
