"""Inspection gauge for Part A: checks every JSONL sample and Parquet layer that exists under data/.

Run from the repo root:  python tools/check_contracts.py [--require raw scores ...]
Prints one line per item and ends with PASS or FAIL. Layers that do not exist yet are SKIPPED
unless listed in --require.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.dataset as ds  # noqa: E402

from contracts import schemas as S  # noqa: E402

ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
FIELDS = ["source", "entity_id", "event_time", "metrics", "attrs", "label"]
problems = []


def bad(msg):
    problems.append(msg)
    print(f"  FAIL {msg}")


def check_record(r, where):
    if list(r.keys()) != FIELDS:
        return bad(f"{where}: fields {list(r.keys())} != {FIELDS}")
    if r["source"] not in S.SOURCES:
        bad(f"{where}: source '{r['source']}' not in {S.SOURCES}")
    if not ISO.match(str(r["event_time"])):
        bad(f"{where}: event_time '{r['event_time']}' is not yyyy-mm-ddThh:mm:ss.SSSZ")
    allowed = S.SMD_METRICS if r["source"] == "smd" else S.ALIBABA_METRICS
    extra = set(r["metrics"]) - set(allowed)
    if extra:
        bad(f"{where}: unknown metric keys {sorted(extra)[:5]}")
    if any(not isinstance(v, (int, float)) for v in r["metrics"].values()):
        bad(f"{where}: metric values must be numbers")
    if r["label"] not in (0, 1, None):
        bad(f"{where}: label must be 0, 1 or null")


def check_jsonl():
    files = sorted((ROOT / "data/sample").glob("*.jsonl"))
    if not files:
        print("SKIP  jsonl samples (none in data/sample/)")
    for f in files:
        n0 = len(problems)
        with f.open() as fh:
            for i, line in enumerate(fh, 1):
                check_record(json.loads(line), f"{f.name} line {i}")
                if len(problems) - n0 > 5:
                    break
        print(f"{'OK   ' if len(problems) == n0 else 'FAIL '} {f.relative_to(ROOT)}")


def check_layer(layer, required):
    path = ROOT / S.PATHS[layer]
    if not path.exists() or not any(path.rglob("*.parquet")):
        if required:
            bad(f"{layer}: required but no Parquet files under {S.PATHS[layer]}/")
        else:
            print(f"SKIP  {layer} (nothing in {S.PATHS[layer]}/ yet)")
        return
    n0 = len(problems)
    dset = ds.dataset(path, format="parquet", partitioning="hive")
    got = {f.name: f.type for f in dset.schema}
    for name, code in S.LAYERS[layer].items():
        if name not in got:
            bad(f"{layer}: missing column '{name}'")
        elif not S.type_matches(code, got[name]):
            bad(f"{layer}: column '{name}' is {got[name]}, contract says {code}")
    extra = set(got) - set(S.LAYERS[layer]) - {"source"}
    if extra:
        bad(f"{layer}: extra columns not in contract {sorted(extra)}")
    if layer != "smd_interpretation":
        srcs = [d.name.split("=", 1)[1] for d in path.iterdir() if d.is_dir() and d.name.startswith("source=")]
        if not srcs or any(s not in S.SOURCES for s in srcs):
            bad(f"{layer}: partition folders must be source=smd or source=alibaba, found {srcs}")
    rows = dset.count_rows()
    print(f"{'OK   ' if len(problems) == n0 else 'FAIL '} {layer}: {rows} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", nargs="*", default=[])
    a = ap.parse_args()
    print("Contract revision B")
    check_jsonl()
    for layer in S.LAYERS:
        check_layer(layer, layer in a.require)
    print("PASS" if not problems else f"FAIL ({len(problems)} problems)")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
