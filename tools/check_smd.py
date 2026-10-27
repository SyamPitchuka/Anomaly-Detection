"""Inspection gauge for WP-3 data: SMD converted to the raw layer and the interpretation reference.

Run from the repo root:  python tools/check_smd.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

from contracts import schemas as S  # noqa: E402

fails = 0


def check(ok, msg):
    global fails
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {msg}")


raw = ROOT / S.PATHS["raw"] / "source=smd"
real = [p for p in raw.rglob("*.parquet") if p.name != "fake.parquet"] if raw.exists() else []
check(bool(real), f"real SMD Parquet files under {raw.relative_to(ROOT)}/ (fake.parquet ignored)")
if real:
    t = ds.dataset(real, format="parquet").to_table(columns=["entity_id", "metrics", "attrs", "label"])
    ents = pc.unique(t["entity_id"]).to_pylist()
    check(len(ents) == 28, f"28 machines (found {len(ents)})")
    check(all(e.startswith("machine-") for e in ents), "entity_id looks like machine-1-1")
    keys = set(pc.unique(pc.list_flatten(pc.map_keys(t["metrics"]))).to_pylist())
    check(keys == set(S.SMD_METRICS), f"metric keys are exactly m00..m37 (found {len(keys)})")
    splits = pc.list_flatten(pc.map_values(t["attrs"]))
    n_test = pc.sum(pc.equal(splits, "test")).as_py() or 0
    n_train = pc.sum(pc.equal(splits, "train")).as_py() or 0
    check(n_train > 0 and n_test > 0, f"attrs.split has train ({n_train}) and test ({n_test}) rows")
    n_pos = pc.sum(pc.equal(t["label"], 1)).as_py() or 0
    check(n_pos > 0, f"test rows include label = 1 ({n_pos} anomalous rows)")
    print(f"     total rows: {t.num_rows}")

ref = ROOT / S.PATHS["smd_interpretation"]
has_ref = ref.exists() and any(ref.rglob("*.parquet"))
check(has_ref, f"interpretation reference under {S.PATHS['smd_interpretation']}/")
if has_ref:
    r = ds.dataset(ref, format="parquet").to_table()
    mk = set(pc.unique(pc.list_flatten(r["metrics"])).to_pylist())
    check(mk <= set(S.SMD_METRICS), "interpretation metrics use m00..m37 names")
    print(f"     interpretation segments: {r.num_rows}")

print("PASS" if not fails else "FAIL")
sys.exit(0 if not fails else 1)
