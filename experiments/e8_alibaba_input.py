"""E8 input for Alibaba: AE scores and incidents of the 40 Alibaba machines on hours 0-72, in a staging root.

Run from the repo root (venv active; PyTorch from requirements-ml.txt):
    python experiments/e8_alibaba_input.py            # -> data/e8/alibaba_root/data/{raw,scores,incidents}/source=alibaba

1. raw: the rows of data/raw/source=alibaba with event_time in [T0, T0 + 72 h) (T0 = 2026-01-01T00:00:00Z), one
   Parquet file per machine (score_layer wants each machine in one file). These hours were never seen in training:
   models/ae_alibaba was trained on hours 72-192 (experiments/e5b_alibaba_ae.py, --train-from-hours 72). No faults
   are injected; this is the original trace.
2. scores: src/detection/autoencoder.py score_layer(models/ae_alibaba, raw) -- every window treated as a test
   window in time order (recenter on the machine's previous 24 windows, adaptive limit over its previous 288 scores
   once 144 exist, global threshold as saved), written with the scores-layer schema; is_anomaly = ae_flag.
3. incidents: src/incidents/group.py run_batch(source=alibaba, root=data/e8/alibaba_root), the same grouping rule as
   the pipeline (10-minute session gap).
The pipeline layers (data/raw, data/scores, data/incidents) and the model are only read, never written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402

SOURCE = "alibaba"
HOURS = 72
RAW_DIR = ROOT / S.PATHS["raw"] / f"source={SOURCE}"
MODEL_DIR = ROOT / "models" / "ae_alibaba"
STAGE_ROOT = ROOT / "data" / "e8" / "alibaba_root"


def stage_dir(root: Path, layer: str) -> Path:
    return root / S.PATHS[layer] / f"source={SOURCE}"


def slice_raw(raw_dir: Path, out_dir: Path, hours: float = HOURS) -> dict:
    """Rows with T0 <= event_time < T0 + hours, one file per machine, raw schema. Returns counts."""
    t0 = pd.Timestamp(S.TIME_BASE_UTC)
    t1 = t0 + pd.Timedelta(hours=hours)
    schema = S.arrow_schema("raw")
    files = [p for p in Path(raw_dir).glob("*.parquet")]
    dataset = ds.dataset(files, schema=schema, format="parquet")
    ts = pa.scalar(t0.to_pydatetime(), type=schema.field("event_time").type)
    te = pa.scalar(t1.to_pydatetime(), type=schema.field("event_time").type)
    table = dataset.to_table(filter=(ds.field("event_time") >= ts) & (ds.field("event_time") < te))
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    machines = sorted(pc.unique(table["entity_id"]).to_pylist())
    for m in machines:
        part = table.filter(pc.equal(table["entity_id"], m))
        part = part.take(pc.sort_indices(part, sort_keys=[("event_time", "ascending")]))
        pq.write_table(part, out_dir / f"{m}.parquet")
    return {"rows": table.num_rows, "machines": len(machines),
            "first": str(pc.min(table["event_time"]).as_py()) if table.num_rows else None,
            "last": str(pc.max(table["event_time"]).as_py()) if table.num_rows else None}


def score(model_dir: Path, raw_dir: Path, out_dir: Path) -> dict:
    from src.detection import autoencoder as M
    meta, ae_score, ae_flag, top = M.score_layer(model_dir, raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(M.scores_table(meta, ae_score, ae_flag, top), out_dir / "scores.parquet")
    days = (meta["window_start"].max() - meta["window_start"].min()) / 86_400e6 + 5 / 1440
    return {"windows": int(len(meta)), "machines": int(meta["entity_id"].nunique()),
            "flagged": int(ae_flag.sum()), "scored": int(pd.notna(ae_score).sum()),
            "flags_per_machine_day": float(ae_flag.sum() / meta["entity_id"].nunique() / days) if days else None}


def _rel(path) -> str:
    full = Path(path).resolve()
    return full.relative_to(ROOT).as_posix() if full.is_relative_to(ROOT) else Path(path).as_posix()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--model", type=Path, default=MODEL_DIR)
    ap.add_argument("--root", type=Path, default=STAGE_ROOT, help="staging root (gets data/raw|scores|incidents)")
    ap.add_argument("--hours", type=float, default=HOURS)
    args = ap.parse_args(argv)
    t_all = time.perf_counter()
    info = {"hours": args.hours, "model": _rel(args.model)}
    t = time.perf_counter()
    info["raw"] = slice_raw(args.raw, stage_dir(args.root, "raw"), args.hours)
    info["raw_s"] = time.perf_counter() - t
    print(f"raw slice: {info['raw']}")
    t = time.perf_counter()
    info["scores"] = score(args.model, stage_dir(args.root, "raw"), stage_dir(args.root, "scores"))
    info["score_s"] = time.perf_counter() - t
    print(f"scores: {info['scores']}")
    from src.incidents import group as G
    t = time.perf_counter()
    spark = G.build_spark("e8_alibaba_incidents")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        info["incidents"] = G.run_batch(SOURCE, Path(args.root), spark)
    finally:
        spark.stop()
    info["incidents_s"] = time.perf_counter() - t
    info["total_s"] = time.perf_counter() - t_all
    info["commands"] = [f"python experiments/e8_alibaba_input.py --root {_rel(args.root)}",
                        "(incidents step = python src/incidents/group.py --source alibaba --root "
                        f"{_rel(args.root)})"]
    (Path(args.root) / "input.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Alibaba input built in {info['total_s']:.1f} s -> {Path(args.root).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
