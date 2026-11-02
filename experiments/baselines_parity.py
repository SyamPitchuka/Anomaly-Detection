"""Tier 2 / 2b parity: streaming score job vs batch baselines.py on the same models.

Run from the repo root, after baselines.py has written models/ and data/scores/source=smd:
    python src/streaming/score_job.py --input files --machines 2 --once --reset
    python experiments/baselines_parity.py --machines 2

Compares data/scores_stream/source=smd with data/scores/source=smd for the first N machine files, matched
on (entity_id, window_start) (stream duplicates removed, as the contract says). Reports per machine and
overall: matched windows, windows missing from the stream (expected: the last 2-3 test windows of a
machine stay open in the state), max |tier2_score diff|, max |pca_score diff| and % identical tier2_flag.
PASS (same targets as tools/check_stream_parity.py) if there are matched windows, both max diffs <= 0.001
and tier2_flag is identical on >= 99.9% of them.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

from contracts import schemas as S  # noqa: E402

STREAM_DIR = ROOT / "data" / "scores_stream" / "source=smd"
BATCH_DIR = ROOT / S.PATHS["scores"] / "source=smd"
RAW_DIR = ROOT / S.PATHS["raw"] / "source=smd"
KEYS = ["entity_id", "window_start"]
COLS = KEYS + ["tier2_score", "tier2_flag", "pca_score"]
MAX_DIFF = 1e-3
MIN_AGREEMENT = 0.999


def _read(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet")) if path.exists() else []
    if not files:
        return pd.DataFrame(columns=COLS)
    df = ds.dataset([str(p) for p in files], format="parquet").to_table(columns=COLS).to_pandas()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    return df


def _flag(s: pd.Series) -> pd.Series:
    return s.astype("boolean").fillna(False).astype(bool)


def compare(stream: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    """One row per machine plus 'ALL': stream/batch windows, matched, missing, max diffs, % flags identical."""
    stream = stream.drop_duplicates(KEYS, keep="last")
    m = stream.merge(batch, on=KEYS, suffixes=("_s", "_b"), validate="one_to_one")
    m["d_tier2"] = (m["tier2_score_s"] - m["tier2_score_b"]).abs()
    m["d_pca"] = (m["pca_score_s"] - m["pca_score_b"]).abs()
    m["same_flag"] = _flag(m["tier2_flag_s"]) == _flag(m["tier2_flag_b"])
    rows = []
    for name in sorted(batch["entity_id"].unique()) + ["ALL"]:
        sel = (lambda d: d) if name == "ALL" else (lambda d, n=name: d[d["entity_id"] == n])
        mm = sel(m)
        rows.append({"machine": name, "stream": len(sel(stream)), "batch": len(sel(batch)), "matched": len(mm),
                     "missing": len(sel(batch)) - len(mm),
                     "max_d_tier2": mm["d_tier2"].max() if len(mm) else np.nan,
                     "max_d_pca": mm["d_pca"].max() if len(mm) else np.nan,
                     "nan_scores": int(mm[["tier2_score_s", "pca_score_s"]].isna().any(axis=1).sum()),
                     "flag_identical_%": 100 * mm["same_flag"].mean() if len(mm) else np.nan,
                     "flags_s/b": f"{int(_flag(mm['tier2_flag_s']).sum())}/{int(_flag(mm['tier2_flag_b']).sum())}"})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machines", type=int, default=2, help="first N machine files (default 2)")
    args = parser.parse_args()

    machines = sorted(p.stem for p in RAW_DIR.glob("*.parquet") if p.name != "fake.parquet")[:args.machines]
    stream, batch = _read(STREAM_DIR), _read(BATCH_DIR)
    stream = stream[stream["entity_id"].isin(machines)]
    batch = batch[batch["entity_id"].isin(machines)]
    print(f"machines: {', '.join(machines)}; stream duplicates removed: "
          f"{int(stream.duplicated(KEYS).sum())}")
    table = compare(stream, batch)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        print(table.to_string(index=False))

    a = table.iloc[-1]
    ok = (a["matched"] > 0 and a["nan_scores"] == 0 and a["max_d_tier2"] <= MAX_DIFF
          and a["max_d_pca"] <= MAX_DIFF and a["flag_identical_%"] >= 100 * MIN_AGREEMENT)
    print(f"{'PASS' if ok else 'FAIL'} baselines parity: {a['matched']} matched windows, "
          f"max |tier2 diff| {a['max_d_tier2']:.3g}, max |pca diff| {a['max_d_pca']:.3g}, "
          f"tier2_flag identical {a['flag_identical_%']:.2f}% "
          f"(targets <= {MAX_DIFF}, >= {100 * MIN_AGREEMENT:.1f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
