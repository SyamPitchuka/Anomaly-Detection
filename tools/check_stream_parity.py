"""Inspection gauge: the streaming score job gives the same AE scores as the batch autoencoder.

Run from the repo root, with the same model for both:
    python src/detection/autoencoder.py --sample 2
    python src/streaming/score_job.py --input files --machines 2 --once --reset
    python tools/check_stream_parity.py --machines 2

Compares data/scores_stream/source=smd with data/scores_ae/source=smd for the first N machine files,
matched on (entity_id, window_start) (stream duplicates removed, as the contract says). Reports per machine
the windows on each side, windows missing from the stream (expected: the last 2-3 test windows of a
machine stay open in the state until later rows arrive), max |ae_score diff| and ae_flag agreement.
PASS if there are matched windows, max |diff| <= 0.001 and ae_flag agrees on >= 99.9% of them.
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
BATCH_DIR = ROOT / "data" / "scores_ae" / "source=smd"
RAW_DIR = ROOT / S.PATHS["raw"] / "source=smd"
MAX_DIFF = 1e-3
MIN_AGREEMENT = 0.999
COLS = ["entity_id", "window_start", "ae_score", "ae_flag"]


def _read(path: Path) -> pd.DataFrame:
    files = sorted(p for p in path.glob("*.parquet")) if path.exists() else []
    if not files:
        return pd.DataFrame(columns=COLS)
    df = ds.dataset([str(p) for p in files], format="parquet").to_table(columns=COLS).to_pandas()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    return df


def _flag(s: pd.Series) -> pd.Series:
    """Nullable boolean -> bool, null counts as not flagged."""
    return s.astype("boolean").fillna(False).astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machines", type=int, default=None, help="only the first N machine files")
    args = parser.parse_args()

    machines = sorted(p.stem for p in RAW_DIR.glob("*.parquet") if p.name != "fake.parquet")
    if args.machines:
        machines = machines[:args.machines]
    stream, batch = _read(STREAM_DIR), _read(BATCH_DIR)
    stream = stream[stream["entity_id"].isin(machines)]
    batch = batch[batch["entity_id"].isin(machines)]
    dups = int(stream.duplicated(["entity_id", "window_start"]).sum())
    stream = stream.drop_duplicates(["entity_id", "window_start"], keep="last")
    print(f"stream {len(stream)} windows ({dups} duplicates removed), batch {len(batch)} windows, "
          f"{len(machines)} machines")

    m = batch.merge(stream, on=["entity_id", "window_start"], how="outer", suffixes=("_b", "_s"),
                    indicator=True)
    matched = m[m["_merge"] == "both"].copy()
    a, b = matched["ae_score_s"].astype(float).to_numpy(), matched["ae_score_b"].astype(float).to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    matched["diff"] = np.where(both_nan, 0.0, np.abs(a - b))
    matched["diff"] = matched["diff"].fillna(np.inf)      # one side null
    matched["agree"] = _flag(matched["ae_flag_s"]) == _flag(matched["ae_flag_b"])

    rows = []
    for entity in machines:
        e = m[m["entity_id"] == entity]
        mt = matched[matched["entity_id"] == entity]
        rows.append({"machine": entity, "batch": int((e["_merge"] != "right_only").sum()),
                     "stream": int((e["_merge"] != "left_only").sum()), "matched": len(mt),
                     "missing_in_stream": int((e["_merge"] == "left_only").sum()),
                     "extra_in_stream": int((e["_merge"] == "right_only").sum()),
                     "max_abs_diff": mt["diff"].max() if len(mt) else float("nan"),
                     "flag_agree": mt["agree"].mean() if len(mt) else float("nan"),
                     "flags_b/s": f"{int(_flag(mt['ae_flag_b']).sum())}/{int(_flag(mt['ae_flag_s']).sum())}"})
    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        print(pd.DataFrame(rows).to_string(index=False))

    missing = m[m["_merge"] == "left_only"]
    if len(missing):
        last = batch.groupby("entity_id")["window_start"].max()
        tail = (missing["window_start"] > missing["entity_id"].map(last) - pd.Timedelta(minutes=15)).sum()
        print(f"missing in stream: {len(missing)} windows, {tail} of them in the last 15 minutes of a machine "
              "(still open in the stream state)")
    n = len(matched)
    max_diff = float(matched["diff"].max()) if n else float("nan")
    agree = float(matched["agree"].mean()) if n else float("nan")
    ok_n, ok_diff, ok_agree = n > 0, n > 0 and max_diff <= MAX_DIFF, n > 0 and agree >= MIN_AGREEMENT
    print(f"{'OK  ' if ok_n else 'FAIL'} {n} matched windows")
    print(f"{'OK  ' if ok_diff else 'FAIL'} max |ae_score diff| {max_diff:.3g} (limit {MAX_DIFF})")
    print(f"{'OK  ' if ok_agree else 'FAIL'} ae_flag agreement {agree:.4%} "
          f"({int((~matched['agree']).sum()) if n else 0} disagree, limit {MIN_AGREEMENT:.1%})")
    ok = ok_n and ok_diff and ok_agree
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
