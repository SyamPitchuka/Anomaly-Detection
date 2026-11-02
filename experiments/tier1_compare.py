"""Tier 1 v1 stream vs batch parity, and the alarm metrics with Tier 1 on vs off.

Run from the repo root after
    python src/detection/tier1_zscore.py          # batch Tier 1 -> data/scores_tier1 (+ baselines; rename to
                                                  # models/tier1_smd/baselines_v1.parquet for the stream)
    python src/detection/autoencoder.py           # batch AE v2 -> data/scores_ae
    python src/streaming/score_job.py --input files --once --reset --tier1 v1
    python experiments/tier1_compare.py

Matches stream and batch test windows on (entity_id, window_start) (stream duplicates removed). Reports
max |tier1_score diff| and |ae_score diff|, % identical tier1_flag / ae_flag, whether the stream's
is_anomaly equals ae_flag (AE-only alarm), how many windows tier1_flag adds on top of ae_flag, and composite
F1, faults caught, alarm precision, false alarms per machine per day and median detection delay for
"Tier 1 off" (ae_flag) vs "Tier 1 on" (ae_flag OR tier1_flag), scored as in src/detection/autoencoder.py.
Results: docs/results/tier1_stream.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.detection.evaluate import composite_metrics, detection_delays, false_alarms_per_day  # noqa: E402

KEYS = ["entity_id", "window_start"]
WINDOW_MINUTES = 5
STREAM_DIR = "data/scores_stream/source=smd"
TIER1_DIR = "data/scores_tier1/source=smd"
AE_DIR = "data/scores_ae/source=smd"


def read(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=cols)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    return df


def flag(s: pd.Series) -> np.ndarray:
    """Nullable boolean -> bool, null counts as not flagged."""
    return s.astype("boolean").fillna(False).astype(bool).to_numpy()


def abs_diff(a: pd.Series, b: pd.Series) -> np.ndarray:
    """|a - b|, 0 where both are null, inf where only one is."""
    a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
    d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
    return np.nan_to_num(d, nan=np.inf)


def alarm_metrics(df: pd.DataFrame, pred: np.ndarray, label_col: str = "label") -> dict:
    """Overall metrics as in autoencoder.evaluate: composite over machines joined by a (0, 0) separator,
    false alarms per machine-day, median delay (minutes) over all detected faults."""
    df = df.assign(_p=np.asarray(pred, dtype=int))
    all_t, all_p, test_t, test_p, delays = [], [], [], [], []
    for _, g in df.sort_values(KEYS).groupby("entity_id", sort=True):
        t, p = g[label_col].to_numpy(dtype=int), g["_p"].to_numpy()
        all_t += [0, *t]
        all_p += [0, *p]
        test_t += t.tolist()
        test_p += p.tolist()
        delays += detection_delays(t, p)
    c = composite_metrics(all_t, all_p)
    return {"composite_f1": c["composite_f1"], "faults_caught": f"{c['events_detected']}/{c['events']}",
            "alarm_precision": c["alarm_precision"],
            "false_alarms_per_machine_day": false_alarms_per_day(test_t, test_p, WINDOW_MINUTES),
            "median_delay_min": float(np.median(delays)) * WINDOW_MINUTES if delays else float("nan"),
            "flagged": int(np.sum(test_p))}


def compare(stream: pd.DataFrame, tier1: pd.DataFrame, ae: pd.DataFrame) -> dict:
    """stream: KEYS + tier1_score, tier1_flag, ae_score, ae_flag, is_anomaly; tier1 (batch): KEYS + tier1_score,
    tier1_flag; ae (batch): KEYS + ae_score, ae_flag, label."""
    dups = int(stream.duplicated(KEYS).sum())
    stream = stream.drop_duplicates(KEYS, keep="last")
    batch = ae.merge(tier1, on=KEYS, how="left")
    m = batch.merge(stream, on=KEYS, suffixes=("_b", "_s"))
    f1s, f1b = flag(m["tier1_flag_s"]), flag(m["tier1_flag_b"])
    aes, aeb = flag(m["ae_flag_s"]), flag(m["ae_flag_b"])
    label = m["label"].to_numpy(dtype=int)
    adds = f1s & ~aes
    return {
        "stream_windows": len(stream), "duplicates": dups, "batch_windows": len(batch), "matched": len(m),
        "max_tier1_diff": float(abs_diff(m["tier1_score_s"], m["tier1_score_b"]).max()) if len(m) else np.nan,
        "max_ae_diff": float(abs_diff(m["ae_score_s"], m["ae_score_b"]).max()) if len(m) else np.nan,
        "tier1_flag_identical": float((f1s == f1b).mean()) if len(m) else np.nan,
        "ae_flag_identical": float((aes == aeb).mean()) if len(m) else np.nan,
        "is_anomaly_is_ae_flag": bool((flag(m["is_anomaly"]) == aes).all()),
        "tier1_adds": int(adds.sum()), "tier1_adds_on_anomalies": int((adds & (label == 1)).sum()),
        "rows": {"Tier 1 off (ae_flag)": alarm_metrics(m, aes),
                 "Tier 1 on (ae_flag OR tier1_flag)": alarm_metrics(m, aes | f1s)},
    }


def main(root: Path = ROOT) -> int:
    stream = read(root / STREAM_DIR, KEYS + ["tier1_score", "tier1_flag", "ae_score", "ae_flag", "is_anomaly"])
    tier1 = read(root / TIER1_DIR, KEYS + ["tier1_score", "tier1_flag"])
    ae = read(root / AE_DIR, KEYS + ["ae_score", "ae_flag", "label"])
    r = compare(stream, tier1, ae)
    print(f"stream {r['stream_windows']} windows ({r['duplicates']} duplicates removed), batch {r['batch_windows']} "
          f"test windows, matched {r['matched']} (missing in stream {r['batch_windows'] - r['matched']})")
    print(f"max |tier1_score diff| {r['max_tier1_diff']:.3g}   max |ae_score diff| {r['max_ae_diff']:.3g}")
    print(f"tier1_flag identical {r['tier1_flag_identical']:.4%}   ae_flag identical {r['ae_flag_identical']:.4%}   "
          f"stream is_anomaly == ae_flag: {r['is_anomaly_is_ae_flag']}")
    print(f"tier1_flag adds {r['tier1_adds']} windows on top of ae_flag "
          f"({r['tier1_adds_on_anomalies']} of them labelled anomalous)")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(r["rows"]).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
