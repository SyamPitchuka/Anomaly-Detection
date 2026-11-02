"""Tier 1 v2 grid, batch only (not wired into the stream).

Run from the repo root after `python src/detection/autoencoder.py` (data/scores_ae):
    python experiments/tier1_v2_grid.py

Per machine (pandas, one raw file at a time, so memory stays small): 5-minute epoch-aligned windows, mean
per metric (missing values skipped), split = test if any row is test, label = max (null -> 0).
- MAD floor: train median and MAD per metric from train windows (as src/detection/tier1_zscore.py), then
  mad = max(mad, 0.01 x |median| + 1e-3); scale = 1.4826 x mad.
- Recenter: z = |x - centre| / scale, centre = causal rolling median of the previous 24 windows (at least 12,
  current window excluded, else the train median), as autoencoder.recenter_centers.
- Agreement: raw flag if at least k metrics have z > t in the same window, k in {2, 3}, t in {4, 6}.
- Persistence: flag at window i only if the raw flags at i and i - 1 are both set (causal).
Evaluated on the AE v2 batch test windows (data/scores_ae), alone and as ae_flag OR tier1-v2.

Selection rule, fixed before the results were seen: among settings whose Tier 1 v2 false alarms are
<= 2 per machine per day, the highest composite F1; ties go to fewer false alarms. Results:
docs/results/tier1_stream.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402
from experiments.tier1_compare import AE_DIR, alarm_metrics, read  # noqa: E402

METRICS = list(S.SMD_METRICS)
KS, TS = (2, 3), (4.0, 6.0)
RECENTER, PERSIST = 24, 2
MAD_REL, MAD_ABS, MAD_TO_SIGMA = 0.01, 1e-3, 1.4826
MAX_FA_DAY = 2.0
WINDOW_US = 5 * 60 * 1_000_000
SKIP_FILES = {"fake.parquet"}


def windows(path: Path, metrics: list[str] = METRICS) -> pd.DataFrame:
    """One raw machine file -> one row per window: window_start (int64 us), <metrics> means, is_test, label."""
    t = pq.read_table(path, columns=["entity_id", "event_time", "metrics", "attrs", "label"])
    n = t.num_rows
    ts = t.column("event_time").combine_chunks().cast("int64").to_numpy()
    mcol = t.column("metrics").combine_chunks()
    row = np.repeat(np.arange(n), np.diff(mcol.offsets.to_numpy()))
    wide = (pd.DataFrame({"row": row, "k": mcol.keys.to_numpy(zero_copy_only=False),
                          "v": mcol.items.to_numpy(zero_copy_only=False).astype(float)})
            .pivot_table(index="row", columns="k", values="v", aggfunc="first")
            .reindex(index=np.arange(n), columns=metrics))
    wide["window_start"] = ts - ts % WINDOW_US
    wide["is_test"] = [int(dict(a or []).get("split") == "test") for a in t.column("attrs").to_pylist()]
    wide["label"] = pd.Series(t.column("label").to_pandas()).fillna(0).astype(int).to_numpy()
    g = wide.groupby("window_start", sort=True)
    out = g[metrics].mean()
    out["is_test"], out["label"] = g["is_test"].max(), g["label"].max()
    out = out.reset_index()
    out.insert(0, "entity_id", t.column("entity_id")[0].as_py())
    return out


def z_matrix(w: pd.DataFrame, metrics: list[str] = METRICS, recenter: int = RECENTER) -> np.ndarray:
    """|x - centre| / (1.4826 x floored MAD) per window and metric (NaN where the window has no value)."""
    X = w[metrics].to_numpy(dtype=float)
    train = X[w["is_test"].to_numpy() == 0]
    med = np.nanmedian(train, axis=0)
    mad = np.nanmedian(np.abs(train - med), axis=0)
    mad = np.maximum(np.nan_to_num(mad, nan=0.0), MAD_REL * np.abs(np.nan_to_num(med)) + MAD_ABS)
    rolled = pd.DataFrame(X).rolling(recenter, min_periods=max(1, recenter // 2)).median().shift(1).to_numpy()
    centre = np.where(np.isnan(rolled), med, rolled)
    with np.errstate(invalid="ignore"):
        return np.abs(X - centre) / (MAD_TO_SIGMA * mad)


def v2_flags(Z: np.ndarray, k: int, t: float, persist: int = PERSIST) -> np.ndarray:
    """At least k metrics with z > t, held for `persist` consecutive windows (flag on the last of them)."""
    raw = (np.nan_to_num(Z, nan=0.0) > t).sum(axis=1) >= k
    out = raw.copy()
    for lag in range(1, persist):
        out &= np.concatenate([np.zeros(lag, dtype=bool), raw[:-lag]])
    return out


def setting(k: int, t: float) -> str:
    return f"k{k}_t{t:g}"


def score_machines(files: list[Path]) -> pd.DataFrame:
    """Test windows of every machine with one boolean column per grid setting."""
    parts = []
    for f in files:
        w = windows(f)
        Z = z_matrix(w)
        keep = w[["entity_id", "window_start", "is_test", "label"]].copy()
        for k in KS:
            for t in TS:
                keep[setting(k, t)] = v2_flags(Z, k, t)
        parts.append(keep[keep["is_test"] == 1])
    v2 = pd.concat(parts, ignore_index=True)
    v2["window_start"] = pd.to_datetime(v2["window_start"], unit="us", utc=True)
    return v2


def run_grid(df: pd.DataFrame) -> pd.DataFrame:
    """df: entity_id, window_start, label, ae_flag (bool) and one bool column per setting."""
    ae = df["ae_flag"].to_numpy(dtype=bool)
    rows = []
    for k in KS:
        for t in TS:
            v2 = df[setting(k, t)].to_numpy(dtype=bool)
            a, o = alarm_metrics(df, v2), alarm_metrics(df, ae | v2)
            rows.append({"k": k, "t": t, **{f"t1_{x}": v for x, v in a.items()}, "adds_on_ae": int((v2 & ~ae).sum()),
                         **{f"or_{x}": o[x] for x in ("composite_f1", "false_alarms_per_machine_day",
                                                      "median_delay_min")}})
    return pd.DataFrame(rows)


def select(grid: pd.DataFrame, max_fa: float = MAX_FA_DAY) -> pd.Series | None:
    """The pre-registered rule: FA <= max_fa per machine per day, highest composite F1, ties -> fewer FA."""
    ok = grid[grid["t1_false_alarms_per_machine_day"] <= max_fa]
    if ok.empty:
        return None
    return ok.sort_values(["t1_composite_f1", "t1_false_alarms_per_machine_day"], ascending=[False, True]).iloc[0]


def main(root: Path = ROOT) -> int:
    raw_dir = root / S.PATHS["raw"] / "source=smd"
    files = sorted(p for p in raw_dir.glob("*.parquet") if p.name not in SKIP_FILES)
    v2 = score_machines(files)
    ae = read(root / AE_DIR, ["entity_id", "window_start", "ae_flag", "label"])
    df = ae.merge(v2.drop(columns=["label", "is_test"]), on=["entity_id", "window_start"], how="left")
    print(f"{len(files)} machines, AE test windows {len(ae)}, without a v2 window {int(df['k2_t4'].isna().sum())}")
    for c in [setting(k, t) for k in KS for t in TS] + ["ae_flag"]:
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    grid = run_grid(df)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print("\nGRID (t1_ = Tier 1 v2 alone, or_ = ae_flag OR Tier 1 v2)")
        print(grid.to_string(index=False))
    best = select(grid)
    if best is None:
        print(f"\nSELECTION: no setting has Tier 1 v2 false alarms <= {MAX_FA_DAY:g} per machine per day")
        return 0
    c = setting(int(best["k"]), float(best["t"]))
    ae_flag, v2 = df["ae_flag"].to_numpy(dtype=bool), df[c].to_numpy(dtype=bool)
    table = pd.DataFrame({"AE only": alarm_metrics(df, ae_flag), f"AE OR Tier 1 v2 ({c})": alarm_metrics(df, ae_flag | v2)}).T
    print(f"\nSELECTION: {c}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
