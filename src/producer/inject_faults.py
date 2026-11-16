"""Inject labelled faults into a slice of the Alibaba raw layer, replay it and report recall and detection delay.

Run from the repo root:
    python src/producer/inject_faults.py                                  # inject, replay, detect, report
    python src/producer/inject_faults.py --machines 12 --hours 24 --per-type 3 --seed 7
    python src/producer/inject_faults.py --scores data/scores/source=alibaba --flag-col is_anomaly   # report only, real scores

1. Slice: the first --machines machine_ids (sorted) of data/raw/source=alibaba, rows from --start-hours to
   --start-hours + --hours after 2026-01-01T00:00:00Z. Nothing in data/raw is changed.
2. Faults (one per machine, --per-type machines per type, random machines and 5-minute-aligned start times, all after
   the first --clean-hours so there is a fault-free baseline; every injected row gets label = 1, all other rows 0):
     spike  one random metric x 3 for 5 minutes
     drift  one random metric +2 % per minute for 30 minutes (x (1 + 0.02 * minutes since the start))
     break  net_in x 0.1 for 15 minutes while cpu is untouched, placed where cpu is high (top quarter of 5-minute means)
3. Output under --out-root (default data/inject): data/raw/source=alibaba/<machine>.parquet (a raw layer that
   src/producer/replay.py reads with --root <out-root>) and injection_log.json (type, machine, metric, start/end, rows).
4. Replay: the injected raw layer goes through src/producer/replay.py (--speedup 0, event_time order) into a capture
   producer instead of Kafka, so the detector sees exactly the JSON messages Kafka would carry.
5. Detect: Tier 1 style robust z-score on the replayed messages (5-minute window means, median/MAD per machine-metric
   from the clean part, score = max z, flag = score > --threshold, metrics with MAD 0 skipped), evaluated on windows
   from the end of the clean part. With --scores the flags of a real scores layer (entity_id, window_start, --flag-col)
   are used instead.
6. Report per fault type with src/detection/evaluate.py: events, detected, recall (share of faults with at least one
   flagged window), detection delay in minutes (fault start to the first flagged window, detected faults only), and
   false alarms per day on the same machines.
"""
import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.detection import evaluate as E  # noqa: E402
from src.producer import replay as R  # noqa: E402

SOURCE = "alibaba"
METRICS = S.ALIBABA_METRICS
BASE_S = int(datetime.fromisoformat(S.TIME_BASE_UTC.replace("Z", "+00:00")).timestamp())
WINDOW_S = 300                                  # 5-minute windows, as everywhere in the pipeline
FAULT_TYPES = ("spike", "drift", "break")
SPIKE_FACTOR, SPIKE_MINUTES = 3.0, 5
DRIFT_RATE, DRIFT_MINUTES = 0.02, 30
BREAK_FACTOR, BREAK_MINUTES, BREAK_CPU_QUANTILE = 0.1, 15, 0.75
DURATION_S = {"spike": SPIKE_MINUTES * 60, "drift": DRIFT_MINUTES * 60, "break": BREAK_MINUTES * 60}
MIN_COVERAGE = 0.9                              # a metric is injectable if present in this share of a machine's rows
MIN_ROW_SHARE = 0.8                             # a fault interval needs this share of the rows a gap-free machine would have
MIN_BASELINE_WINDOWS = 12                       # a machine needs this many 5-minute windows (1 h) before the first fault
MAD_TO_SIGMA = 1.4826
THRESHOLD = 3.0
LOG_NAME = "injection_log.json"
DEFAULT_OUT = "data/inject"


def iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


# ---------------------------------------------------------------- slice

def raw_dir(root: Path) -> Path:
    return Path(root) / S.PATHS["raw"] / f"source={SOURCE}"


def read_slice(root: Path, machines: int, start_hours: float, hours: float) -> dict[str, pd.DataFrame]:
    """entity_id -> DataFrame(ts [epoch s], one float column per metric (NaN = missing), label) for the first N machines."""
    files = [f for f in sorted(raw_dir(root).glob("*.parquet")) if f.name != "fake.parquet"]
    if not files:
        raise FileNotFoundError(f"no Parquet files in {raw_dir(root)}")
    ids = sorted(set().union(*(set(pq.read_table(f, columns=["entity_id"]).column("entity_id").to_pylist())
                               for f in files)))[:machines]
    lo, hi = BASE_S + start_hours * 3600, BASE_S + (start_hours + hours) * 3600
    rows: dict[str, list] = {e: [] for e in ids}
    for f in files:
        table = pq.read_table(f, columns=["entity_id", "event_time", "metrics"])
        for entity, when, metrics in zip(table.column("entity_id").to_pylist(), table.column("event_time").to_pylist(),
                                         table.column("metrics").to_pylist()):
            if entity in rows and lo <= when.timestamp() < hi:
                rows[entity].append({"ts": int(when.timestamp()), **dict(metrics)})
    out = {}
    for entity, items in rows.items():
        if items:
            df = pd.DataFrame(items).reindex(columns=["ts", *METRICS]).sort_values("ts").reset_index(drop=True)
            df["label"] = 0
            out[entity] = df
    if not out:
        raise ValueError("the slice is empty: check --machines, --start-hours and --hours")
    return out


# ---------------------------------------------------------------- planning and injection

def usable_metrics(df: pd.DataFrame) -> list[str]:
    """Metrics present in most rows and not constantly zero (x 3 of zero would change nothing)."""
    return [m for m in METRICS if df[m].notna().mean() >= MIN_COVERAGE and df[m].median() > 0]


def _starts(df: pd.DataFrame, clean_end: int, duration_s: int) -> list[int]:
    """5-minute-aligned starts after clean_end whose interval has enough rows (real machines have gaps in time)."""
    if df.loc[df["ts"] < clean_end, "ts"].floordiv(WINDOW_S).nunique() < MIN_BASELINE_WINDOWS:
        return []
    ts = df["ts"].to_numpy()
    step = float(np.median(np.diff(ts))) or 1.0
    first = math.ceil(clean_end / WINDOW_S) * WINDOW_S
    last = (int(ts.max()) + 1 - duration_s) // WINDOW_S * WINDOW_S
    starts = np.arange(first, last + 1, WINDOW_S)
    rows = np.searchsorted(ts, starts + duration_s) - np.searchsorted(ts, starts)
    return [int(s) for s in starts[rows >= MIN_ROW_SHARE * duration_s / step]]


def _high_cpu_starts(df: pd.DataFrame, starts: list[int]) -> list[int]:
    """Starts whose 15 minutes all have a cpu window mean in the top quarter; else the single best start."""
    cpu = df.groupby(df["ts"] // WINDOW_S * WINDOW_S)["cpu"].mean()
    high = cpu.quantile(BREAK_CPU_QUANTILE)
    n = BREAK_MINUTES * 60 // WINDOW_S
    lowest = {s: min(cpu.get(s + i * WINDOW_S, -math.inf) for i in range(n)) for s in starts}
    good = [s for s in starts if lowest[s] >= high]
    return good or [max(starts, key=lambda s: lowest[s])]


def plan_faults(series: dict[str, pd.DataFrame], clean_end: int, per_type: int, rng: random.Random) -> list[dict]:
    """One fault per machine: per_type machines for each type, random metric and start (5-minute aligned)."""
    usable = {e: usable_metrics(df) for e, df in series.items()}
    room = {e: {t: _starts(df, clean_end, DURATION_S[t]) for t in FAULT_TYPES} for e, df in series.items()}
    free = sorted(series)
    plan = []
    for kind in ("break", "spike", "drift"):                    # break first: it is the most demanding
        ok = [e for e in free if room[e][kind] and usable[e]
              and (kind != "break" or ("cpu" in usable[e] and "net_in" in usable[e]))]
        if len(ok) < per_type:
            raise ValueError(f"need {per_type} machines for '{kind}' faults, only {len(ok)} are left; use more "
                             f"--machines, more --hours or a smaller --clean-hours / --per-type")
        for entity in sorted(rng.sample(ok, per_type)):
            free.remove(entity)
            starts = _high_cpu_starts(series[entity], room[entity][kind]) if kind == "break" else room[entity][kind]
            start = rng.choice(starts)
            metric = "net_in" if kind == "break" else rng.choice(usable[entity])
            plan.append({"fault_id": f"{kind}-{entity}", "type": kind, "entity_id": entity, "metric": metric,
                         "start_s": start, "end_s": start + DURATION_S[kind]})
    return sorted(plan, key=lambda f: (f["start_s"], f["entity_id"]))


def apply_fault(df: pd.DataFrame, fault: dict) -> int:
    """Change the metric in place inside [start, end), set label = 1 there; returns the rows covered."""
    inside = (df["ts"] >= fault["start_s"]) & (df["ts"] < fault["end_s"])
    metric = fault["metric"]
    if fault["type"] == "spike":
        df.loc[inside, metric] *= SPIKE_FACTOR
    elif fault["type"] == "drift":
        minutes = (df.loc[inside, "ts"] - fault["start_s"]) / 60.0
        df.loc[inside, metric] *= 1.0 + DRIFT_RATE * minutes
    else:
        df.loc[inside, metric] *= BREAK_FACTOR
    df.loc[inside, "label"] = 1
    return int(inside.sum())


def describe(fault: dict, n_rows: int) -> dict:
    params = {"spike": {"factor": SPIKE_FACTOR}, "drift": {"rate_per_minute": DRIFT_RATE},
              "break": {"factor": BREAK_FACTOR, "cpu": "unchanged, high"}}[fault["type"]]
    return {"fault_id": fault["fault_id"], "type": fault["type"], "entity_id": fault["entity_id"],
            "metric": fault["metric"], "start_time": iso(fault["start_s"]), "end_time": iso(fault["end_s"]),
            "n_rows": n_rows, "params": params}


def write_raw(out_root: Path, series: dict[str, pd.DataFrame]) -> None:
    folder = raw_dir(out_root)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.parquet"):
        old.unlink()
    for entity, df in series.items():
        values = df[METRICS].to_numpy(dtype=float)
        metrics = [[(m, float(v)) for m, v in zip(METRICS, row) if not math.isnan(v)] for row in values]
        times = pa.array(df["ts"].to_numpy(dtype=np.int64) * 1_000_000, type=pa.int64()).cast(pa.timestamp("us", tz="UTC"))
        table = pa.table({"entity_id": pa.array([entity] * len(df), pa.string()), "event_time": times,
                          "metrics": pa.array(metrics, S.arrow_schema("raw").field("metrics").type),
                          "attrs": pa.array([[] for _ in range(len(df))], S.arrow_schema("raw").field("attrs").type),
                          "label": pa.array(df["label"].to_numpy(dtype=np.int32), pa.int32())},
                         schema=S.arrow_schema("raw"))
        pq.write_table(table, folder / f"{entity}.parquet")


def inject(root: Path = ROOT, out_root: Path | None = None, machines: int = 12, start_hours: float = 0,
           hours: float = 24, clean_hours: float = 8, per_type: int = 3, seed: int = 7) -> dict:
    """Slice, inject, write the raw layer and the log under out_root; returns the log."""
    out_root = Path(out_root or Path(root) / DEFAULT_OUT)
    series = read_slice(root, machines, start_hours, hours)
    clean_end = int(BASE_S + (start_hours + clean_hours) * 3600)
    plan = plan_faults(series, clean_end, per_type, random.Random(seed))
    faults = [describe(f, apply_fault(series[f["entity_id"]], f)) for f in plan]
    write_raw(out_root, series)
    log = {"source": SOURCE, "seed": seed, "machines": sorted(series), "slice_start": iso(BASE_S + start_hours * 3600),
           "slice_end": iso(BASE_S + (start_hours + hours) * 3600), "clean_end": iso(clean_end),
           "rows": int(sum(len(df) for df in series.values())), "faults": faults}
    (out_root / LOG_NAME).write_text(json.dumps(log, indent=2) + "\n")
    print(f"injected {len(faults)} faults into {len(series)} machines ({log['rows']} rows) -> {out_root}")
    return log


# ---------------------------------------------------------------- replay and detection

class CaptureProducer:
    """Stands in for the Kafka producer: keeps every (topic, key, value) that replay.py sends."""

    def __init__(self):
        self.messages = []

    def produce(self, topic, key=None, value=None, on_delivery=None):
        self.messages.append((topic, key, value))
        if on_delivery is not None:
            on_delivery(None, None)

    def poll(self, timeout=0):
        return 0

    def flush(self, timeout=None):
        return 0


def replay_records(out_root: Path) -> list[dict]:
    """The injected raw layer replayed through src/producer/replay.py, as the JSON records of the topic."""
    producer = CaptureProducer()
    code = R.main(["--source", SOURCE, "--speedup", "0", "--root", str(out_root)], producer=producer)
    if code:
        raise RuntimeError("replay failed")
    return [json.loads(value) for _, _, value in producer.messages]


def to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"entity_id": r["entity_id"], "ts": int(datetime.fromisoformat(r["event_time"].replace("Z", "+00:00")).timestamp()),
                        "label": r["label"] or 0, **r["metrics"]} for r in records]).reindex(
        columns=["entity_id", "ts", "label", *METRICS])
    df["window"] = df["ts"] // WINDOW_S * WINDOW_S
    return df


def window_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["entity_id", "window"], as_index=False)["label"].max()


def detect(df: pd.DataFrame, clean_end: int, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Robust z-score per machine-metric on 5-minute window means; baseline = windows before clean_end."""
    means = df.groupby(["entity_id", "window"])[METRICS].mean()
    base = means[means.index.get_level_values("window") < clean_end]
    median = base.groupby(level="entity_id").median()
    mad = base.sub(median, level="entity_id").abs().groupby(level="entity_id").median()
    scale = (MAD_TO_SIGMA * mad).where(mad > 0)                  # MAD 0: not scored, as in tier1_zscore
    test = means[means.index.get_level_values("window") >= clean_end]
    z = (test - median.reindex(test.index.get_level_values("entity_id")).to_numpy()).abs() \
        / scale.reindex(test.index.get_level_values("entity_id")).to_numpy()
    out = test.index.to_frame(index=False)
    out["score"] = z.max(axis=1, skipna=True).fillna(0.0).to_numpy()
    out["flag"] = out["score"] > threshold
    return out


def external_flags(path: Path, flag_col: str) -> pd.DataFrame:
    table = ds.dataset(path, format="parquet", partitioning="hive").to_table(columns=["entity_id", "window_start", flag_col])
    df = table.to_pandas()
    seconds = (pd.to_datetime(df["window_start"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    out = pd.DataFrame({"entity_id": df["entity_id"], "window": seconds // WINDOW_S * WINDOW_S,
                        "flag": df[flag_col].fillna(False).astype(bool)})
    return out.groupby(["entity_id", "window"], as_index=False)["flag"].max()


# ---------------------------------------------------------------- report

def _sequences(windows: pd.DataFrame, flags: pd.DataFrame, machines: list[str], clean_end: int):
    """y_true / y_pred over the windows of the given machines, one 0 between machines (as in tier1_zscore)."""
    merged = windows[windows["window"] >= clean_end].merge(flags[["entity_id", "window", "flag"]], how="left",
                                                           on=["entity_id", "window"])
    merged["flag"] = merged["flag"].astype("boolean").fillna(False).astype(bool)
    y_true, y_pred = [], []
    for entity in machines:
        own = merged[merged["entity_id"] == entity].sort_values("window")
        y_true += own["label"].astype(int).tolist() + [0]
        y_pred += own["flag"].astype(int).tolist() + [0]
    return np.array(y_true), np.array(y_pred)


def report(log: dict, windows: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Per fault type: events, detected, recall, mean/median detection delay (minutes), false alarms per day."""
    clean_end = int(datetime.fromisoformat(log["clean_end"].replace("Z", "+00:00")).timestamp())
    rows = []
    groups = [(t, [f for f in log["faults"] if f["type"] == t]) for t in FAULT_TYPES] + [("all", log["faults"])]
    for name, faults in groups:
        if not faults:
            continue
        y_true, y_pred = _sequences(windows, flags, sorted({f["entity_id"] for f in faults}), clean_end)
        c = E.composite_metrics(y_true, y_pred)
        delays = [d * WINDOW_S / 60 for d in E.detection_delays(y_true, y_pred)]
        rows.append({"fault_type": name, "events": c["events"], "detected": c["events_detected"],
                     "recall": c["event_recall"],
                     "mean_delay_min": float(np.mean(delays)) if delays else float("nan"),
                     "median_delay_min": float(np.median(delays)) if delays else float("nan"),
                     "false_alarms/day": E.false_alarms_per_day(y_true, y_pred, WINDOW_S // 60)})
    return pd.DataFrame(rows).set_index("fault_type")


def run(root: Path = ROOT, out_root: Path | None = None, threshold: float = THRESHOLD, scores: Path | None = None,
        flag_col: str = "is_anomaly", **inject_args) -> tuple[dict, pd.DataFrame]:
    out_root = Path(out_root or Path(root) / DEFAULT_OUT)
    log = inject(root, out_root, **inject_args)
    df = to_frame(replay_records(out_root))
    clean_end = int(datetime.fromisoformat(log["clean_end"].replace("Z", "+00:00")).timestamp())
    flags = external_flags(scores, flag_col) if scores else detect(df, clean_end, threshold)
    return log, report(log, window_labels(df), flags)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/raw/source=alibaba")
    ap.add_argument("--out-root", type=Path, default=None, help=f"where the injected raw layer and the log go (default {DEFAULT_OUT})")
    ap.add_argument("--machines", type=int, default=12, help="first N machine_ids (sorted) of the slice")
    ap.add_argument("--start-hours", type=float, default=0, help="slice start, hours after 2026-01-01T00:00:00Z")
    ap.add_argument("--hours", type=float, default=24, help="slice length in hours")
    ap.add_argument("--clean-hours", type=float, default=8, help="fault-free start of the slice (detector baseline)")
    ap.add_argument("--per-type", type=int, default=3, help="faults (= machines) per fault type")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="z-score threshold of the built-in detector")
    ap.add_argument("--scores", type=Path, default=None, help="use the flags of this scores layer instead of the built-in detector")
    ap.add_argument("--flag-col", default="is_anomaly", help="flag column of --scores")
    a = ap.parse_args()
    log, table = run(a.root, a.out_root, a.threshold, a.scores, a.flag_col, machines=a.machines,
                     start_hours=a.start_hours, hours=a.hours, clean_hours=a.clean_hours, per_type=a.per_type, seed=a.seed)
    print(f"log: {(a.out_root or a.root / DEFAULT_OUT) / LOG_NAME}")
    print(table.to_string(float_format=lambda x: f"{x:.2f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
