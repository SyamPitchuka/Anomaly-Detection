"""Streaming score job: SMD telemetry -> 5-minute windows scored with the offline models -> scores layer.

Run from the repo root:
    python src/streaming/score_job.py --input kafka            # telemetry.smd, runs until stopped
    python src/streaming/score_job.py --input kafka --once     # what is in Kafka now, then stop
    python src/streaming/score_job.py --input files --machines 2 --once   # replay data/raw/source=smd
    python src/streaming/score_job.py --input files --once --reset        # wipe output + checkpoint first
    python src/streaming/score_job.py --input files --once --tier1 v1     # also fill tier1_score/tier1_flag
    python src/streaming/score_job.py --input files --once --master local[2] --shuffle-partitions 4
    python src/streaming/score_job.py --input files --once --files-per-trigger 8   # 8 machines per batch

Input: Kafka topic telemetry.smd parsed with contracts.schemas.TELEMETRY_JSON_DDL, or a readStream of the
raw-layer Parquet files (one file per trigger, or --files-per-trigger N; fake.parquet skipped). 10-minute
watermark on event_time.

Windows and scoring happen in one applyInPandasWithState keyed by entity_id (GroupStateTimeout.NoTimeout).
The windows are built inside it rather than by a Spark window aggregation for two reasons: Spark 3.5 does
not allow applyInPandasWithState after a streaming aggregation in append mode, and with the file replay
each micro-batch is one whole machine whose times overlap the previous machine's, so a global watermark
would drop every machine after the first one as late. Here each machine has its own watermark:
    a window [s, s + 5 min) is complete once s + 5 min <= (max event_time seen for that machine) - 10 min,
    rows of a window that was already emitted are dropped as late,
which is the per-machine version of Spark's append-mode semantics. The newest windows of a machine stay
open until later rows of that machine arrive.

A window = mean and max of each metric m00..m37 over its rows (missing values skipped), split = test if
any row is test, label = max (test: null -> 0; train: null), exactly as src/detection/autoencoder.py.
State per machine: open windows (count, sum, max per metric, test, label), the last `recenter` (24)
windows of model inputs and the last `adaptive` (288) ae scores, so the AE v2 logic runs on the stream
with the same functions as the batch job (recenter_centers, standardise, score, adaptive_limits, flags).
Train windows are scored too (they feed the rolling median and the score history) but only test
windows are written.

Hooks, each optional:
- Tier 1 (--tier1 v1, default off): models/tier1_smd/baselines_v1.parquet (entity_id, metric, median, mad)
  -> see load_tier1_baselines() and tier1_scores(). Off -> tier1_score / tier1_flag null. Tier 1 is a
  baseline and explainer only: it never changes is_anomaly (see docs/results/tier1_stream.md). The switch
  reaches the Python workers through the SCORE_JOB_TIER1 environment variable, set before Spark starts.
- Tier 2 / 2b: if src/detection/stream_baselines.py exists, score_baselines(pdf, models) is called on the
  emitted windows. pdf has entity_id, window_start (int64 microseconds UTC), split and <metric>_mean,
  <metric>_max for m00..m37 (NaN = no value); models is this job's model dict, with
  models["baselines"] = stream_baselines.load_models() if that function exists (else None). It must return
  a DataFrame aligned with pdf that has tier2_score, tier2_flag and pca_score. Missing module -> null.

Output: test windows -> data/scores_stream/source=smd/batch-<id>.parquet (one file per micro-batch, so a
replayed batch overwrites its own file), columns of contracts.schemas.LAYERS['scores'],
is_anomaly = coalesce(ae_flag, false) (AE only). Checkpoint: checkpoints/score/source=smd.
With --input kafka, rows with is_anomaly also go to the anomalies topic (key entity_id, value JSON).

Latency side output (--input kafka only; not a contract layer, LAYERS['scores'] is unchanged): every
emitted window of the micro-batch, train and test, -> data/latency_stream/source=smd/batch-<id>.parquet
with entity_id, window_start, max_ingest_ts_ms (newest attrs.ingest_ts of the records in the window),
batch_max_ingest_ts_ms (newest attrs.ingest_ts of that machine's records in the micro-batch that closed
the window), emit_ts_ms (wall clock once the batch's scores file and anomalies are written) and batch_id.
The state keeps the newest ingest time per open window for this. src/streaming/latency_report.py reads it.
Each query progress is also appended to data/latency_stream/progress.jsonl.

Stopping without Ctrl+C (for scripted runs): --max-seconds S stops the query after S wall seconds,
--idle-stop S once rows were seen and no micro-batch had input rows for S seconds.

Diagnostics: each batch line is followed by the batch's durationMs and the time of each foreachBatch step.
With SCORE_JOB_WORKER_LOG=<file>, every process that imports this module (driver and Python workers)
appends JSON lines: module_import (import time, process age), load_models (seconds) and one score_group
line per machine and batch. On Windows, Spark does not reuse Python workers (no daemon), so every task of
every batch is a new process that imports torch and loads the models again; see
docs/results/e2e_latency_notes.md.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_IMPORT_T0 = time.time()   # for the worker_log 'module_import' event

# On Windows, Spark must start workers with this same venv Python (as in tools/check_spark_kafka.py).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
# One math thread per Python worker: Spark already runs one worker per core, so OpenMP / MKL / OpenBLAS
# pools sized to every hardware thread would oversubscribe the CPU. Set before the SparkSession exists,
# so the local-mode Python workers inherit them; load_models() also calls torch.set_num_threads(1).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Python workers import this module by name (src.streaming.score_job), so the model cache below lives
# once per worker process instead of being shipped with every task.
if str(ROOT) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.detection import autoencoder as AE  # noqa: E402

SOURCE = "smd"
METRICS = S.SMD_METRICS
K = len(METRICS)
WINDOW_US = AE.WINDOW_US
WATERMARK = "10 minutes"
WATERMARK_US = 10 * 60 * 1_000_000
TRIGGER = "30 seconds"

AE_DIR = ROOT / S.MODELS["ae"].replace("<source>", SOURCE)
TIER1_PATH = ROOT / "models" / f"tier1_{SOURCE}" / "baselines_v1.parquet"
TIER1_MODES = ("off", "v1")
TIER1_ENV = "SCORE_JOB_TIER1"   # how --tier1 reaches the Python workers (inherited from the driver)
TIER1_MAD_SCALE = 1.4826
TIER1_Z = 3.0
BASELINES_MODULE = ROOT / "src" / "detection" / "stream_baselines.py"
OUTPUT_DIR = ROOT / "data" / "scores_stream" / f"source={SOURCE}"
CHECKPOINT_DIR = ROOT / "checkpoints" / "score" / f"source={SOURCE}"
LATENCY_ROOT = ROOT / "data" / "latency_stream"
LATENCY_DIR = LATENCY_ROOT / f"source={SOURCE}"
PROGRESS_LOG = LATENCY_ROOT / "progress.jsonl"
LATENCY_SCHEMA = pa.schema([("entity_id", pa.string()), ("window_start", pa.timestamp("us", tz="UTC")),
                            ("max_ingest_ts_ms", pa.int64()), ("batch_max_ingest_ts_ms", pa.int64()),
                            ("emit_ts_ms", pa.int64()), ("batch_id", pa.int64())])

_DDL_TYPES = {"string": "STRING", "ts": "TIMESTAMP", "double": "DOUBLE", "long": "BIGINT", "int": "INT",
              "bool": "BOOLEAN", "map_sd": "MAP<STRING,DOUBLE>", "map_ss": "MAP<STRING,STRING>",
              "list_s": "ARRAY<STRING>"}
RAW_DDL = ", ".join(f"{name} {_DDL_TYPES[code]}" for name, code in S.LAYERS["raw"].items())

# rows handed to the stateful function: one column per metric (NaN = missing)
OUTPUT_DDL = ("entity_id STRING, ws_us BIGINT, split STRING, label DOUBLE, ae_score DOUBLE, ae_flag BOOLEAN, "
              "top_metrics ARRAY<STRING>, tier1_score DOUBLE, tier1_flag BOOLEAN, tier2_score DOUBLE, "
              "tier2_flag BOOLEAN, pca_score DOUBLE, max_ingest_ms DOUBLE, batch_ingest_ms DOUBLE")
OUTPUT_COLUMNS = [c.split()[0] for c in OUTPUT_DDL.split(", ")]
STATE_FIELDS = ("max_us", "emitted_end", "open_starts", "open_cnt", "open_sum", "open_max", "open_test",
                "open_label", "open_ingest", "hist_x", "hist_scores")
STATE_DDL = ("max_us BIGINT, emitted_end BIGINT, open_starts ARRAY<BIGINT>, open_cnt ARRAY<DOUBLE>, "
             "open_sum ARRAY<DOUBLE>, open_max ARRAY<DOUBLE>, open_test ARRAY<BOOLEAN>, "
             "open_label ARRAY<DOUBLE>, open_ingest ARRAY<DOUBLE>, hist_x ARRAY<DOUBLE>, "
             "hist_scores ARRAY<DOUBLE>")


# ---------------------------------------------------------------- models (loaded once per process)

def ae_bundle(model, scaler: dict, threshold: dict) -> dict:
    """Everything the AE v2 scoring needs, from the model and the contents of scaler.json / threshold.json."""
    aggs = AE.INPUTS[scaler["inputs"]]
    recenter, adaptive = scaler.get("recenter"), threshold.get("adaptive")
    return {"model": model, "scalers": scaler["entities"], "metrics": scaler["metrics"], "aggs": aggs,
            "feature_metrics": [m for _ in aggs for m in scaler["metrics"]],
            "recenter": recenter["windows"] if recenter else 0,
            "adaptive": adaptive["windows"] if adaptive else 0,
            "adaptive_percentile": adaptive["percentile"] if adaptive else AE.ADAPTIVE_PERCENTILE,
            "thresholds": threshold, "mode": threshold.get("mode", "global")}


def load_tier1_baselines(path: Path = TIER1_PATH, metrics: list[str] = METRICS) -> dict | None:
    """Tier 1 input, the only place that knows its format: entity_id -> (median, mad), arrays over
    `metrics` (NaN where a metric has no baseline). None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["entity_id", "metric", "median", "mad"])
    col = df["metric"].map({m: i for i, m in enumerate(metrics)})
    df = df[col.notna()].assign(col=col[col.notna()].astype(int))
    out = {}
    for entity, g in df.groupby("entity_id"):
        median, mad = np.full(len(metrics), np.nan), np.full(len(metrics), np.nan)
        median[g["col"].to_numpy()] = g["median"].to_numpy(dtype=np.float64)
        mad[g["col"].to_numpy()] = g["mad"].to_numpy(dtype=np.float64)
        out[entity] = (median, mad)
    return out


def tier1_scores(entity: str, means: np.ndarray, baselines: dict | None) -> tuple[np.ndarray, list]:
    """Robust z per window: max over metrics of |mean - median| / (1.4826 mad), metrics with mad 0 (or no
    baseline / no value) skipped; flag = score > 3. Null when the machine or every metric is skipped."""
    n = len(means)
    if baselines is None or entity not in baselines:
        return np.full(n, np.nan), [None] * n
    median, mad = baselines[entity]
    usable = np.isfinite(mad) & (mad > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(means[:, usable] - median[usable]) / (TIER1_MAD_SCALE * mad[usable])
    valid = np.isfinite(z).any(axis=1)
    score = np.where(valid, np.where(np.isfinite(z), z, -np.inf).max(axis=1, initial=-np.inf), np.nan)
    return score, [bool(s > TIER1_Z) if v else None for s, v in zip(score, valid)]


def tier1_mode() -> str:
    """--tier1 as seen by this process: SCORE_JOB_TIER1, default off."""
    mode = os.environ.get(TIER1_ENV, "off")
    if mode not in TIER1_MODES:
        raise ValueError(f"{TIER1_ENV}={mode!r}, expected one of {TIER1_MODES}")
    return mode


def load_models(ae_dir: Path = AE_DIR, tier1_path: Path = TIER1_PATH, tier1: str | None = None) -> dict:
    """tier1: 'off' or 'v1' (None -> tier1_mode()). Tier 1 baselines are loaded only for 'v1'."""
    import torch
    tier1 = tier1_mode() if tier1 is None else tier1
    torch.set_num_threads(1)  # runs on each worker: several Python workers already run in parallel
    models = {"ae": ae_bundle(*AE.load_model(ae_dir)),
              "tier1": load_tier1_baselines(tier1_path) if tier1 == "v1" else None,
              "baselines_module": None, "baselines": None}
    if BASELINES_MODULE.exists():
        from src.detection import stream_baselines
        models["baselines_module"] = stream_baselines
        if hasattr(stream_baselines, "load_models"):
            models["baselines"] = stream_baselines.load_models(history_windows=models["ae"]["recenter"])
    return models


_MODELS: dict = {}
WORKER_LOG_ENV = "SCORE_JOB_WORKER_LOG"   # diagnostics: JSON-lines file each Python worker appends to


def worker_log(event: str, **fields) -> None:
    """Append {ts_ms, pid, event, ...} to $SCORE_JOB_WORKER_LOG when set (off by default)."""
    path = os.environ.get(WORKER_LOG_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts_ms": int(time.time() * 1000), "pid": os.getpid(), "event": event,
                                **fields}) + "\n")


def get_models() -> dict:
    """Lazy per-process cache: the first window a worker scores loads the models, later ones reuse them."""
    if not _MODELS:
        t0 = time.perf_counter()
        _MODELS.update(load_models())
        worker_log("load_models", seconds=round(time.perf_counter() - t0, 3))
    return _MODELS


# ---------------------------------------------------------------- per-machine state

def new_state(n_in: int) -> dict:
    return {"max_us": None, "emitted_end": None, "open_starts": np.empty(0, np.int64),
            "open_cnt": np.zeros((0, K)), "open_sum": np.zeros((0, K)), "open_max": np.zeros((0, K)),
            "open_test": np.empty(0, bool), "open_label": np.empty(0), "open_ingest": np.empty(0),
            "hist_x": np.zeros((0, n_in)),
            "hist_scores": np.empty(0)}


def encode_state(st: dict) -> tuple:
    """dict of numpy arrays -> tuple of plain Python values for GroupState.update (2-D arrays flattened)."""
    def plain(v):
        return v if v is None else (np.asarray(v).ravel().tolist() if isinstance(v, np.ndarray) else int(v))
    return tuple(plain(st[f]) for f in STATE_FIELDS)


def decode_state(values, n_in: int) -> dict:
    v = dict(zip(STATE_FIELDS, values))
    return {"max_us": v["max_us"], "emitted_end": v["emitted_end"],
            "open_starts": np.asarray(v["open_starts"], dtype=np.int64),
            **{f: np.asarray(v[f], dtype=np.float64).reshape(-1, K) for f in ("open_cnt", "open_sum", "open_max")},
            "open_test": np.asarray(v["open_test"], dtype=bool),
            "open_label": np.asarray(v["open_label"], dtype=np.float64),
            "open_ingest": np.asarray(v["open_ingest"], dtype=np.float64),
            "hist_x": np.asarray(v["hist_x"], dtype=np.float64).reshape(-1, n_in),
            "hist_scores": np.asarray(v["hist_scores"], dtype=np.float64)}


def _add_rows(st: dict, rows: pd.DataFrame) -> int:
    """Fold raw rows into the open windows; returns the number of late rows dropped."""
    event_us = rows["event_us"].to_numpy(dtype=np.int64)
    on_time = event_us >= (st["emitted_end"] if st["emitted_end"] is not None else np.iinfo(np.int64).min)
    rows, event_us = rows[on_time], event_us[on_time]
    if len(rows):
        st["max_us"] = int(max(event_us.max(), st["max_us"] if st["max_us"] is not None else event_us.max()))
        vals = rows[METRICS].to_numpy(dtype=np.float64)
        ws = event_us - np.mod(event_us, WINDOW_US)
        new = pd.DataFrame(np.hstack([~np.isnan(vals), np.nan_to_num(vals), vals]))
        new["start"], new["test"] = ws, (rows["split"] == "test").to_numpy()
        new["label"] = rows["label"].to_numpy(dtype=np.float64)
        new["ingest"] = (rows["ingest_ms"].to_numpy(dtype=np.float64) if "ingest_ms" in rows
                         else np.full(len(rows), np.nan))
        old = pd.DataFrame(np.hstack([st["open_cnt"], st["open_sum"], st["open_max"]]))
        old["start"], old["test"], old["label"] = st["open_starts"], st["open_test"], st["open_label"]
        old["ingest"] = st["open_ingest"]
        agg = {c: ("max" if isinstance(c, int) and c >= 2 * K else "sum") for c in range(3 * K)}
        g = pd.concat([old, new], ignore_index=True).groupby("start", sort=True).agg(
            {**agg, "test": "max", "label": "max", "ingest": "max"})
        block = g[list(range(3 * K))].to_numpy(dtype=np.float64)
        st["open_cnt"], st["open_sum"], st["open_max"] = block[:, :K], block[:, K:2 * K], block[:, 2 * K:]
        st["open_starts"] = g.index.to_numpy(dtype=np.int64)
        st["open_test"] = g["test"].to_numpy(dtype=bool)
        st["open_label"] = g["label"].to_numpy(dtype=np.float64)
        st["open_ingest"] = g["ingest"].to_numpy(dtype=np.float64)
    return int((~on_time).sum())


def _pop_complete(st: dict) -> dict | None:
    """Remove and return the windows whose end is at or before the machine's watermark."""
    if st["max_us"] is None:
        return None
    done = st["open_starts"] + WINDOW_US <= st["max_us"] - WATERMARK_US
    if not done.any():
        return None
    out = {f: st[f][done] for f in ("open_starts", "open_cnt", "open_sum", "open_max", "open_test", "open_label",
                                    "open_ingest")}
    for f in out:
        st[f] = st[f][~done]
    st["emitted_end"] = int(out["open_starts"].max() + WINDOW_US)
    return out


def _window_values(w: dict) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Window means/maxes (NaN = no value), split and label as build_windows() defines them."""
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(w["open_cnt"] > 0, w["open_sum"] / w["open_cnt"], np.nan)
    maxes = np.where(w["open_cnt"] > 0, w["open_max"], np.nan)
    split = np.where(w["open_test"], "test", "train")
    label = np.where(w["open_test"], np.nan_to_num(w["open_label"], nan=0.0), np.nan)
    return means, maxes, split.tolist(), label


def score_windows(entity: str, means: np.ndarray, maxes: np.ndarray, split: list, st: dict,
                  ae: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """AE v2 on new windows of one machine (in time order) with the batch functions of autoencoder.py,
    using the machine's window/score history in st, which is updated. Returns (ae_score, ae_flag, top)."""
    n = len(means)
    X = np.hstack([{"mean": means, "max": maxes}[a] for a in ae["aggs"]])
    hist_x, hist_s = st["hist_x"], st["hist_scores"]
    if entity in ae["scalers"]:
        meta = pd.DataFrame({"entity_id": [entity] * n, "split": split})
        centers = None
        if ae["recenter"]:
            X_all = np.vstack([hist_x, X])
            meta_all = pd.DataFrame({"entity_id": [entity] * len(X_all), "split": ["train"] * len(hist_x) + split})
            centers = AE.recenter_centers(meta_all, X_all, ae["scalers"], ae["recenter"])[len(hist_x):]
        Z, mask = AE.standardise(meta, X, ae["scalers"], ae["feature_metrics"], centers)
        ae_score, top = AE.score(ae["model"], Z, mask, ae["metrics"])
        limit = None
        if ae["adaptive"]:
            s_all = np.concatenate([hist_s, ae_score])
            limit = AE.adaptive_limits(pd.DataFrame({"entity_id": [entity] * len(s_all)}), s_all,
                                       ae["adaptive"], ae["adaptive_percentile"])[len(hist_s):]
        ae_flag = AE.flags(meta, ae_score, ae["thresholds"], ae["mode"], limit)
    else:  # machine unknown to the model: no AE score
        ae_score, ae_flag, top = np.full(n, np.nan), np.zeros(n, bool), [[] for _ in range(n)]
    keep_x, keep_s = ae["recenter"], ae["adaptive"]
    st["hist_x"] = np.vstack([hist_x, X])[-keep_x:] if keep_x else hist_x[:0]
    st["hist_scores"] = np.concatenate([hist_s, ae_score])[-keep_s:] if keep_s else hist_s[:0]
    return ae_score, ae_flag, top


def step(entity: str, rows: pd.DataFrame, st: dict, models: dict) -> tuple[pd.DataFrame, int]:
    """One micro-batch of one machine: fold rows (event_us, m00..m37, split, label) into st, score the
    windows that became complete. Returns (OUTPUT_COLUMNS rows for those windows, late rows dropped)."""
    batch_ingest = (float(np.nanmax(rows["ingest_ms"].to_numpy(dtype=np.float64)))
                    if "ingest_ms" in rows and rows["ingest_ms"].notna().any() else np.nan)
    late = _add_rows(st, rows.sort_values("event_us", kind="stable"))
    w = _pop_complete(st)
    if w is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), late
    means, maxes, split, label = _window_values(w)
    n = len(means)
    i_mean = models["ae"]["aggs"].index("mean")   # previous window means, before score_windows updates hist_x
    history = st["hist_x"][:, i_mean * K:(i_mean + 1) * K]
    ae_score, ae_flag, top = score_windows(entity, means, maxes, split, st, models["ae"])
    out = pd.DataFrame({"entity_id": [entity] * n, "ws_us": w["open_starts"], "split": split, "label": label,
                        "ae_score": ae_score, "ae_flag": pd.Series(ae_flag, dtype=object), "top_metrics": top})
    if entity not in models["ae"]["scalers"]:
        out["ae_flag"] = None
    out["tier1_score"], out["tier1_flag"] = tier1_scores(entity, means, models.get("tier1"))
    out["tier2_score"], out["tier2_flag"], out["pca_score"] = np.nan, None, np.nan
    module = models.get("baselines_module")
    if module is not None:
        pdf = pd.DataFrame(np.hstack([means, maxes]), columns=AE.feature_names(METRICS, ("mean", "max")))
        pdf.insert(0, "entity_id", entity)
        pdf.insert(1, "window_start", w["open_starts"])
        pdf.insert(2, "split", split)
        res = module.score_baselines(pdf, models, history)
        out["tier2_score"] = np.asarray(res["tier2_score"], dtype=np.float64)
        out["tier2_flag"] = [None if pd.isna(f) else bool(f) for f in res["tier2_flag"]]
        out["pca_score"] = np.asarray(res["pca_score"], dtype=np.float64)
    out["max_ingest_ms"], out["batch_ingest_ms"] = w["open_ingest"], batch_ingest
    return out[OUTPUT_COLUMNS], late


def score_group(key, pdf_iter, state):
    """applyInPandasWithState function, keyed by entity_id."""
    t0 = time.perf_counter()
    models = get_models()
    n_in = len(models["ae"]["feature_metrics"])
    rows = pd.concat(list(pdf_iter), ignore_index=True)
    st = decode_state(state.get, n_in) if state.exists else new_state(n_in)
    out, _ = step(key[0], rows, st, models)
    state.update(encode_state(st))
    worker_log("score_group", entity=key[0], rows=len(rows), windows=len(out),
               seconds=round(time.perf_counter() - t0, 3))
    yield out


# ---------------------------------------------------------------- Spark

def read_files(spark, machines: int | None, files_per_trigger: int = 1):
    raw_dir = ROOT / S.PATHS["raw"] / f"source={SOURCE}"
    files = sorted(p for p in raw_dir.glob("*.parquet") if p.name not in AE.SKIP_FILES)
    if machines:
        files = files[:machines]
    if not files:
        raise FileNotFoundError(f"no machine Parquet files under {raw_dir}")
    glob = "{" + ",".join(p.name for p in files) + "}"
    return (spark.readStream.schema(RAW_DDL).option("maxFilesPerTrigger", files_per_trigger)
            .option("pathGlobFilter", glob)
            .parquet(raw_dir.as_posix()).withWatermark("event_time", WATERMARK)), [p.stem for p in files]


def build_scores(telemetry):
    """telemetry (entity_id, event_time watermarked, metrics, attrs, label) -> scored windows with the
    scores-layer columns plus split, ws_us, max_ingest_ms and batch_ingest_ms (null without attrs.ingest_ts)."""
    from pyspark.sql import functions as F
    from pyspark.sql.streaming.state import GroupStateTimeout
    from src.streaming import score_job as mod  # by-name reference, so workers keep their model cache

    rows = telemetry.select("entity_id", F.expr("unix_micros(event_time)").alias("event_us"),
                            *[F.col("metrics")[m].alias(m) for m in METRICS],
                            F.col("attrs")["split"].alias("split"), F.col("label").cast("double").alias("label"),
                            F.col("attrs")["ingest_ts"].cast("double").alias("ingest_ms"))
    windows = rows.groupBy("entity_id").applyInPandasWithState(
        mod.score_group, OUTPUT_DDL, STATE_DDL, "append", GroupStateTimeout.NoTimeout)
    cols = {"window_start": F.expr("timestamp_micros(ws_us)"),
            "window_end": F.expr(f"timestamp_micros(ws_us + {WINDOW_US})"),
            "is_anomaly": F.coalesce("ae_flag", F.lit(False)),   # AE only; Tier 1 never raises the alarm
            "label": F.col("label").cast("int")}
    return windows.select(*[cols.get(c, F.col(c)).alias(c) for c in S.LAYERS["scores"]], "split", "ws_us",
                          "max_ingest_ms", "batch_ingest_ms")


def write_scores_file(pdf: pd.DataFrame, batch_id: int, out_dir: Path = OUTPUT_DIR) -> Path:
    """Test windows of one micro-batch -> out_dir/batch-<id>.parquet with the exact scores-layer types."""
    schema = S.arrow_schema("scores")
    start = pa.array(pdf["ws_us"].to_numpy(dtype=np.int64), type=pa.int64())
    cols = {"window_start": start.cast(schema.field("window_start").type),
            "window_end": pa.array(pdf["ws_us"].to_numpy(dtype=np.int64) + WINDOW_US).cast(
                schema.field("window_end").type),
            "top_metrics": pa.array([list(t) if t is not None else None for t in pdf["top_metrics"]],
                                    type=pa.list_(pa.string())),
            "label": pa.array(pdf["label"].astype("float64"), from_pandas=True).cast(pa.int32())}
    arrays = [cols[f.name] if f.name in cols else
              pa.array(pdf[f.name].astype(object).where(pdf[f.name].notna(), None).tolist(), type=f.type)
              for f in schema]
    out_dir.mkdir(parents=True, exist_ok=True)
    path, tmp = out_dir / f"batch-{batch_id:06d}.parquet", out_dir / f".batch-{batch_id:06d}.tmp"
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), tmp)
    os.replace(tmp, path)
    return path


def write_latency_file(pdf: pd.DataFrame, batch_id: int, emit_ts_ms: int, out_dir: Path = LATENCY_DIR) -> Path:
    """Emitted windows of one micro-batch (entity_id, ws_us, max_ingest_ms, batch_ingest_ms) ->
    out_dir/batch-<id>.parquet with LATENCY_SCHEMA (ingest times null where the records had none)."""
    def ms(col):
        return pa.array(pdf[col].astype("float64").round(), from_pandas=True).cast(pa.int64())
    n = len(pdf)
    start_type = LATENCY_SCHEMA.field("window_start").type
    arrays = [pa.array(pdf["entity_id"].astype(str).tolist(), type=pa.string()),
              pa.array(pdf["ws_us"].to_numpy(dtype=np.int64), type=pa.int64()).cast(start_type),
              ms("max_ingest_ms"), ms("batch_ingest_ms"),
              pa.array(np.full(n, emit_ts_ms, dtype=np.int64)), pa.array(np.full(n, batch_id, dtype=np.int64))]
    out_dir.mkdir(parents=True, exist_ok=True)
    path, tmp = out_dir / f"batch-{batch_id:06d}.parquet", out_dir / f".batch-{batch_id:06d}.tmp"
    pq.write_table(pa.Table.from_arrays(arrays, schema=LATENCY_SCHEMA), tmp)
    os.replace(tmp, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", choices=("kafka", "files"), default="kafka")
    parser.add_argument("--machines", type=int, default=None, help="files input: only the first N machine files")
    parser.add_argument("--once", action="store_true", help="trigger(availableNow=True), then stop")
    parser.add_argument("--reset", action="store_true",
                        help="delete data/scores_stream/source=smd, data/latency_stream and "
                             "checkpoints/score/source=smd first")
    parser.add_argument("--tier1", choices=TIER1_MODES, default="off",
                        help="off (default): tier1 columns null; v1: robust z vs train median from "
                             "models/tier1_smd/baselines_v1.parquet. is_anomaly stays AE only either way")
    parser.add_argument("--files-per-trigger", type=int, default=1,
                        help="files input: machine files per micro-batch (maxFilesPerTrigger, default 1)")
    parser.add_argument("--master", default="local[*]", help="Spark master, e.g. local[4] (default local[*])")
    parser.add_argument("--shuffle-partitions", type=int, default=4,
                        help="spark.sql.shuffle.partitions (default 4)")
    parser.add_argument("--trigger", default=TRIGGER,
                        help=f"processing-time trigger without --once (default '{TRIGGER}')")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="stop the query after this many wall seconds")
    parser.add_argument("--idle-stop", type=float, default=None,
                        help="stop once rows were seen and no micro-batch had input rows for this many seconds")
    args = parser.parse_args(argv)

    if not (AE_DIR / "model.pt").exists():
        print(f"FAIL no autoencoder at {AE_DIR.relative_to(ROOT).as_posix()}: run src/detection/autoencoder.py")
        return 1
    if args.reset:
        for d in (OUTPUT_DIR, CHECKPOINT_DIR, LATENCY_ROOT):
            shutil.rmtree(d, ignore_errors=True)
        print("reset: removed stream output and checkpoint", flush=True)
    if args.tier1 == "v1" and not TIER1_PATH.exists():
        print(f"FAIL --tier1 v1 needs {TIER1_PATH.relative_to(ROOT).as_posix()}: run src/detection/tier1_zscore.py")
        return 1
    os.environ[TIER1_ENV] = args.tier1   # before the SparkSession, so the Python workers inherit it
    print(f"tier1: {args.tier1} (is_anomaly = ae_flag only)", flush=True)
    if not BASELINES_MODULE.exists():
        print(f"note: {BASELINES_MODULE.relative_to(ROOT).as_posix()} not found, "
              "tier2_score/tier2_flag/pca_score stay null", flush=True)

    from pyspark.sql import functions as F
    from src.streaming.features_job import build_spark, read_telemetry

    kafka = args.input == "kafka"
    spark = build_spark("score_job", with_kafka=kafka, master=args.master,
                        shuffle_partitions=args.shuffle_partitions)
    spark.sparkContext.setLogLevel("ERROR")
    if kafka:
        telemetry, what = read_telemetry(spark), S.TOPIC_SMD
    else:
        telemetry, names = read_files(spark, args.machines, args.files_per_trigger)
        what = f"{len(names)} machine files ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})"

    stats: dict[int, tuple[int, int, int]] = {}
    steps_ms: dict[int, dict] = {}   # wall time of each foreachBatch step, printed with the batch line

    def write_batch(batch_df, batch_id):
        t = [time.perf_counter()]
        marks = {}

        def mark(name):
            t.append(time.perf_counter())
            marks[name] = round((t[-1] - t[-2]) * 1000)

        batch_df.persist()
        try:
            n_windows = batch_df.count()
            mark("count(stateful scoring)")
            test = batch_df.filter(F.col("split") == "test")
            pdf = test.drop("split", "window_start", "window_end").toPandas()
            mark("toPandas(test)")
            if len(pdf):
                write_scores_file(pdf, batch_id)
            mark("write_scores")
            n_anom = int(pdf["is_anomaly"].sum()) if len(pdf) else 0
            lat = (batch_df.select("entity_id", "ws_us", "max_ingest_ms", "batch_ingest_ms").toPandas()
                   if kafka and n_windows else None)
            mark("toPandas(latency)")
            if kafka and n_anom:
                ts = {"timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"}
                (test.filter("is_anomaly")
                 .select(F.col("entity_id").cast("string").alias("key"),
                         F.to_json(F.struct(*S.LAYERS["scores"]), ts).alias("value"))
                 .write.format("kafka").option("kafka.bootstrap.servers", S.KAFKA_BOOTSTRAP)
                 .option("topic", S.TOPIC_ANOMALIES).save())
            mark("kafka(anomalies)")
            if lat is not None:
                write_latency_file(lat, batch_id, int(time.time() * 1000))
            mark("write_latency")
            stats[batch_id] = (n_windows, len(pdf), n_anom)
        finally:
            batch_df.unpersist()
            steps_ms[batch_id] = marks

    writer = (build_scores(telemetry).writeStream.foreachBatch(write_batch)
              .option("checkpointLocation", CHECKPOINT_DIR.as_posix()))
    writer = writer.trigger(availableNow=True) if args.once else writer.trigger(processingTime=args.trigger)
    query = writer.start()
    print(f"score_job started: {what} -> {OUTPUT_DIR.relative_to(ROOT).as_posix()}"
          f"{' + topic ' + S.TOPIC_ANOMALIES if kafka else ''}", flush=True)

    last_batch = -1
    started = last_input = time.monotonic()
    rows_seen = False

    def report():
        nonlocal last_batch, last_input, rows_seen
        for p in query.recentProgress:
            if p["batchId"] > last_batch:
                if p.get("numInputRows", 0):
                    rows_seen, last_input = True, time.monotonic()
                if kafka:
                    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
                    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"logged_ts_ms": int(time.time() * 1000), **p}) + "\n")
                state_rows = sum(op.get("numRowsTotal", 0) for op in p.get("stateOperators", []))
                n_windows, n_test, n_anom = stats.get(p["batchId"], (0, 0, 0))
                print(f"batch {p['batchId']}: input rows {p.get('numInputRows', 0)}, windows scored {n_windows} "
                      f"(test {n_test}, anomalies {n_anom}), duration "
                      f"{p.get('durationMs', {}).get('triggerExecution', 0)} ms, state rows {state_rows}", flush=True)
                print(f"  durationMs {json.dumps(p.get('durationMs', {}), sort_keys=True)}", flush=True)
                if p["batchId"] in steps_ms:
                    print(f"  foreachBatch steps ms {json.dumps(steps_ms[p['batchId']])}", flush=True)
                last_batch = p["batchId"]

    try:
        while True:
            stopped = query.awaitTermination(5)
            report()
            if stopped:
                break
            now = time.monotonic()
            if args.max_seconds is not None and now - started >= args.max_seconds:
                print(f"stopping score_job: --max-seconds {args.max_seconds:g} reached", flush=True)
                query.stop()
                break
            if args.idle_stop is not None and rows_seen and now - last_input >= args.idle_stop:
                print(f"stopping score_job: no input rows for {args.idle_stop:g} s", flush=True)
                query.stop()
                break
    except KeyboardInterrupt:
        print("stopping score_job", flush=True)
        query.stop()
    finally:
        spark.stop()
    return 0


def _log_import() -> None:
    """worker_log: how long importing this module took, and how old the process was when it finished."""
    try:
        import psutil
        age = round(time.time() - psutil.Process().create_time(), 3)
    except Exception:  # noqa: BLE001  (psutil is optional)
        age = None
    worker_log("module_import", seconds=round(time.time() - _IMPORT_T0, 3), process_age_s=age,
               main=__name__ == "__main__")


_log_import()

if __name__ == "__main__":
    sys.exit(main())
