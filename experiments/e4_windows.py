"""E4: window size study on SMD (1, 5 and 15 minute windows, every detector re-run in batch).

Run from the repo root:
    python experiments/e4_windows.py                    # windows 1 5 15, all 28 SMD machines
    python experiments/e4_windows.py --sample 2         # first 2 machines, quick run
    python experiments/e4_windows.py --windows 5        # only the 5-minute parity run
    python experiments/e4_windows.py --report-only      # rebuild the report from data/e4/w<N>/, no training

For each window size w (minutes) the SMD windows are rebuilt from data/raw/source=smd and every detector is
re-fitted on train windows and scored on test windows with the existing functions (nothing is copied):
    Tier 1 robust z-score   src/detection/tier1_zscore.py  build_windows / fit_stats (MAD 0 skipped) / score_windows
    Tier 2 KMeans           src/detection/baselines.py     k in 3..10 by silhouette, re-fitted per w
    Tier 2b PCA             src/detection/baselines.py     fewest components for 90% variance, re-fitted per w
    Tier 3 AE v2            src/detection/autoencoder.py   mean_max inputs, seed 42, global threshold + adaptive
    Final rule              is_anomaly = ae_flag (contract rev C)
Time spans stay the same at every w: recenter = 120 min (120 / w windows, 24 at 5 min), adaptive = 1 day
(1440 / w windows, 288 at 5 min). Thresholds stay p99.5 of the train score x 1.1 (global).

Metrics per detector and w come from src/detection/autoencoder.py `evaluate` (the function e1_report.py uses),
with window_minutes = w: composite F1, faults caught, alarm precision, false alarms per machine per day,
false-alarm minutes per machine per day (= false alarms per machine per day x w; false alarms and alarm precision
count windows, so the minutes are the fair comparison across w), median and mean detection delay in minutes, PR-AUC, windows flagged. A fault is a run of consecutive anomalous
windows, so the number of faults depends on w; "caught of the 5-minute faults" also counts how many of the
faults defined on 5-minute windows (326 on all of SMD) overlap at least one flagged window, so window sizes
can be compared on one denominator.

Check first: the 5-minute run must reproduce docs/results/e1.json (all 28 machines). With --sample only the
per-machine Tier 1 numbers are comparable (Tier 1 is fitted per machine; the other models and the global
thresholds are fitted on the sampled machines only).

Writes only:
    data/e4/w<N>/scores/source=smd/scores.parquet          scores-layer columns, test windows
    models/e4/w<N>/{tier1_smd,kmeans_smd,pca_smd,ae_smd}/  the re-fitted models
    docs/results/e4.md, e4.json, e4.png                    full run
    data/e4/sample<N>/e4.md, e4.json, e4.png               --sample N run (keeps docs/results clean)
Never data/scores*, data/features or the production models/ folders.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# On Windows, Spark must start workers with this same venv Python (as in tools/check_spark_kafka.py).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.detection import autoencoder as A  # noqa: E402
from src.detection import baselines as B  # noqa: E402
from src.detection import tier1_zscore as T  # noqa: E402
from src.detection.e1_report import _as_flag  # noqa: E402
from src.detection.evaluate import detection_delays  # noqa: E402

SOURCE = "smd"
WINDOWS = (1, 5, 15)
RECENTER_MINUTES = 120         # 24 windows at 5 minutes
ADAPTIVE_MINUTES = 24 * 60     # 288 windows at 5 minutes
REFERENCE_WINDOW = 5           # faults counted on 5-minute windows (326 on all of SMD), and the E1 parity run
E1_JSON = Path("docs/results/e1.json")
OUT_DIR = Path("docs/results")
DATA_DIR = Path("data/e4")
MODEL_DIR = Path("models/e4")
CHECK_TOL = 1e-3
US_PER_MIN = 60_000_000

# key -> (name, score column, flag column; None = score > the PCA threshold), same rows as e1_report.py
DETECTORS = {
    "tier1": ("Tier 1 robust z-score", "tier1_score", "tier1_flag"),
    "kmeans": ("Tier 2 KMeans (recenter 120 min)", "tier2_score", "tier2_flag"),
    "pca": ("Tier 2b PCA (recenter 120 min)", "pca_score", None),
    "ae_v2": ("Tier 3 AE v2", "ae_score", "ae_flag"),
    "is_anomaly": ("Final rule: is_anomaly = ae_flag", "ae_score", "is_anomaly"),
}
# compared with docs/results/e1.json in the 5-minute check
CHECK_KEYS = ("composite_f1", "faults_caught", "faults", "alarm_precision", "false_alarms_per_machine_day",
              "median_delay_min", "pr_auc", "windows_flagged", "windows")


def spans(window: int) -> dict:
    """Window counts that keep the recenter and adaptive time spans fixed at every window size."""
    return {"recenter": max(1, round(RECENTER_MINUTES / window)),
            "adaptive": max(1, round(ADAPTIVE_MINUTES / window))}


def window_str(window: int) -> str:
    return f"{window} minute" if window == 1 else f"{window} minutes"


def build_spark(driver_memory: str = "4g") -> SparkSession:
    """Same settings as baselines.build_spark (so the 5-minute run reproduces E1), memory configurable."""
    return (SparkSession.builder.appName("e4_windows").master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


# ---------------------------------------------------------------- detectors

def run_tier1(spark: SparkSession, files: list[Path], window: int, model_dir: Path) -> pd.DataFrame:
    """Tier 1 v1 (MAD 0 skipped, as the stream) on w-minute windows. Test windows: entity_id, ws_us,
    tier1_score, tier1_flag."""
    raw = spark.read.parquet(*[p.as_posix() for p in files])
    means, meta = T.build_windows(raw, S.SMD_METRICS, window_str(window))
    means, meta = means.cache(), meta.cache()
    stats = T.fit_stats(means, meta, "skip").cache()
    T.save_baselines(stats.toPandas(), model_dir)
    test = (T.score_windows(means, meta, stats)
            .select("entity_id", F.expr("unix_micros(window_start)").alias("ws_us"), "tier1_score", "tier1_flag")
            .toPandas())
    for df in (stats, means, meta):
        df.unpersist()
    return test


def run_baselines(spark: SparkSession, files: list[Path], window: int, recenter: int,
                  model_dir: Path) -> tuple[pd.DataFrame, dict]:
    """KMeans and PCA as in baselines.run (recenter on), without writing the scores layer or models/.
    Test windows: entity_id, ws_us, label, tier2_score, tier2_flag, pca_score, pca_flag."""
    metrics = S.SMD_METRICS
    raw = spark.read.parquet(*[p.as_posix() for p in files])
    windows = B.build_windows(raw, metrics, window_str(window)).cache()
    stats = B.scaler_stats(windows, metrics).cache()
    z = B.standardise(windows, stats, metrics).cache()
    counts = {r["split"]: r["count"] for r in z.groupBy("split").count().collect()}
    train = z.filter(F.col("split") == "train")
    kmeans, k, silhouette = B.fit_kmeans(train, metrics, B.K_MAX)
    pca, n_comp, cumulative = B.fit_pca(train, metrics)
    centers = B.recenter_stats(windows, stats, metrics, recenter).cache()
    z_score = B.standardise(windows, stats, metrics, centers).cache()
    scored = B.score_windows(z_score, kmeans, pca, metrics).cache()
    thr = {"tier2_score": B.threshold(scored, "tier2_score"), "pca_score": B.threshold(scored, "pca_score")}

    scaler = {"metrics": metrics, "window": window_str(window), "recenter": B.recenter_info(recenter),
              "entities": B.stats_to_json(stats, metrics)}
    common = {"percentile": B.THRESHOLD_PERCENTILE, "factor": B.THRESHOLD_FACTOR, "mode": "global",
              "window": window_str(window), "metrics": metrics}
    B.save_model(kmeans, model_dir / f"kmeans_{SOURCE}", {**common, "score": "tier2_score",
                 "global": thr["tier2_score"], "k": k, "silhouette": {str(a): b for a, b in silhouette.items()},
                 "silhouette_fraction": B.SILHOUETTE_FRACTION, "seed": B.SEED}, scaler)
    B.save_model(pca, model_dir / f"pca_{SOURCE}", {**common, "score": "pca_score", "global": thr["pca_score"],
                 "n_components": n_comp, "variance_target": B.PCA_VARIANCE,
                 "cumulative_variance": cumulative}, scaler)

    test = (scored.filter(F.col("split") == "test")
            .select("entity_id", "ws_us", "label", "tier2_score", "pca_score").toPandas())
    test["tier2_flag"] = test["tier2_score"] > thr["tier2_score"]
    test["pca_flag"] = test["pca_score"] > thr["pca_score"]
    for df in (scored, z_score, centers, z, stats, windows):
        df.unpersist()
    info = {"train_windows": int(counts.get("train", 0)), "test_windows": int(counts.get("test", 0)),
            "kmeans_k": k, "kmeans_silhouette": silhouette[k], "pca_components": n_comp,
            "pca_variance": cumulative[n_comp - 1], "threshold_tier2": thr["tier2_score"],
            "threshold_pca": thr["pca_score"]}
    return test, info


def run_autoencoder(raw_dir: Path, sample: int | None, window: int, recenter: int, adaptive: int,
                    model_dir: Path, epochs: int = A.MAX_EPOCHS) -> tuple[pd.DataFrame, dict]:
    """AE v2 exactly as autoencoder.run with its defaults (mean_max, global threshold, no std floor, no flat
    masking), but on w-minute windows and saving to model_dir. Test windows: entity_id, ws_us, label,
    ae_score, ae_flag, top_metrics."""
    metrics = A.METRICS[SOURCE]
    inputs = A.DEFAULT_INPUTS
    aggs = A.INPUTS[inputs]
    features = A.feature_names(metrics, aggs)
    feature_metrics = [m for _ in aggs for m in metrics]
    meta, X = A.load_windows(raw_dir, metrics, sample, aggs, window * US_PER_MIN)
    is_train = (meta["split"] == "train").to_numpy()

    scalers = A.fit_scalers(meta, X, A.STD_FLOOR)
    for entity, masked in A.flat_metrics(meta, X, metrics, A.FLAT_STD).items():
        scalers[entity]["masked"] = masked
    centers = A.recenter_centers(meta, X, scalers, recenter)
    Z, mask = A.standardise(meta, X, scalers, feature_metrics, centers)
    model, history = A.train(Z[is_train], mask[is_train], epochs=epochs, verbose=False)
    ae_score, top = A.score(model, Z, mask, metrics)
    adaptive_info = {"windows": adaptive, "percentile": A.ADAPTIVE_PERCENTILE, "min_periods": max(1, adaptive // 2)}
    threshold_info = {"mode": "global", **A.thresholds(meta, ae_score), "adaptive": adaptive_info}
    limit = A.adaptive_limits(meta, ae_score, adaptive)
    ae_flag = A.flags(meta, ae_score, threshold_info, "global", limit)

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_in": len(features), "hidden": list(A.HIDDEN),
                "metrics": metrics, "inputs": inputs, "features": features, "window_minutes": window},
               model_dir / "model.pt")
    recenter_info = {"windows": recenter, "min_periods": max(1, recenter // 2), "statistic": "median",
                     "applies_to": "test", "fallback": "train mean"}
    (model_dir / "scaler.json").write_text(json.dumps(
        {"metrics": metrics, "inputs": inputs, "features": features, "std_floor": A.STD_FLOOR,
         "flat_std": A.FLAT_STD, "recenter": recenter_info, "window_minutes": window, "entities": scalers}))
    (model_dir / "threshold.json").write_text(json.dumps(threshold_info, indent=2))
    (model_dir / "history.json").write_text(json.dumps(history, indent=2))

    test = ~is_train
    out = pd.DataFrame({"entity_id": meta.loc[test, "entity_id"].to_numpy(),
                        "ws_us": meta.loc[test, "window_start"].to_numpy(dtype=np.int64),
                        "ae_label": meta.loc[test, "label"].to_numpy(),
                        "ae_score": ae_score[test], "ae_flag": ae_flag[test],
                        "top_metrics": [top[i] for i in np.flatnonzero(test)]})
    best = min(history, key=lambda h: h["val_loss"])
    info = {"ae_epochs": len(history), "ae_best_epoch": best["epoch"], "threshold_ae": threshold_info["global"]}
    return out, info


# ---------------------------------------------------------------- one window size

def combine(base: pd.DataFrame, tier1: pd.DataFrame, ae: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The baselines' test windows with tier1 and AE columns left-joined on (entity_id, ws_us), sorted."""
    keys = ["entity_id", "ws_us"]
    df = base.merge(tier1, on=keys, how="left").merge(ae, on=keys, how="left")
    df = df.sort_values(keys).reset_index(drop=True)
    df["is_anomaly"] = _as_flag(df["ae_flag"])                       # rev C: AE alone raises the alarm
    match = {"windows": len(df), "tier1_windows": len(tier1), "ae_windows": len(ae),
             "tier1_missing": int(df["tier1_flag"].isna().sum()),
             "ae_missing": int(df["ae_flag"].isna().sum()),
             "label_mismatch": int((df["ae_label"].notna() & (df["ae_label"] != df["label"])).sum())}
    return df.drop(columns="ae_label"), match


def write_scores(df: pd.DataFrame, window: int, path: Path) -> None:
    """Test windows with the scores-layer schema (contracts/schemas.py)."""
    schema = S.arrow_schema("scores")
    start = pa.array(df["ws_us"].to_numpy(dtype=np.int64), type=pa.int64())

    def opt(col, t):
        return pa.array(df[col].astype(object).where(df[col].notna(), None).tolist(), type=t)

    cols = {"entity_id": pa.array(df["entity_id"].tolist(), type=pa.string()),
            "window_start": start.cast(schema.field("window_start").type),
            "window_end": pc.add(start, window * US_PER_MIN).cast(schema.field("window_end").type),
            "tier1_score": opt("tier1_score", pa.float64()), "tier1_flag": opt("tier1_flag", pa.bool_()),
            "tier2_score": opt("tier2_score", pa.float64()), "tier2_flag": opt("tier2_flag", pa.bool_()),
            "pca_score": opt("pca_score", pa.float64()), "ae_score": opt("ae_score", pa.float64()),
            "ae_flag": opt("ae_flag", pa.bool_()), "is_anomaly": pa.array(df["is_anomaly"].tolist(), pa.bool_()),
            "top_metrics": pa.array([t if isinstance(t, list) else None for t in df["top_metrics"]],
                                    type=pa.list_(pa.string())),
            "label": pa.array(df["label"].astype("Int64").tolist(), type=pa.int32())}
    arrays = [cols[f.name] if f.name in cols else pa.nulls(len(df), type=f.type) for f in schema]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


def run_window(spark: SparkSession, window: int, root: Path = ROOT, sample: int | None = None,
               epochs: int = A.MAX_EPOCHS) -> dict:
    """Rebuild the windows, re-fit and score every detector at one window size; returns the test frame,
    the PCA threshold and run info. Writes only under data/e4/w<N>/ and models/e4/w<N>/."""
    t0 = time.perf_counter()
    raw_dir = root / S.PATHS["raw"] / f"source={SOURCE}"
    files = B.raw_files(raw_dir, sample)
    sp = spans(window)
    models = root / MODEL_DIR / f"w{window}"
    print(f"\n=== window {window} min: {len(files)} machines, recenter {sp['recenter']} windows "
          f"({sp['recenter'] * window} min), adaptive {sp['adaptive']} windows ({sp['adaptive'] * window} min)")

    base, info = run_baselines(spark, files, window, sp["recenter"], models)
    t1 = time.perf_counter()
    print(f"  baselines: {info['train_windows']} train + {info['test_windows']} test windows, KMeans k = "
          f"{info['kmeans_k']}, PCA {info['pca_components']} components ({info['pca_variance']:.3f}), "
          f"{t1 - t0:.0f} s")
    tier1 = run_tier1(spark, files, window, models / f"tier1_{SOURCE}")
    t2 = time.perf_counter()
    print(f"  tier 1: {len(tier1)} test windows, {t2 - t1:.0f} s")
    ae, ae_info = run_autoencoder(raw_dir, sample, window, sp["recenter"], sp["adaptive"],
                                  models / f"ae_{SOURCE}", epochs)
    t3 = time.perf_counter()
    print(f"  AE v2: {ae_info['ae_epochs']} epochs (best {ae_info['ae_best_epoch']}), {t3 - t2:.0f} s")

    df, match = combine(base, tier1, ae)
    write_scores(df, window, root / DATA_DIR / f"w{window}" / "scores" / f"source={SOURCE}" / "scores.parquet")
    if match["tier1_missing"] or match["ae_missing"] or match["label_mismatch"]:
        print(f"  WARNING window match: {match}")
    seconds = time.perf_counter() - t0
    print(f"  window {window} min done in {seconds:.0f} s ({seconds / 60:.1f} min)")
    return {"window": window, "df": df, "pca_threshold": info["threshold_pca"],
            "info": {**info, **ae_info, **sp, "machines": len(files), "match": match, "seconds": seconds,
                     "seconds_baselines": t1 - t0, "seconds_tier1": t2 - t1, "seconds_ae": t3 - t2}}


# ---------------------------------------------------------------- evaluation

def reference_faults(raw_dir: Path, sample: int | None, window: int = REFERENCE_WINDOW) -> dict[str, list]:
    """Per machine, the faults on `window`-minute test windows (runs of windows with any label-1 row) as
    (start_us, end_us) intervals. On all of SMD with 5 minutes there are 326, the E1 denominator."""
    out = {}
    w_us = window * US_PER_MIN
    for path in B.raw_files(raw_dir, sample):     # one machine per file
        t = pq.read_table(path, columns=["entity_id", "event_time", "attrs", "label"])
        us = t["event_time"].cast(pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy()
        df = pd.DataFrame({"ws": us - np.mod(us, w_us),
                           "test": pc.map_lookup(t["attrs"].combine_chunks(), "split", "first")
                           .to_numpy(zero_copy_only=False) == "test",
                           "label": t["label"].to_pandas().fillna(0).astype(int)})
        g = df.groupby("ws", sort=True).agg(test=("test", "max"), label=("label", "max"))
        g = g[g["test"]]
        lab = g["label"].to_numpy() > 0
        ws = g.index.to_numpy(dtype=np.int64)
        edges = np.diff(np.concatenate(([0], lab.astype(int), [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        out[t["entity_id"][0].as_py()] = [(int(ws[s]), int(ws[e - 1]) + w_us) for s, e in zip(starts, ends)]
    return out


def reference_caught(df: pd.DataFrame, flag: np.ndarray, window: int, faults: dict[str, list]) -> int:
    """How many reference faults overlap at least one flagged window [ws, ws + window)."""
    w_us = window * US_PER_MIN
    caught = 0
    flagged = df.loc[flag, ["entity_id", "ws_us"]]
    by_entity = {e: np.sort(g["ws_us"].to_numpy()) for e, g in flagged.groupby("entity_id")}
    for entity, intervals in faults.items():
        fws = by_entity.get(entity)
        if fws is None or not len(fws):
            continue
        for start, end in intervals:
            i = np.searchsorted(fws, start - w_us, side="right")     # first flagged window ending after start
            caught += bool(i < len(fws) and fws[i] < end)
    return caught


def delays_minutes(df: pd.DataFrame, flag: np.ndarray, window: int) -> list[float]:
    """Detection delays in minutes over all detected faults (per machine, time-ordered, evaluate.py)."""
    work = pd.DataFrame({"entity_id": df["entity_id"], "ws": df["ws_us"], "t": df["label"].astype(int),
                         "p": flag.astype(int)}).sort_values(["entity_id", "ws"])
    out = []
    for _, g in work.groupby("entity_id", sort=True):
        out += [d * window for d in detection_delays(g["t"].to_numpy(), g["p"].to_numpy())]
    return out


def detector_flags(df: pd.DataFrame, pca_thr: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Same rule as e1_report.detector_flags: a flag column, or pca_score > the PCA threshold."""
    out = {}
    for key, (_, score_col, flag_col) in DETECTORS.items():
        score = df[score_col].to_numpy(dtype=np.float64)
        flag = np.nan_to_num(score, nan=-np.inf) > pca_thr if flag_col is None else _as_flag(df[flag_col])
        out[key] = (flag, score)
    return out


def evaluate_window(df: pd.DataFrame, pca_thr: float, window: int,
                    faults: dict[str, list] | None = None) -> tuple[list[dict], pd.DataFrame]:
    """One row per detector (keys as in e1.json) and a per-machine composite F1 table."""
    meta = pd.DataFrame({"entity_id": df["entity_id"], "window_start": df["ws_us"], "split": "test",
                         "label": df["label"].astype("float64")})
    rows, per_machine = [], {}
    for key, (flag, score) in detector_flags(df, pca_thr).items():
        pm, o = A.evaluate(meta, flag, score, window_minutes=window)
        c = o["composite"]
        delays = delays_minutes(df, flag, window)
        rows.append({
            "window_min": window, "detector": DETECTORS[key][0], "key": key,
            "composite_f1": c["composite_f1"], "faults_caught": c["events_detected"], "faults": c["events"],
            "event_recall": c["event_recall"], "alarm_precision": c["alarm_precision"],
            "false_alarms_per_machine_day": o["fa_day"],
            "false_alarm_minutes_per_machine_day": o["fa_day"] * window,
            "median_delay_min": o["delay_min"],
            "mean_delay_min": float(np.mean(delays)) if delays else float("nan"),
            "pr_auc": o["pr_auc"], "point_f1": o["point"]["f1"], "event_f1": o["event"]["f1"],
            "windows_flagged": int(flag.sum()), "windows": int(len(flag)),
            "ref_faults_caught": reference_caught(df, flag, window, faults) if faults is not None else None,
            "ref_faults": sum(len(v) for v in faults.values()) if faults is not None else None,
        })
        per_machine[key] = pm.set_index("entity_id")["comp_f1"]
    return rows, pd.DataFrame(per_machine).rename_axis("machine").reset_index()


# ---------------------------------------------------------------- check against E1

def check_e1(rows: list[dict], pm: pd.DataFrame, e1: dict, sample: int | None) -> dict:
    """5-minute rows against docs/results/e1.json. Full run: every CHECK_KEYS value per detector.
    Sample run: per-machine composite F1 of Tier 1 only (the other detectors' models and global thresholds
    depend on which machines they were fitted on)."""
    lines, ok = [], True
    if sample:
        ref = {r["machine"]: r for r in e1["per_machine_composite_f1"]}
        for _, r in pm.iterrows():
            e = ref.get(r["machine"])
            if e is None:
                continue
            d = abs(r["tier1"] - e["tier1"])
            ok &= d <= CHECK_TOL
            lines.append({"detector": "tier1", "item": f"composite F1 {r['machine']}", "e4": r["tier1"],
                          "e1": e["tier1"], "diff": d, "ok": d <= CHECK_TOL})
        note = (f"sample run ({sample} machines): only Tier 1 per-machine composite F1 is comparable; "
                "KMeans, PCA and AE are fitted (and their global thresholds taken) on the sampled machines only")
    else:
        ref = {r["key"]: r for r in e1["detectors"]}
        for r in rows:
            e = ref[r["key"]]
            for k in CHECK_KEYS:
                a, b = r[k], e[k]
                d = 0.0 if (a is None or (isinstance(a, float) and math.isnan(a))) and b is None else abs(a - b)
                good = d <= CHECK_TOL
                ok &= good
                lines.append({"detector": r["key"], "item": k, "e4": a, "e1": b, "diff": d, "ok": good})
        note = "full run: every detector row of e1.json, tolerance 0.001"
    return {"passed": bool(ok), "note": note, "items": lines}


def print_check(check: dict) -> None:
    print(f"\n=== Check: 5-minute run vs {E1_JSON.as_posix()} ({check['note']})")
    view = pd.DataFrame(check["items"])
    if len(view):
        view["ok"] = view["ok"].map({True: "OK", False: "DIFF"})
        with pd.option_context("display.width", 200, "display.max_rows", None, "display.float_format", "{:.4f}".format):
            print(view.to_string(index=False))
    print(f"E4 5-MINUTE CHECK {'PASS' if check['passed'] else 'FAIL'}")


# ---------------------------------------------------------------- output

def _f(v, fmt="{:.3f}"):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


TABLE_HEADER = ("| window | detector | composite F1 | faults caught | caught of 5-min faults | alarm precision "
                "| false alarms / machine / day | false-alarm minutes / machine / day | median delay (min) "
                "| mean delay (min) | PR-AUC | windows flagged |\n"
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")


def main_table(rows: list[dict]) -> str:
    lines = [TABLE_HEADER]
    for r in rows:
        ref = f"{r['ref_faults_caught']}/{r['ref_faults']}" if r["ref_faults"] is not None else "n/a"
        lines.append(
            f"| {r['window_min']} min | {r['detector']} | {_f(r['composite_f1'])} | {r['faults_caught']}/{r['faults']} "
            f"| {ref} | {_f(r['alarm_precision'])} | {_f(r['false_alarms_per_machine_day'], '{:.2f}')} "
            f"| {_f(r['false_alarm_minutes_per_machine_day'], '{:.1f}')} "
            f"| {_f(r['median_delay_min'], '{:.0f}')} | {_f(r['mean_delay_min'], '{:.1f}')} | {_f(r['pr_auc'])} "
            f"| {r['windows_flagged']:,} / {r['windows']:,} |")
    return "\n".join(lines)


def settings_table(runs: list[dict]) -> str:
    lines = ["| window | machines | train windows | test windows | recenter (windows) | adaptive (windows) "
             "| KMeans k | PCA components | AE epochs (best) | run time (min) |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in runs:
        i = r["info"]
        lines.append(f"| {r['window']} min | {i['machines']} | {i['train_windows']:,} | {i['test_windows']:,} "
                     f"| {i['recenter']} | {i['adaptive']} | {i['kmeans_k']} | {i['pca_components']} "
                     f"| {i['ae_epochs']} ({i['ae_best_epoch']}) | {i['seconds'] / 60:.1f} |")
    return "\n".join(lines)


def tradeoff(rows: list[dict]) -> str:
    """A short, number-driven paragraph; it states what the table shows and its limits, nothing more."""
    by = {(r["key"], r["window_min"]): r for r in rows}
    ws = sorted({r["window_min"] for r in rows})
    if len(ws) < 2:
        return "Only one window size was run, so there is no trade-off to describe."
    parts = []
    lo, hi = ws[0], ws[-1]
    for key in ("ae_v2", "kmeans", "pca", "tier1"):
        name = DETECTORS[key][0]
        vals = ", ".join(f"{by[(key, w)]['composite_f1']:.3f} at {w} min" for w in ws)
        fa = ", ".join(f"{by[(key, w)]['false_alarms_per_machine_day']:.2f}" for w in ws)
        fam = ", ".join(f"{by[(key, w)]['false_alarm_minutes_per_machine_day']:.1f}" for w in ws)
        md = ", ".join(_f(by[(key, w)]["mean_delay_min"], "{:.1f}") for w in ws)
        a, b = by[(key, lo)], by[(key, hi)]
        count_ratio = a["false_alarms_per_machine_day"] / b["false_alarms_per_machine_day"] \
            if b["false_alarms_per_machine_day"] else float("nan")
        minute_ratio = b["false_alarm_minutes_per_machine_day"] / a["false_alarm_minutes_per_machine_day"] \
            if a["false_alarm_minutes_per_machine_day"] else float("nan")
        if math.isnan(count_ratio) or math.isnan(minute_ratio):
            fa_text = ""
        elif minute_ratio >= 1:
            fa_text = (f" Going from {lo} to {hi} min the false-alarm count per day falls {count_ratio:.1f}x, but "
                       f"false-alarm minutes rise {minute_ratio:.2f}x, so the larger window does not cut false-alarm "
                       "time; the drop in the count comes from counting longer windows.")
        else:
            fa_text = (f" Going from {lo} to {hi} min the false-alarm count per day falls {count_ratio:.1f}x and "
                       f"false-alarm minutes fall {1 / minute_ratio:.2f}x; only the second is a like-for-like "
                       "reduction.")
        parts.append(f"{name}: composite F1 {vals}; false alarms per machine per day {fa} "
                     f"(false-alarm minutes {fam}); mean delay {md} min.{fa_text}")
    best = {w: max((by[(k, w)] for k in ("tier1", "kmeans", "pca", "ae_v2")), key=lambda r: r["composite_f1"])
            for w in ws}
    parts.append("Highest composite F1 per window size: "
                 + "; ".join(f"{w} min {best[w]['detector']} ({best[w]['composite_f1']:.3f})" for w in ws) + ".")
    parts.append(
        "Delays are measured from the start of a fault's first window to the start of the first flagged window, "
        "so they do not include the time until a window closes and can be scored (up to w minutes); in the stream "
        "a 15-minute window can raise its alarm at the earliest 15 minutes after the fault begins, a 1-minute "
        "window after 1 minute. Faults are runs of anomalous windows, so their number changes with w (nearby faults "
        "merge at 15 minutes and can split at 1 minute); the \"caught of 5-min faults\" column keeps one "
        "denominator. False alarms per day and alarm precision count windows (see the note under the table), so "
        "across window sizes the false-alarm minutes column is the like-for-like comparison of false-alarm time; "
        "alarm precision, and composite F1 through it, still carry the window-counting effect, and a larger window "
        "is labelled anomalous if any of its minutes is, which also changes what counts as a correct alarm. "
        "These are results of one train/test split of one dataset (SMD) with one AE seed; differences of a few "
        "faults or a few hundredths of composite F1 are within what a retrain can change (E1 notes a 0.002 shift "
        "from retraining the AE on another laptop), so only larger gaps should be read as an effect of w.")
    return " ".join(parts)


def counting_note(rows: list[dict]) -> str:
    """Why false alarms per day and alarm precision are not comparable across w, and what the fault columns use."""
    faults = {}
    for r in rows:
        faults.setdefault(r["window_min"], r["faults"])
    ref = next((r["ref_faults"] for r in rows if r["ref_faults"] is not None), None)
    own = ", ".join(f"{n} at {w} min" for w, n in sorted(faults.items()))
    text = ("**Comparing window sizes.** *False alarms / machine / day* and *alarm precision* count windows, not "
            "minutes: a 5-minute false episode is 5 false alarms at 1 min but 1 at 5 min. Across window sizes the "
            "*false-alarm minutes / machine / day* column (= false alarms per machine per day x window minutes) is "
            "the fair comparison; within one window size the minutes are the count times a constant, so they rank "
            "the detectors the same way. "
            f"*Faults caught* uses each run's own faults ({own})")
    if ref is not None:
        text += (f"; *caught of 5-min faults* (`caught_5min` in the console table) maps each run's detections onto "
                 f"the {ref} faults defined on 5-minute windows, the E1 faults")
    return text + "."


def write_markdown(path: Path, rows: list[dict], runs: list[dict], check: dict | None, sample: int | None,
                   report_only: bool = False) -> None:
    check_text = "The 5-minute run was not part of this run, so it was not checked against E1."
    if check is not None:
        bad = [i for i in check["items"] if not i["ok"]]
        check_text = (f"**{'PASS' if check['passed'] else 'FAIL'}** ({check['note']}). "
                      + ("Every compared value is within 0.001 of `docs/results/e1.json`."
                         if not bad else "Values outside 0.001:\n\n"
                         + "\n".join(f"- {i['detector']} {i['item']}: E4 {i['e4']}, E1 {i['e1']}" for i in bad)))
    scope = f"the first {sample} SMD machines (`--sample {sample}`, quick run)" if sample else "all SMD machines"
    command = "python experiments/e4_windows.py" + (f" --sample {sample}" if sample else "")
    generated = (f"Scores and models from `{command}` on {scope}; this report was rebuilt from the saved scores "
                 f"under `data/e4/w<N>/` with `{command} --report-only` (nothing re-trained)."
                 if report_only else f"Generated by `{command}` on {scope}.")
    text = f"""# E4: window size study on SMD

{generated}

## Method

For each window size w in {{{', '.join(str(r['window']) for r in runs)}}} minutes the SMD windows are rebuilt from
`data/raw/source=smd` (tumbling, epoch-aligned, per machine; a window is `test` if any of its rows is a test row and
its label is 1 if any row is labelled anomalous) and every detector is re-fitted on train windows only and scored on
test windows, with the existing functions in `src/detection/` (window size passed as a parameter, default 5 minutes):
Tier 1 robust z-score v1 (train median/MAD per machine-metric, MAD 0 skipped, flag if z > 3);
Tier 2 KMeans (k in 3..10 by silhouette, re-chosen per w); Tier 2b PCA (fewest components for 90% of the train
variance, re-chosen per w); Tier 3 AE v2 (mean and max inputs, seed 42, early stopping). KMeans, PCA and AE v2 centre
each test window on the median of the machine's previous 120 minutes of windows (120 / w windows; 24 at 5 min);
AE v2 also requires its score to exceed the 99th percentile of the machine's previous day of scores (1440 / w
windows; 288 at 5 min). Thresholds: 99.5th percentile of the train score x 1.1, over all machines, fitted per w.
**Final rule** `is_anomaly = ae_flag` (contract rev C). **Metrics**: the `evaluate` function of
`src/detection/autoencoder.py` that `e1_report.py` uses, with w-minute windows (definitions in `docs/results/e1.md`);
mean delay averages the same per-fault delays as the median. *Caught of 5-min faults* counts the faults defined on
5-minute windows that overlap at least one flagged window, so every row has the same denominator.

## Settings per window size

{settings_table(runs)}

## Results (test windows)

{main_table(rows)}

{counting_note(rows)}

![composite F1, median and mean detection delay and false-alarm minutes vs window size](e4.png)

## Trade-off

{tradeoff(rows)}

## Check: 5-minute run against E1

{check_text}

## Reproduce

```
{command}                 # re-fit and score every window size, then write the report
{command} --report-only   # rebuild e4.md / e4.json / e4.png from data/e4/w<N>/ (no training)
```

Outputs: `data/e4/w<N>/scores/source=smd/scores.parquet` (scores-layer columns) and `models/e4/w<N>/`; the
production `data/scores*` and `models/*_smd` folders are not touched.
"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def write_json(path: Path, rows: list[dict], runs: list[dict], check: dict | None, sample: int | None,
               per_machine: dict) -> None:
    doc = {"experiment": "E4", "source": SOURCE, "sample": sample,
           "windows_min": [r["window"] for r in runs],
           "recenter_minutes": RECENTER_MINUTES, "adaptive_minutes": ADAPTIVE_MINUTES,
           "threshold": "p99.5 of train score x 1.1, global", "ae_seed": A.SEED,
           "is_anomaly_rule": "ae_flag (contract rev C)",
           "settings": [{"window_min": r["window"], **r["info"]} for r in runs],
           "detectors": rows, "check_e1": check,
           "per_machine_composite_f1": {str(w): pm.to_dict("records") for w, pm in per_machine.items()}}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(_clean(doc), indent=2))
        fh.write("\n")


LABEL_GAP = 0.06    # minimum vertical distance between end-of-line labels, as a fraction of the axes height


def spread_labels(ys: list[float], gap: float = LABEL_GAP, lo: float = 0.0, hi: float = 1.0) -> list[float]:
    """Move positions (axes fractions) apart so neighbours are at least `gap` apart, keeping their order and,
    where possible, staying inside [lo, hi]. Returned in the input order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for i in range(1, len(pos)):                       # push up
        pos[i] = max(pos[i], pos[i - 1] + gap)
    if pos and pos[-1] > hi:                           # then push down from the top if that ran off the axes
        pos[-1] = hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - gap)
    if len(pos) > 1 and pos[0] < lo:                   # more labels than fit at `gap`: spread evenly, keep order
        pos = np.linspace(lo, hi, len(pos)).tolist()
    out = [0.0] * len(ys)
    for p, i in zip(pos, order):
        out[i] = p
    return out


def _end_labels(ax, items: list[tuple[str, float, str]], x_end: float, muted: str) -> None:
    """Value labels at the right end of each line, spread vertically so they never overlap. Labels with the same
    text that would overlap are merged into one muted label (e.g. four lines ending at 0); the others take their
    line's colour so a moved label can still be matched to its line."""
    to_axes = ax.transData + ax.transAxes.inverted()
    pts = sorted(((text, to_axes.transform((x_end, y))[1], color) for text, y, color in items), key=lambda t: t[1])
    merged: list[list] = []
    for text, yf, color in pts:
        if merged and merged[-1][0] == text and yf - merged[-1][1] < LABEL_GAP:
            merged[-1][2] = muted
            continue
        merged.append([text, yf, color])
    x_frac = to_axes.transform((x_end, 1.0))[0]
    for (text, yf, color), y_new in zip(merged, spread_labels([m[1] for m in merged])):
        ax.annotate(text, xy=(x_frac, yf), xycoords="axes fraction", xytext=(x_frac + 0.03, y_new),
                    textcoords="axes fraction", va="center", fontsize=8, color=color,
                    arrowprops=None if abs(y_new - yf) < 0.01 else
                    {"arrowstyle": "-", "color": color, "linewidth": 0.6, "shrinkA": 0, "shrinkB": 4})


def write_plot(path: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e5e7eb"
    colors = {"tier1": "#2a78d6", "kmeans": "#eb6834", "pca": "#1baf7a", "ae_v2": "#eda100"}  # fixed slot order
    ws = sorted({r["window_min"] for r in rows})
    by = {(r["key"], r["window_min"]): r for r in rows}
    # (column, title, label format, log y): Tier 1 is ~100x the others in false-alarm minutes, hence log
    panels = (("composite_f1", "Composite F1 (higher is better)", "{:.2f}", False),
              ("median_delay_min", "Median detection delay, min (lower is better)", "{:.0f}", False),
              ("mean_delay_min", "Mean detection delay, min (lower is better)", "{:.1f}", False),
              ("false_alarm_minutes_per_machine_day",
               "False-alarm minutes / machine / day, log scale (lower is better)", "{:.1f}", True))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    axes = axes.ravel()
    for ax, (col, title, fmt, log_y) in zip(axes, panels):
        for key, color in colors.items():
            ys = [by[(key, w)][col] for w in ws]
            ys = [np.nan if v is None else v for v in ys]
            label = DETECTORS[key][0] + (" = final rule" if key == "ae_v2" else "")
            ax.plot(ws, ys, color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=label)
        ax.set_xscale("log")
        ax.set_xticks(ws)
        ax.set_xticklabels([f"{w} min" for w in ws])
        ax.minorticks_off()
        ax.set_xlim(ws[0] / 1.4, ws[-1] * 1.8)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9, length=0)
        if log_y:
            ax.set_yscale("log")
            ax.minorticks_off()
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylim(ax.get_ylim())          # freeze the limits so the label positions below stay valid
        _end_labels(ax, [(fmt.format(by[(key, ws[-1])][col]), by[(key, ws[-1])][col], color)
                         for key, color in colors.items() if by[(key, ws[-1])][col] is not None
                         and not math.isnan(by[(key, ws[-1])][col])], ws[-1], muted)
    for ax in axes[2:]:
        ax.set_xlabel("window size", fontsize=9, color=muted)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("E4 on SMD: detectors re-fitted per window size (test windows)", fontsize=11, color=ink,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def print_table(rows: list[dict]) -> None:
    view = pd.DataFrame([{
        "window": f"{r['window_min']} min", "detector": r["detector"], "comp_F1": r["composite_f1"],
        "caught": f"{r['faults_caught']}/{r['faults']}",
        "caught_5min": f"{r['ref_faults_caught']}/{r['ref_faults']}" if r["ref_faults"] is not None else "n/a",
        "alarm_P": r["alarm_precision"], "FA/mach/day": r["false_alarms_per_machine_day"],
        "FA_min/mach/day": r["false_alarm_minutes_per_machine_day"],
        "med_delay": r["median_delay_min"], "mean_delay": r["mean_delay_min"], "PR-AUC": r["pr_auc"],
        "flagged": f"{r['windows_flagged']}/{r['windows']}"} for r in rows])
    with pd.option_context("display.width", 250, "display.max_rows", None, "display.float_format", "{:.3f}".format):
        print(view.to_string(index=False))


# ---------------------------------------------------------------- driver

def run(windows: list[int] = list(WINDOWS), sample: int | None = None, root: Path = ROOT,
        spark: SparkSession | None = None, epochs: int = A.MAX_EPOCHS, out_dir: Path | None = None) -> dict:
    spark = spark or build_spark()
    raw_dir = root / S.PATHS["raw"] / f"source={SOURCE}"
    faults = reference_faults(raw_dir, sample)
    print(f"reference faults on {REFERENCE_WINDOW}-minute test windows: {sum(len(v) for v in faults.values())}")
    runs, rows, per_machine, check = [], [], {}, None
    for w in sorted(windows):
        r = run_window(spark, w, root, sample, epochs)
        wr, pm = evaluate_window(r["df"], r["pca_threshold"], w, faults)
        runs.append(r)
        rows += wr
        per_machine[w] = pm
        if w == REFERENCE_WINDOW and (root / E1_JSON).exists():
            check = check_e1(wr, pm, json.loads((root / E1_JSON).read_text()), sample)
            print_check(check)

    out = out_dir or default_out(root, sample)
    write_report(out, rows, runs, check, sample, per_machine)
    return {"rows": rows, "runs": runs, "check": check, "per_machine": per_machine, "out": out}


def default_out(root: Path, sample: int | None) -> Path:
    return root / (DATA_DIR / f"sample{sample}" if sample else OUT_DIR)


def write_report(out: Path, rows: list[dict], runs: list[dict], check: dict | None, sample: int | None,
                 per_machine: dict, report_only: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_markdown(out / "e4.md", rows, runs, check, sample, report_only)
    write_json(out / "e4.json", rows, runs, check, sample, per_machine)
    write_plot(out / "e4.png", rows)
    print("\n=== E4 table (test windows)")
    print_table(rows)
    print("\nrun time per window size (training run): "
          + ", ".join(f"{r['window']} min {r['info']['seconds']:.0f} s" for r in runs))
    print(f"wrote {(out / 'e4.md').as_posix()}, e4.json, e4.png")


def load_scores(path: Path) -> pd.DataFrame:
    """A saved data/e4/w<N> scores file back as the frame evaluate_window expects (ws_us = window_start in us)."""
    t = pq.read_table(path)
    df = t.drop(["window_start", "window_end"]).to_pandas()
    df.insert(1, "ws_us", t["window_start"].cast(pa.int64()).to_numpy())
    return df.sort_values(["entity_id", "ws_us"]).reset_index(drop=True)


def report_only(windows: list[int] | None = None, sample: int | None = None, root: Path = ROOT,
                out_dir: Path | None = None) -> dict:
    """Rebuild e4.md / e4.json / e4.png from the saved scores under data/e4/w<N>/ without training anything.
    Run settings (train windows, k, epochs, run time, ...) come from the existing e4.json in the output folder;
    the PCA flag threshold from models/e4/w<N>/pca_smd/thresholds.json (the saved scores have no PCA flag)."""
    out = out_dir or default_out(root, sample)
    prev_path = out / "e4.json"
    if not prev_path.exists():
        raise FileNotFoundError(f"{prev_path.as_posix()} not found: run the full experiment once first")
    prev = json.loads(prev_path.read_text())
    if prev.get("sample") != sample:
        raise ValueError(f"{prev_path.as_posix()} is from sample={prev.get('sample')}, not sample={sample}")
    settings = {s["window_min"]: s for s in prev["settings"]}
    windows = sorted(windows or settings)
    faults = reference_faults(root / S.PATHS["raw"] / f"source={SOURCE}", sample)
    print(f"reference faults on {REFERENCE_WINDOW}-minute test windows: {sum(len(v) for v in faults.values())}")
    runs, rows, per_machine, check = [], [], {}, None
    for w in windows:
        if w not in settings:
            raise ValueError(f"window {w} min is not in {prev_path.as_posix()}")
        info = {k: v for k, v in settings[w].items() if k != "window_min"}
        df = load_scores(root / DATA_DIR / f"w{w}" / "scores" / f"source={SOURCE}" / "scores.parquet")
        if len(df) != info["test_windows"]:
            raise ValueError(f"data/e4/w{w} has {len(df)} test windows, {prev_path.as_posix()} says "
                             f"{info['test_windows']}: the saved scores are from another run (--sample?)")
        thr_path = root / MODEL_DIR / f"w{w}" / f"pca_{SOURCE}" / "thresholds.json"
        pca_thr = json.loads(thr_path.read_text())["global"] if thr_path.exists() else info["threshold_pca"]
        print(f"window {w} min: {len(df):,} saved test windows, PCA threshold {pca_thr:.4f}")
        wr, pm = evaluate_window(df, pca_thr, w, faults)
        runs.append({"window": w, "info": info})
        rows += wr
        per_machine[w] = pm
        if w == REFERENCE_WINDOW and (root / E1_JSON).exists():
            check = check_e1(wr, pm, json.loads((root / E1_JSON).read_text()), sample)
            print_check(check)
    write_report(out, rows, runs, check, sample, per_machine, report_only=True)
    return {"rows": rows, "runs": runs, "check": check, "per_machine": per_machine, "out": out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS), help="window sizes in minutes")
    parser.add_argument("--sample", type=int, default=None, help="use only the first N machines")
    parser.add_argument("--driver-memory", default="4g", help="Spark driver memory (1-minute windows need more)")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild e4.md/e4.json/e4.png from the saved scores under data/e4/w<N>/ (no training)")
    args = parser.parse_args()
    if any(w <= 0 for w in args.windows):
        parser.error("window sizes must be positive minutes")
    if args.report_only:
        result = report_only(args.windows, args.sample, ROOT)
        return 0 if result["check"] is None or result["check"]["passed"] else 1
    spark = build_spark(args.driver_memory)
    spark.sparkContext.setLogLevel("ERROR")
    try:
        result = run(args.windows, args.sample, ROOT, spark)
    finally:
        spark.stop()
    return 0 if result["check"] is None or result["check"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
