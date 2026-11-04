"""Tier 2 (MLlib KMeans) and Tier 2b (MLlib PCA) baselines on 5-minute SMD windows (offline, Spark).

Run from the repo root:
    python src/detection/baselines.py                 # all SMD machines
    python src/detection/baselines.py --sample 2      # first 2 machines, quick test
    python src/detection/baselines.py --kmax 6
    python src/detection/baselines.py --recenter 24   # test windows centred on a rolling median

Steps:
1. Read data/raw/source=smd/*.parquet (fake.parquet is skipped) and build 5-minute tumbling windows per
   entity_id with Spark, the same windows as src/detection/autoencoder.py: window_start = event_time
   floored to 5 minutes (epoch-aligned), window_end = window_start + 5 minutes. The metrics map is exploded
   and each of m00..m37 is averaged over the rows that have it, then pivoted to 38 columns (null = no
   value in the window). A window is `test` if any of its rows is a test row; test windows get
   label = max(label) (null -> 0), train windows null.
2. Standardise per entity_id with the mean and population std of its train windows (Spark aggregation +
   join; std 0 or undefined -> 1, undefined mean -> 0); a missing value becomes 0 after standardising.
   --recenter N (default 0 = off): test windows are centred on the median of the machine's previous N windows
   (train or test; current window excluded; at least N // 2 values, else the train mean) instead of the train
   mean; std stays from train. Computed per machine with applyInPandas and recenter_centers from
   autoencoder.py. Train windows, the fitted models and the thresholds do not change. Saved in scaler.json.
3. Tier 2: Pipeline VectorAssembler -> KMeans (seed 42). k in 3..--kmax is chosen by the silhouette
   (ClusteringEvaluator) of a model fitted on a 20% sample of train windows; the final model is fitted on
   all train windows. tier2_score = Euclidean distance to the nearest centroid.
4. Tier 2b: Pipeline VectorAssembler -> PCA with the fewest components explaining >= 90% of the train
   variance. pca_score = mean squared error of x - x P P^T over the 38 metrics (mapInPandas + numpy).
   Spark's PCA does not centre; the standardised train windows have mean 0, so no centring is needed.
5. Thresholds as for the autoencoder: 99.5th percentile of the train score (all machines) x 1.1.
   tier2_flag = tier2_score > threshold; pca_flag likewise (not in the scores contract: evaluation only).
6. Models: models/kmeans_smd/ and models/pca_smd/ (PipelineModel, model.write().overwrite()), each with
   thresholds.json and scaler.json (per-entity train mean/std, needed by the streaming job).
7. Test windows go to data/scores/source=smd with the scores-layer columns. tier1_score/tier1_flag come
   from data/scores_tier1/source=smd and ae_score/ae_flag/top_metrics from data/scores_ae/source=smd,
   joined on (entity_id, window_start) when those folders exist, else null.
   is_anomaly = coalesce(ae_flag, false): the autoencoder alone raises the alarm, as in
   src/streaming/score_job.py (see docs/results/tier1_stream.md). tier1_score/tier1_flag are kept in the
   output as a baseline and explainer only; they never change is_anomaly.
8. For Tier 2, Tier 2b and the autoencoder (if present), the same report as autoencoder.py: composite F1,
   faults caught, alarm precision, false alarms per machine per day, median delay, PR-AUC of the score,
   point and point-adjusted event precision/recall/F1, per machine on time-ordered test windows and overall.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# On Windows, Spark must start workers with this same venv Python (as in tools/check_spark_kafka.py).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.ml import Pipeline, PipelineModel  # noqa: E402
from pyspark.ml.clustering import KMeans  # noqa: E402
from pyspark.ml.evaluation import ClusteringEvaluator  # noqa: E402
from pyspark.ml.feature import PCA, VectorAssembler  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from contracts import schemas as S  # noqa: E402
# Reused so the baseline rows are printed exactly like the AE v2 rows.
from src.detection.autoencoder import SCORES_AE_DIR, SKIP_FILES, _print_report, recenter_centers  # noqa: E402
from src.detection.autoencoder import evaluate as evaluate_flags  # noqa: E402

METRICS = {"smd": S.SMD_METRICS}
WINDOW = "5 minutes"
SCORES_TIER1_DIR = "data/scores_tier1"
KEYS = ["entity_id", "window_start"]

SEED = 42
K_MIN = 3
K_MAX = 10
SILHOUETTE_FRACTION = 0.20
PCA_VARIANCE = 0.90
THRESHOLD_PERCENTILE = 99.5
THRESHOLD_FACTOR = 1.1
RECENTER = 0

# Tier name -> (score column, flag column) for the evaluation report.
TIERS = {"Tier 2 KMeans": ("tier2_score", "tier2_flag"),
         "Tier 2b PCA": ("pca_score", "pca_flag"),
         "Tier 3 AE": ("ae_score", "ae_flag")}


def build_spark(app_name: str = "baselines") -> SparkSession:
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


# ---------------------------------------------------------------- windows

def raw_files(raw_dir: Path, sample: int | None) -> list[Path]:
    files = sorted(p for p in raw_dir.glob("*.parquet") if p.name not in SKIP_FILES)
    if sample:
        files = files[:sample]
    if not files:
        raise FileNotFoundError(f"no machine Parquet files under {raw_dir}")
    return files


def build_windows(raw: DataFrame, metrics: list[str], window: str = WINDOW) -> DataFrame:
    """Window means (5 minutes unless `window` says otherwise), one column per metric (null = no value in the window).

    Columns: entity_id, window_start, window_end, ws_us (window_start in microseconds), split, label, *metrics.
    """
    rows = (raw.withColumn("w", F.window("event_time", window))
            .select("entity_id", F.col("w.start").alias("window_start"), F.col("w.end").alias("window_end"),
                    "metrics", F.coalesce(F.col("attrs")["split"] == "test", F.lit(False)).alias("is_test"),
                    "label"))
    keys = ["entity_id", "window_start", "window_end"]
    values = (rows.select(*keys, F.explode("metrics").alias("metric", "value"))
              .filter(F.col("value").isNotNull() & ~F.isnan("value") & F.col("metric").isin(metrics))
              .groupBy(*keys).pivot("metric", metrics).agg(F.avg("value")))
    meta = (rows.groupBy(*keys)
            .agg(F.max(F.col("is_test").cast("int")).alias("is_test"), F.max("label").alias("label"))
            .select(*keys,
                    F.when(F.col("is_test") == 1, "test").otherwise("train").alias("split"),
                    F.when(F.col("is_test") == 1, F.coalesce("label", F.lit(0))).cast("int").alias("label")))
    return (meta.join(values, keys, "left")
            .select(*keys, F.expr("unix_micros(window_start)").alias("ws_us"), "split", "label", *metrics))


def scaler_stats(windows: DataFrame, metrics: list[str]) -> DataFrame:
    """Per entity_id: <metric>__mean and <metric>__std over its train windows (std 0/undefined -> 1)."""
    aggs = ([F.avg(m).alias(f"{m}__mean") for m in metrics]
            + [F.stddev_pop(m).alias(f"{m}__std") for m in metrics])
    stats = windows.filter(F.col("split") == "train").groupBy("entity_id").agg(*aggs)
    return stats.select(
        "entity_id",
        *[F.coalesce(F.col(f"{m}__mean"), F.lit(0.0)).alias(f"{m}__mean") for m in metrics],
        *[F.when(F.col(f"{m}__std").isNull() | (F.col(f"{m}__std") == 0), 1.0)
          .otherwise(F.col(f"{m}__std")).alias(f"{m}__std") for m in metrics])


def recenter_info(windows: int) -> dict | None:
    """The recenter setting as saved in scaler.json (same keys as autoencoder.py); None when off."""
    return ({"windows": windows, "min_periods": max(1, windows // 2), "statistic": "median",
             "applies_to": "test", "fallback": "train mean"} if windows else None)


def recenter_stats(windows: DataFrame, stats: DataFrame, metrics: list[str], n: int) -> DataFrame:
    """Per test window: <metric>__center = median of the machine's previous n windows (train and test,
    current one excluded, at least n // 2 values, else the train mean), via recenter_centers in
    autoencoder.py, run once per entity_id with applyInPandas. Columns: entity_id, ws_us, *<metric>__center."""
    means = [f"{m}__mean" for m in metrics]
    centers = [f"{m}__center" for m in metrics]
    joined = (windows.join(stats.select("entity_id", *means), "entity_id", "left")
              .select("entity_id", "ws_us", "split", *metrics, *means))
    schema = "entity_id string, ws_us long, " + ", ".join(f"{c} double" for c in centers)

    def _centers(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("ws_us").reset_index(drop=True)
        X = pdf[metrics].to_numpy(dtype=np.float64)                    # null -> NaN, skipped by the median
        mean = np.nan_to_num(pdf[means].iloc[0].to_numpy(dtype=np.float64), nan=0.0)  # no train windows -> 0
        C = recenter_centers(pdf[["entity_id", "split"]], X, {pdf["entity_id"].iloc[0]: {"mean": mean}}, n)
        test = (pdf["split"] == "test").to_numpy()
        out = pd.DataFrame(C[test], columns=centers)
        out.insert(0, "ws_us", pdf.loc[test, "ws_us"].to_numpy())
        out.insert(0, "entity_id", pdf.loc[test, "entity_id"].to_numpy())
        return out

    return joined.groupBy("entity_id").applyInPandas(_centers, schema)


def standardise(windows: DataFrame, stats: DataFrame, metrics: list[str],
                centers: DataFrame | None = None) -> DataFrame:
    """Metric columns replaced by (value - mean) / std; a machine without train windows uses mean 0, std 1;
    a missing value becomes 0. With centers (from recenter_stats), test windows use <metric>__center
    instead of the train mean; train windows always use the train mean."""
    joined = windows.join(stats, "entity_id", "left")
    if centers is not None:
        joined = joined.join(centers, ["entity_id", "ws_us"], "left")

    def center(m: str):
        mean = F.coalesce(F.col(f"{m}__mean"), F.lit(0.0))
        if centers is None:
            return mean
        return F.when(F.col("split") == "test", F.coalesce(F.col(f"{m}__center"), mean)).otherwise(mean)

    z = [F.coalesce((F.col(m) - center(m)) / F.coalesce(F.col(f"{m}__std"), F.lit(1.0)), F.lit(0.0)).alias(m)
         for m in metrics]
    return joined.select("entity_id", "window_start", "window_end", "ws_us", "split", "label", *z)


def stats_to_json(stats: DataFrame, metrics: list[str]) -> dict:
    return {r["entity_id"]: {"mean": [r[f"{m}__mean"] for m in metrics], "std": [r[f"{m}__std"] for m in metrics]}
            for r in stats.collect()}


# ---------------------------------------------------------------- models

def _assembler(metrics: list[str]) -> VectorAssembler:
    return VectorAssembler(inputCols=metrics, outputCol="features")


def kmeans_pipeline(metrics: list[str], k: int) -> Pipeline:
    return Pipeline(stages=[_assembler(metrics), KMeans(k=k, seed=SEED, featuresCol="features",
                                                        predictionCol="cluster")])


def pca_pipeline(metrics: list[str], k: int) -> Pipeline:
    return Pipeline(stages=[_assembler(metrics), PCA(k=k, inputCol="features", outputCol="pca_features")])


def fit_kmeans(train: DataFrame, metrics: list[str], kmax: int = K_MAX) -> tuple[PipelineModel, int, dict]:
    """k in K_MIN..kmax by silhouette on a SILHOUETTE_FRACTION sample of train windows; final fit on all."""
    sample = train.sample(fraction=SILHOUETTE_FRACTION, seed=SEED).cache()
    n = sample.count()
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette")
    silhouette = {}
    for k in range(K_MIN, kmax + 1):
        if k >= n:
            break
        model = kmeans_pipeline(metrics, k).fit(sample)
        silhouette[k] = float(evaluator.evaluate(model.transform(sample)))
        print(f"  kmeans k={k:2d}  silhouette {silhouette[k]:.4f}  ({n} sampled train windows)")
    sample.unpersist()
    if not silhouette:
        raise ValueError(f"too few sampled train windows ({n}) to try k >= {K_MIN}")
    best = max(silhouette, key=silhouette.get)
    return kmeans_pipeline(metrics, best).fit(train), best, silhouette


def fit_pca(train: DataFrame, metrics: list[str], variance: float = PCA_VARIANCE) -> tuple[PipelineModel, int, list]:
    """Fewest components whose explained variance on train windows reaches `variance`."""
    full = pca_pipeline(metrics, len(metrics)).fit(train)
    cumulative = np.cumsum(full.stages[-1].explainedVariance.toArray())
    k = int(np.searchsorted(cumulative, variance - 1e-12) + 1)
    k = min(k, len(metrics))
    return pca_pipeline(metrics, k).fit(train), k, cumulative.tolist()


def kmeans_distance(Z: np.ndarray, centers: np.ndarray, cluster: np.ndarray | None = None) -> np.ndarray:
    """tier2_score: Euclidean distance of each standardised window to its centroid (cluster = the KMeans
    prediction; None -> the nearest centroid, which is what KMeansModel predicts)."""
    if cluster is None:
        cluster = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return np.sqrt(((Z - centers[cluster]) ** 2).sum(axis=1))


def pca_projector(pc: np.ndarray) -> np.ndarray:
    """P P^T for the n_metrics x k principal components (orthonormal columns)."""
    return pc @ pc.T


def pca_error(Z: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """pca_score: mean squared error of x - x P P^T over the metrics."""
    return ((Z - Z @ projector) ** 2).mean(axis=1)


def score_windows(z: DataFrame, kmeans: PipelineModel, pca: PipelineModel, metrics: list[str]) -> DataFrame:
    """Adds tier2_score (distance to the nearest centroid) and pca_score (mean squared reconstruction
    error) to z; the numpy work runs in mapInPandas."""
    centers = np.asarray(kmeans.stages[-1].clusterCenters())
    projector = pca_projector(pca.stages[-1].pc.toArray())

    def _scores(batches):
        for pdf in batches:
            Z = pdf[metrics].to_numpy(dtype=np.float64)
            tier2 = kmeans_distance(Z, centers, pdf["cluster"].to_numpy())
            pca_err = pca_error(Z, projector)
            yield pd.DataFrame({"entity_id": pdf["entity_id"], "ws_us": pdf["ws_us"],
                                "tier2_score": tier2, "pca_score": pca_err})

    clustered = kmeans.transform(z).select("entity_id", "ws_us", "cluster", *metrics)
    scores = clustered.mapInPandas(_scores, "entity_id string, ws_us long, tier2_score double, pca_score double")
    return z.drop(*metrics).join(scores, ["entity_id", "ws_us"])


def threshold(scored: DataFrame, column: str) -> float:
    """THRESHOLD_PERCENTILE of the train scores (all machines, exact, linear interpolation like numpy)
    x THRESHOLD_FACTOR."""
    p = (scored.filter(F.col("split") == "train")
         .agg(F.expr(f"percentile({column}, {THRESHOLD_PERCENTILE / 100})")).first()[0])
    return float(p) * THRESHOLD_FACTOR


def save_model(model: PipelineModel, model_dir: Path, info: dict, scaler: dict) -> None:
    model.write().overwrite().save(model_dir.as_posix())
    (model_dir / "thresholds.json").write_text(json.dumps(info, indent=2))
    (model_dir / "scaler.json").write_text(json.dumps(scaler))


# ---------------------------------------------------------------- scores layer

def _read_optional(spark: SparkSession, folder: Path, columns: list[str]) -> DataFrame | None:
    if not folder.exists() or not any(folder.rglob("*.parquet")):
        return None
    df = spark.read.parquet(folder.as_posix()).select(*KEYS, *columns)
    return df.dropDuplicates(KEYS)


def attach_other_tiers(spark: SparkSession, test: DataFrame, root: Path, source: str) -> tuple[DataFrame, dict]:
    """Left-joins tier1 and AE columns on (entity_id, window_start); null columns when a folder is missing."""
    found = {}
    parts = {SCORES_TIER1_DIR: {"tier1_score": "double", "tier1_flag": "boolean"},
             SCORES_AE_DIR: {"ae_score": "double", "ae_flag": "boolean", "top_metrics": "array<string>"}}
    for folder, cols in parts.items():
        path = root / folder / f"source={source}"
        other = _read_optional(spark, path, list(cols))
        found[folder] = other is not None
        if other is None:
            print(f"note: {folder}/source={source} not found, {', '.join(cols)} left null")
            test = test.select("*", *[F.lit(None).cast(t).alias(c) for c, t in cols.items()])
        else:
            test = test.join(other, KEYS, "left")
    test = test.withColumn("is_anomaly", F.coalesce("ae_flag", F.lit(False)))   # AE only; Tier 1 never alarms
    return test, found


def scores_layer(test: DataFrame) -> DataFrame:
    types = {"string": "string", "ts": "timestamp", "double": "double", "bool": "boolean",
             "list_s": "array<string>", "int": "int"}
    return test.select(*[F.col(c).cast(types[code]).alias(c) for c, code in S.LAYERS["scores"].items()])


# ---------------------------------------------------------------- evaluation

def report(test_pdf: pd.DataFrame, tiers: dict = TIERS) -> dict:
    """Prints the autoencoder.py report for each tier whose score column has values; returns overall metrics."""
    meta = test_pdf.rename(columns={"ws_us": "window_start"})[["entity_id", "window_start", "split", "label"]]
    meta = meta.assign(label=meta["label"].astype("float64"))
    results = {}
    for name, (score_col, flag_col) in tiers.items():
        if test_pdf[score_col].isna().all():
            print(f"\n=== {name}: no {score_col} values, skipped")
            continue
        missing = int(test_pdf[score_col].isna().sum())
        flag = test_pdf[flag_col].map(lambda v: v is True or v is np.True_).to_numpy(dtype=bool)
        score = test_pdf[score_col].to_numpy(dtype=np.float64)
        per_machine, overall = evaluate_flags(meta, flag, score)
        note = f", {missing} test windows without a score (counted as not flagged)" if missing else ""
        print(f"\n=== {name} ({score_col}, {flag_col}){note}")
        _print_report(per_machine, overall)
        with pd.option_context("display.max_rows", None, "display.width", 250,
                               "display.float_format", "{:.3f}".format):
            print(per_machine[["entity_id", "flagged", "pt_p", "pt_r", "pt_f1", "ev_p", "ev_r", "ev_f1"]]
                  .rename(columns={"entity_id": "machine"}).to_string(index=False))
        print(f"windows flagged {int(flag.sum())} of {len(flag)}")
        results[name] = {**overall, "flagged": int(flag.sum()), "per_machine": per_machine}
    if results:
        rows = [{"tier": name, "composite_f1": r["composite"]["composite_f1"],
                 "caught": f"{r['composite']['events_detected']}/{r['composite']['events']}",
                 "alarm_prec": r["composite"]["alarm_precision"], "false_alarms/day": r["fa_day"],
                 "med_delay_min": r["delay_min"], "pr_auc": r["pr_auc"], "point_f1": r["point"]["f1"],
                 "event_f1(ref)": r["event"]["f1"], "flagged": r["flagged"]} for name, r in results.items()]
        print("\n=== Summary (test windows)")
        with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
            print(pd.DataFrame(rows).to_string(index=False))
    return results


# ---------------------------------------------------------------- driver

def run(source: str = "smd", sample: int | None = None, kmax: int = K_MAX, root: Path = ROOT,
        spark: SparkSession | None = None, recenter: int = RECENTER) -> dict:
    metrics = METRICS[source]
    spark = spark or build_spark()
    files = raw_files(root / S.PATHS["raw"] / f"source={source}", sample)
    raw = spark.read.parquet(*[p.as_posix() for p in files])

    windows = build_windows(raw, metrics).cache()
    stats = scaler_stats(windows, metrics).cache()
    z = standardise(windows, stats, metrics).cache()
    counts = {r["split"]: r["count"] for r in z.groupBy("split").count().collect()}
    print(f"{len(files)} machines, {counts.get('train', 0)} train + {counts.get('test', 0)} test windows")
    scaler = {"metrics": metrics, "window": WINDOW, "recenter": recenter_info(recenter),
              "entities": stats_to_json(stats, metrics)}

    train = z.filter(F.col("split") == "train")
    kmeans, k, silhouette = fit_kmeans(train, metrics, kmax)
    print(f"kmeans: best k = {k} (silhouette {silhouette[k]:.4f})")
    pca, n_comp, cumulative = fit_pca(train, metrics)
    print(f"pca: {n_comp} of {len(metrics)} components explain {cumulative[n_comp - 1]:.3f} of the train variance "
          f"(target {PCA_VARIANCE})")

    # The models are fitted on z above; recentring only changes the test windows that get scored.
    z_score, centers = z, None
    if recenter:
        centers = recenter_stats(windows, stats, metrics, recenter).cache()
        z_score = standardise(windows, stats, metrics, centers).cache()
        print(f"recenter: test windows centred on the median of the previous {recenter} windows "
              f"(at least {max(1, recenter // 2)}, else the train mean)")
    scored = score_windows(z_score, kmeans, pca, metrics).cache()
    thr = {"tier2_score": threshold(scored, "tier2_score"), "pca_score": threshold(scored, "pca_score")}
    scored = (scored.withColumn("tier2_flag", F.col("tier2_score") > thr["tier2_score"])
              .withColumn("pca_flag", F.col("pca_score") > thr["pca_score"]))
    print(f"thresholds (p{THRESHOLD_PERCENTILE} of train score x {THRESHOLD_FACTOR}, global): "
          f"tier2 {thr['tier2_score']:.5f}, pca {thr['pca_score']:.5f}")

    common = {"percentile": THRESHOLD_PERCENTILE, "factor": THRESHOLD_FACTOR, "mode": "global",
              "window": WINDOW, "metrics": metrics}
    kmeans_info = {**common, "score": "tier2_score", "global": thr["tier2_score"], "k": k,
                   "silhouette": {str(key): v for key, v in silhouette.items()},
                   "silhouette_fraction": SILHOUETTE_FRACTION, "seed": SEED}
    pca_info = {**common, "score": "pca_score", "global": thr["pca_score"], "n_components": n_comp,
                "variance_target": PCA_VARIANCE, "cumulative_variance": cumulative}
    kmeans_dir = root / S.MODELS["kmeans"].replace("<source>", source)
    pca_dir = root / S.MODELS["pca"].replace("<source>", source)
    save_model(kmeans, kmeans_dir, kmeans_info, scaler)
    save_model(pca, pca_dir, pca_info, scaler)
    print(f"saved {kmeans_dir.relative_to(root).as_posix()}/ and {pca_dir.relative_to(root).as_posix()}/")

    test, found = attach_other_tiers(spark, scored.filter(F.col("split") == "test"), root, source)
    test = test.cache()
    out_dir = root / S.PATHS["scores"] / f"source={source}"
    scores_layer(test).coalesce(1).write.mode("overwrite").parquet(out_dir.as_posix())

    test_pdf = (test.select("entity_id", "ws_us", "split", "label", "tier1_flag", "tier2_score", "tier2_flag",
                            "pca_score", "pca_flag", "ae_score", "ae_flag", "is_anomaly")
                .orderBy("entity_id", "ws_us").toPandas())
    for folder, col in ((SCORES_TIER1_DIR, "tier1_flag"), (SCORES_AE_DIR, "ae_flag")):
        if found[folder]:
            print(f"{folder}: matched {int(test_pdf[col].notna().sum())} of {len(test_pdf)} test windows")
    print(f"wrote {len(test_pdf)} test windows to {out_dir.relative_to(root).as_posix()} "
          f"(tier2 flagged {int(test_pdf['tier2_flag'].sum())}, pca flagged {int(test_pdf['pca_flag'].sum())}, "
          f"is_anomaly {int(test_pdf['is_anomaly'].sum())})")

    results = report(test_pdf)
    for df in (test, scored, z_score, z, stats, windows) + ((centers,) if centers is not None else ()):
        df.unpersist()
    return {"test": test_pdf, "thresholds": thr, "k": k, "silhouette": silhouette, "n_components": n_comp,
            "results": results, "recenter": recenter,
            "centers": np.asarray(kmeans.stages[-1].clusterCenters()), "pc": pca.stages[-1].pc.toArray()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=sorted(METRICS), default="smd")
    parser.add_argument("--sample", type=int, default=None, help="use only the first N machines")
    parser.add_argument("--kmax", type=int, default=K_MAX, help=f"largest k tried for KMeans (from {K_MIN})")
    parser.add_argument("--recenter", type=int, default=RECENTER,
                        help="centre test windows on the median of the previous N windows (0 = train mean)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/ and models/")
    args = parser.parse_args()
    if args.kmax < K_MIN:
        parser.error(f"--kmax must be at least {K_MIN}")
    if args.recenter < 0:
        parser.error("--recenter must be 0 or more")
    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        run(args.source, args.sample, args.kmax, args.root, spark, recenter=args.recenter)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
