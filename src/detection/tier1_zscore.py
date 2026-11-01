"""Tier 1 robust z-score per machine-metric on 5-minute SMD windows (offline, Spark).

Run from the repo root:
    python src/detection/tier1_zscore.py
    python src/detection/tier1_zscore.py --sample 2      # first 2 machines, quick test

Steps:
1. Read data/raw/source=smd/*.parquet (fake.parquet is skipped) and build 5-minute tumbling windows per
   entity_id (epoch-aligned, the same windows as baselines.py and autoencoder.py). Each metric in the
   metrics map is averaged over the rows of the window that have it (long format: one row per
   entity_id, window, metric). A window is `test` if any of its rows is a test row; test windows get
   label = max(label) (null -> 0).
2. Per machine-metric, from train windows only: median and MAD = median(|x - median|); scale = 1.4826 x MAD.
   Guard for MAD = 0 (--mad-zero): `skip` (default, what the stream scoring job does) leaves that
   machine-metric unscored; `fallback` uses 1.2533 x mean absolute deviation, then SCALE_FLOOR if that is 0 too.
   Saved as the single file models/tier1_smd/baselines.parquet with exactly entity_id, metric, median, mad
   (the contract with the stream job: z = |mean - median| / (1.4826 x mad), metrics with mad = 0 skipped).
3. Test windows: z = |x - median| / scale per metric; tier1_score = max z over the metrics of the window,
   tier1_flag = tier1_score > 3. A machine-metric without train windows is not scored; a window without
   any scored metric has a null score and is not flagged.
4. Output: data/scores_tier1/source=smd with entity_id, window_start, window_end, tier1_score, tier1_flag, label
   (baselines.py joins it into the scores layer).
5. Prints point-adjusted event precision/recall/F1 from src/detection/evaluate.py, per machine on
   time-ordered test windows and overall.
"""
import argparse
import os
import shutil
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
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.detection.evaluate import event_metrics  # noqa: E402

METRICS = {"smd": S.SMD_METRICS}
WINDOW = "5 minutes"
SKIP_FILES = {"fake.parquet"}
SCORES_TIER1_DIR = "data/scores_tier1"
MODEL_DIR = "models/tier1_<source>"

BASELINES_FILE = "baselines.parquet"
BASELINE_COLUMNS = ["entity_id", "metric", "median", "mad"]   # agreed with the stream scoring job
MAD_ZERO = "skip"              # the stream skips metrics with MAD = 0; "fallback" is kept for comparison

MAD_TO_SIGMA = 1.4826         # MAD -> standard deviation for normal data
MEAN_AD_TO_SIGMA = 1.2533      # mean absolute deviation -> standard deviation for normal data (sqrt(pi/2))
SCALE_FLOOR = 1e-3             # last resort for a metric that is constant in train
THRESHOLD = 3.0

WINDOW_KEYS = ["entity_id", "window_start", "window_end"]


def build_spark(app_name: str = "tier1_zscore", driver_memory: str = "4g") -> SparkSession:
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def raw_files(raw_dir: Path, sample: int | None) -> list[Path]:
    files = sorted(p for p in raw_dir.glob("*.parquet") if p.name not in SKIP_FILES)
    if sample:
        files = files[:sample]
    if not files:
        raise FileNotFoundError(f"no machine Parquet files under {raw_dir}")
    return files


def build_windows(raw: DataFrame, metrics: list[str]) -> tuple[DataFrame, DataFrame]:
    """Returns (means, meta).

    means: entity_id, window_start, window_end, metric, x (window mean of the metric).
    meta:  entity_id, window_start, window_end, split ('train'|'test'), label (test windows only, null -> 0).
    """
    rows = (raw.withColumn("w", F.window("event_time", WINDOW))
            .select("entity_id", F.col("w.start").alias("window_start"), F.col("w.end").alias("window_end"),
                    "metrics", F.coalesce(F.col("attrs")["split"] == "test", F.lit(False)).alias("is_test"),
                    "label"))
    means = (rows.select(*WINDOW_KEYS, F.explode("metrics").alias("metric", "value"))
             .filter(F.col("value").isNotNull() & ~F.isnan("value") & F.col("metric").isin(metrics))
             .groupBy(*WINDOW_KEYS, "metric").agg(F.avg("value").alias("x")))
    meta = (rows.groupBy(*WINDOW_KEYS)
            .agg(F.max(F.col("is_test").cast("int")).alias("is_test"), F.max("label").alias("label"))
            .select(*WINDOW_KEYS,
                    F.when(F.col("is_test") == 1, "test").otherwise("train").alias("split"),
                    F.when(F.col("is_test") == 1, F.coalesce("label", F.lit(0))).cast("int").alias("label")))
    return means, meta


def fit_stats(means: DataFrame, meta: DataFrame, mad_zero: str = MAD_ZERO) -> DataFrame:
    """Per machine-metric median, MAD and scale from train windows only.

    mad_zero 'skip': scale is null when MAD = 0, so that machine-metric is not scored (what the stream does).
    mad_zero 'fallback': scale from the mean absolute deviation, then SCALE_FLOOR."""
    train = (means.join(meta.filter(F.col("split") == "train").select(*WINDOW_KEYS), WINDOW_KEYS)
             .select("entity_id", "metric", "x"))
    med = (train.groupBy("entity_id", "metric")
           .agg(F.expr("percentile(x, 0.5)").alias("median"), F.count("x").alias("n_train")))
    dev = (train.join(med, ["entity_id", "metric"])
           .withColumn("ad", F.abs(F.col("x") - F.col("median")))
           .groupBy("entity_id", "metric")
           .agg(F.expr("percentile(ad, 0.5)").alias("mad"), F.avg("ad").alias("mean_ad")))
    scale = F.when(F.col("mad") > 0, MAD_TO_SIGMA * F.col("mad"))
    if mad_zero == "fallback":
        scale = (scale.when(F.col("mean_ad") > 0, MEAN_AD_TO_SIGMA * F.col("mean_ad"))
                 .otherwise(F.lit(SCALE_FLOOR)))
    return (med.join(dev, ["entity_id", "metric"])
            .select("entity_id", "metric", "median", "mad", scale.alias("scale"), "n_train"))


def score_windows(means: DataFrame, meta: DataFrame, stats: DataFrame) -> DataFrame:
    """Test windows with tier1_score = max over metrics of |x - median| / scale and tier1_flag = score > 3."""
    test = meta.filter(F.col("split") == "test")
    z = (means.join(test.select(*WINDOW_KEYS), WINDOW_KEYS)
         .join(stats.filter(F.col("scale").isNotNull()), ["entity_id", "metric"])
         .withColumn("z", F.abs(F.col("x") - F.col("median")) / F.col("scale"))
         .groupBy(*WINDOW_KEYS).agg(F.max("z").alias("tier1_score")))
    return (test.join(z, WINDOW_KEYS, "left")
            .withColumn("tier1_flag", F.coalesce(F.col("tier1_score") > THRESHOLD, F.lit(False)))
            .select(*WINDOW_KEYS, "tier1_score", "tier1_flag", "label"))


def evaluate(test_pdf: pd.DataFrame) -> dict:
    """Event-level (point-adjusted) metrics per machine and overall, via src/detection/evaluate.py."""
    rows, labels, flags = [], [], []
    for entity, g in test_pdf.sort_values(["entity_id", "window_start"]).groupby("entity_id"):
        t, p = g["label"].to_numpy(dtype=int), g["tier1_flag"].to_numpy(dtype=int)
        rows.append({"machine": entity, "windows": len(g), "flagged": int(p.sum()), **event_metrics(t, p)})
        labels += [*t, 0]     # a 0 between machines keeps their anomaly segments apart
        flags += [*p, 0]
    return {"overall": event_metrics(labels, flags), "per_machine": pd.DataFrame(rows)}


def save_baselines(stats: pd.DataFrame, model_dir: Path) -> Path:
    """models/tier1_smd/baselines.parquet: one file, exactly entity_id, metric, median, mad (the stream's input)."""
    shutil.rmtree(model_dir, ignore_errors=True)
    model_dir.mkdir(parents=True)
    path = model_dir / BASELINES_FILE
    table = pa.Table.from_pandas(stats[BASELINE_COLUMNS].sort_values(["entity_id", "metric"]), preserve_index=False)
    pq.write_table(table, path)
    return path


def run(source: str = "smd", sample: int | None = None, root: Path = ROOT,
        spark: SparkSession | None = None, mad_zero: str = MAD_ZERO) -> dict:
    metrics = METRICS[source]
    spark = spark or build_spark()
    files = raw_files(root / S.PATHS["raw"] / f"source={source}", sample)
    raw = spark.read.parquet(*[p.as_posix() for p in files])

    means, meta = build_windows(raw, metrics)
    means, meta = means.cache(), meta.cache()
    stats = fit_stats(means, meta, mad_zero).cache()
    stats_pdf = stats.toPandas()
    baselines = save_baselines(stats_pdf, root / MODEL_DIR.replace("<source>", source))
    print(f"{len(files)} machines, {len(stats_pdf)} machine-metrics fitted on train windows, "
          f"{int((stats_pdf['mad'] == 0).sum())} with MAD = 0 (mad_zero={mad_zero}); "
          f"saved {baselines.relative_to(root).as_posix()}")

    scored = score_windows(means, meta, stats).cache()
    out_dir = root / SCORES_TIER1_DIR / f"source={source}"
    scored.coalesce(1).write.mode("overwrite").parquet(out_dir.as_posix())
    test_pdf = scored.orderBy("entity_id", "window_start").toPandas()
    print(f"wrote {len(test_pdf)} test windows to {out_dir.relative_to(root).as_posix()} "
          f"(tier1 flagged {int(test_pdf['tier1_flag'].sum())}, threshold z > {THRESHOLD:g})")

    result = evaluate(test_pdf)
    o = result["overall"]
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n=== Tier 1 robust z-score, event-level (point-adjusted) per machine")
        print(result["per_machine"].to_string(index=False))
    print(f"\nTier 1 event-level overall: precision {o['precision']:.3f}  recall {o['recall']:.3f}  "
          f"F1 {o['f1']:.3f}")
    for df in (scored, stats, means, meta):
        df.unpersist()
    return {"test": test_pdf, "stats": stats_pdf, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=sorted(METRICS), default="smd")
    parser.add_argument("--sample", type=int, default=None, help="use only the first N machines")
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/ and models/")
    parser.add_argument("--mad-zero", choices=["skip", "fallback"], default=MAD_ZERO,
                        help="metrics with MAD = 0: skip them (as the stream does) or use a fallback scale")
    parser.add_argument("--driver-memory", default="4g", help="Spark driver memory, e.g. 2g on a small laptop")
    args = parser.parse_args()
    spark = build_spark(driver_memory=args.driver_memory)
    spark.sparkContext.setLogLevel("ERROR")
    try:
        run(args.source, args.sample, args.root, spark, args.mad_zero)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
