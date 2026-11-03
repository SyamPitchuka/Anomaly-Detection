"""Group anomalous windows into incidents (PySpark), as specified in docs/incident_rules.md.

Run from the repo root:
    python src/incidents/group.py --source smd                # batch: rebuild all incidents (overwrite)
    python src/incidents/group.py --source smd --stream       # streaming: session windows + 10-minute watermark

Input: data/scores/source=<s> (contract columns of the scores layer). Rows with is_anomaly = true are alerts.
Per entity_id, alerts whose window_start is at most 10 minutes (2 x 5-minute windows) after the previous alert
form one session = one incident. Output: data/incidents/source=<s> with the columns of LAYERS["incidents"]:
incident_id = entity_id + '_' + start_time as yyyyMMddHHmm (UTC), start_time, end_time, n_windows,
max_score (highest ae_score, null if none), top_metrics (3 most frequent, ties by name), created_at.

Batch removes duplicate alerts per (entity_id, window_start) keeping the highest ae_score. The stream keeps the first
alert that arrives for a (entity_id, window_start) (Spark cannot choose the highest in a stream); duplicates are
replays of the same message, so the result is the same unless the replay differs.
"""
import argparse
import os
import sys
from pathlib import Path

# On Windows, Spark must start workers with this same venv Python (as in tools/check_spark_kafka.py).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402
from pyspark.sql import DataFrame, SparkSession, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from contracts import schemas as S  # noqa: E402

WINDOW_MINUTES = 5
SESSION_GAP_MINUTES = 2 * WINDOW_MINUTES
# Spark joins two events into one session only if they are strictly less than the gap apart; the rule says "at most
# 10 minutes", so the gap is one millisecond longer. Windows are 5-minute aligned, so nothing else is affected.
SESSION_GAP = f"{SESSION_GAP_MINUTES * 60 * 1000 + 1} milliseconds"
WATERMARK = "10 minutes"
TOP_N = 3
ID_FORMAT = "yyyyMMddHHmm"
STREAM_TRIGGER_SECONDS = 5


def build_spark(app_name: str = "incidents_group", driver_memory: str = "2g") -> SparkSession:
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def _top_metrics(all_metrics):
    """The TOP_N most frequent entries of an array column; ties by name (rule 8)."""
    counted = F.transform(F.array_distinct(all_metrics),
                          lambda m: F.struct((-F.size(F.filter(all_metrics, lambda x: x == m))).alias("neg_count"),
                                             m.alias("metric")))
    return F.transform(F.slice(F.sort_array(counted), 1, TOP_N), lambda c: c["metric"])


def _sessions(alerts: DataFrame) -> DataFrame:
    """One row per incident from alerts that are already unique per (entity_id, window_start)."""
    grouped = (alerts.groupBy("entity_id", F.session_window("window_start", SESSION_GAP).alias("session"))
               .agg(F.min("window_start").alias("start_time"), F.max("window_end").alias("end_time"),
                    F.count(F.lit(1)).alias("n_windows"), F.max("ae_score").alias("max_score"),
                    F.flatten(F.collect_list("top_metrics")).alias("all_metrics")))
    return grouped.select(
        F.concat_ws("_", F.col("entity_id"), F.date_format("start_time", ID_FORMAT)).alias("incident_id"),
        "entity_id", "start_time", "end_time", "n_windows", "max_score",
        _top_metrics(F.col("all_metrics")).alias("top_metrics"),
        F.current_timestamp().alias("created_at"))


def group_incidents(scores: DataFrame) -> DataFrame:
    """Batch: incidents from a scores table (any rows; only is_anomaly = true counts)."""
    alerts = scores.where(F.col("is_anomaly"))
    best_first = Window.partitionBy("entity_id", "window_start").orderBy(F.col("ae_score").desc_nulls_last())
    unique = (alerts.withColumn("_rank", F.row_number().over(best_first)).where(F.col("_rank") == 1).drop("_rank"))
    return _sessions(unique)


def group_incidents_stream(scores: DataFrame) -> DataFrame:
    """Streaming: the same incidents from a streaming scores table. A session is written once the watermark
    (max window_start seen, minus 10 minutes) has passed its end, so each incident appears once, finished."""
    # Dedup first, then filter: Spark would push a filter placed before the dedup below the watermark, and the clock
    # would then run on alerts only, so a finished incident would wait for the next alert before it is written.
    unique = scores.withWatermark("window_start", WATERMARK).dropDuplicates(["entity_id", "window_start"])
    return _sessions(unique.where(F.col("is_anomaly")))


def scores_dir(root: Path, source: str) -> Path:
    return root / S.PATHS["scores"] / f"source={source}"


def incidents_dir(root: Path, source: str) -> Path:
    return root / S.PATHS["incidents"] / f"source={source}"


def check_schema(out_dir: Path) -> None:
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files written under {out_dir}")
    got = pq.read_schema(files[0])
    want = S.LAYERS["incidents"]
    if got.names != list(want):
        raise ValueError(f"incidents columns are {got.names}, contract says {list(want)}")
    for name, code in want.items():
        if not S.type_matches(code, got.field(name).type):
            raise ValueError(f"incidents column '{name}' is {got.field(name).type}, contract says {code}")


def run_batch(source: str, root: Path = ROOT, spark: SparkSession | None = None) -> int:
    """Rebuild data/incidents/source=<source> from data/scores/source=<source>; returns the number of incidents."""
    spark = spark or build_spark()
    scores = spark.read.schema(S.SCORES_DDL).parquet(scores_dir(root, source).as_posix())
    out_dir = incidents_dir(root, source)
    group_incidents(scores).coalesce(1).write.mode("overwrite").parquet(out_dir.as_posix())
    check_schema(out_dir)
    n = spark.read.parquet(out_dir.as_posix()).count()
    print(f"wrote {n} incidents to {out_dir.relative_to(root).as_posix()}")
    return n


def start_stream(source: str, root: Path = ROOT, spark: SparkSession | None = None, once: bool = False):
    """Start the streaming query (file source -> session windows -> Parquet files). once=True processes the files that
    exist now and stops (Trigger.AvailableNow). Returns the StreamingQuery."""
    spark = spark or build_spark()
    scores = spark.readStream.schema(S.SCORES_DDL).parquet(scores_dir(root, source).as_posix())
    writer = (group_incidents_stream(scores).writeStream.format("parquet").outputMode("append")
              .option("path", incidents_dir(root, source).as_posix())
              .option("checkpointLocation", (root / "checkpoints" / f"incidents_{source}").as_posix()))
    writer = writer.trigger(availableNow=True) if once else writer.trigger(processingTime=f"{STREAM_TRIGGER_SECONDS} seconds")
    return writer.start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, choices=["smd", "alibaba"])
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/")
    parser.add_argument("--stream", action="store_true", help="streaming job instead of the batch rebuild")
    parser.add_argument("--once", action="store_true", help="with --stream: process the existing files, then stop")
    parser.add_argument("--driver-memory", default="2g")
    args = parser.parse_args()
    spark = build_spark(driver_memory=args.driver_memory)
    spark.sparkContext.setLogLevel("ERROR")
    try:
        if args.stream:
            start_stream(args.source, args.root, spark, args.once).awaitTermination()
        else:
            run_batch(args.source, args.root, spark)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
