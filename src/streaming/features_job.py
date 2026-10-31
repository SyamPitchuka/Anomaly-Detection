"""Streaming feature job: telemetry.smd from Kafka -> 5-minute window features in the features layer.

Run from the repo root (Kafka must be up and telemetry.smd filled):
    python src/streaming/features_job.py           # runs until stopped, trigger every 30 seconds
    python src/streaming/features_job.py --once    # processes what is in Kafka now, then stops

Output: data/features/source=smd (Parquet, append), checkpoint: checkpoints/features/source=smd.
One row per (entity_id, 5-minute tumbling window, metric) with mean, max, std, slope and n.
slope is the least-squares slope of value vs minutes since window start, built from the sums
n, sum t, sum x, sum t*x and sum t*t so it is a plain streaming aggregation.
Append mode only emits a window once the watermark (max event_time - 10 minutes) has passed its end,
so the newest windows stay in state until later records arrive.
"""
import argparse
import os
import sys
from pathlib import Path

# On Windows, Spark must start workers with this same venv Python.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from contracts import schemas as S  # noqa: E402

KAFKA_PKG = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
SOURCE = "smd"
WINDOW = "5 minutes"
WINDOW_SECONDS = 300
WATERMARK = "10 minutes"
TRIGGER = "30 seconds"
OUTPUT_DIR = ROOT / S.PATHS["features"] / f"source={SOURCE}"
CHECKPOINT_DIR = ROOT / "checkpoints" / "features" / f"source={SOURCE}"


def build_spark(app_name: str = "features_job", with_kafka: bool = True) -> SparkSession:
    builder = (SparkSession.builder.appName(app_name).master("local[*]")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
               .config("spark.sql.shuffle.partitions", "4")
               # the console progress bar would print over the per-batch lines
               .config("spark.ui.showConsoleProgress", "false"))
    if with_kafka:
        builder = builder.config("spark.jars.packages", KAFKA_PKG)
    return builder.getOrCreate()


def read_telemetry(spark: SparkSession) -> DataFrame:
    """Kafka telemetry.smd parsed with the frozen schema, event_time as timestamp, 10-minute watermark."""
    df = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", S.KAFKA_BOOTSTRAP)
          .option("subscribe", S.TOPIC_SMD).option("startingOffsets", "earliest").load())
    return (df.select(F.from_json(F.col("value").cast("string"), S.TELEMETRY_JSON_DDL).alias("r"))
            .select("r.*")
            .withColumn("event_time", F.to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ss.SSSX"))
            .withWatermark("event_time", WATERMARK))


def build_features(df: DataFrame) -> DataFrame:
    """Window features for batch or streaming telemetry DataFrames (needs entity_id, event_time, metrics).

    t = minutes since window start. Tumbling windows start at multiples of 5 minutes since the epoch,
    so t = (epoch seconds mod 300) / 60, which can be computed before grouping.
    """
    seconds = F.col("event_time").cast("double")
    exploded = (df.select("entity_id", "event_time", F.explode("metrics").alias("metric", "value"))
                .filter(F.col("value").isNotNull())
                .withColumn("t", F.pmod(seconds, F.lit(float(WINDOW_SECONDS))) / 60.0))
    agg = (exploded.groupBy("entity_id", F.window("event_time", WINDOW).alias("w"), "metric")
           .agg(F.mean("value").alias("mean"), F.max("value").alias("max"),
                F.stddev_samp("value").alias("std"), F.count("value").alias("n"),
                F.sum("t").alias("st"), F.sum("value").alias("sx"),
                F.sum(F.col("t") * F.col("value")).alias("stx"),
                F.sum(F.col("t") * F.col("t")).alias("stt")))
    denom = F.col("n") * F.col("stt") - F.col("st") * F.col("st")
    slope = (F.when((F.col("n") < 2) | (denom == 0), F.lit(0.0))
             .otherwise((F.col("n") * F.col("stx") - F.col("st") * F.col("sx")) / denom))
    out = (agg.withColumn("slope", slope)
           .withColumn("window_start", F.col("w.start"))
           .withColumn("window_end", F.col("w.end")))
    return out.select(*S.LAYERS["features"].keys())


def _print_progress(progress: dict) -> None:
    state_rows = sum(op.get("numRowsTotal", 0) for op in progress.get("stateOperators", []))
    duration = progress.get("durationMs", {}).get("triggerExecution", 0)
    print(f"batch {progress['batchId']}: input rows {progress.get('numInputRows', 0)}, "
          f"duration {duration} ms, state rows {state_rows}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true",
                        help="trigger(availableNow=True): process what is in Kafka now, then stop")
    args = parser.parse_args(argv)

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")
    writer = (build_features(read_telemetry(spark)).writeStream.format("parquet").outputMode("append")
              .option("path", OUTPUT_DIR.as_posix())
              .option("checkpointLocation", CHECKPOINT_DIR.as_posix()))
    writer = writer.trigger(availableNow=True) if args.once else writer.trigger(processingTime=TRIGGER)
    query = writer.start()
    print(f"features_job started: {S.TOPIC_SMD} -> {OUTPUT_DIR.relative_to(ROOT).as_posix()}", flush=True)

    last_batch = -1
    try:
        while True:
            stopped = query.awaitTermination(5)
            # lastProgress only keeps the newest batch; recentProgress also catches batches that ran between polls.
            for progress in query.recentProgress:
                if progress["batchId"] > last_batch:
                    _print_progress(progress)
                    last_batch = progress["batchId"]
            if stopped:
                break
    except KeyboardInterrupt:
        print("stopping features_job", flush=True)
        query.stop()
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
