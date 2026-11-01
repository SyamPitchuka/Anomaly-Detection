"""Convert the Alibaba cluster trace v2018 machine_usage CSV to the raw layer (Spark).

Run from the repo root:
    python src/producer/alibaba_to_parquet.py --input path/to/machine_usage.csv
    python src/producer/alibaba_to_parquet.py --input path/to/machine_usage.csv --machines 20   # quick subset

Input: machine_usage.csv without a header, columns
    machine_id, time_stamp, cpu_util_percent, mem_util_percent, mem_gps, mkpi, net_in, net_out, disk_io_percent
(time_stamp = seconds since the start of the trace; plain or .gz CSV, or a folder of CSV parts).

Each row becomes one telemetry record of the raw layer (contracts/schemas.py LAYERS["raw"]):
    entity_id = machine_id, event_time = 2026-01-01T00:00:00Z + time_stamp seconds,
    metrics = {cpu, mem, mem_gps, mkpi, net_in, net_out, disk_io} (empty or NaN values are left out, never 0),
    attrs = {} (Alibaba has no train/test split), label = null.
Rows without machine_id or time_stamp are dropped. A row whose metrics are all missing is kept with an empty map.
Sentinel values (such as -1) are not special-cased: only empty and NaN count as missing.

--machines N keeps the first N machine_ids in sorted (string) order.
Output: data/raw/source=alibaba/ (overwritten), sorted by entity_id, event_time within each file. The written
Parquet schema is checked against arrow_schema('raw').
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# On Windows, Spark must start workers with this same venv Python (as in tools/check_spark_kafka.py).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from contracts import schemas as S  # noqa: E402

CSV_DDL = ("machine_id STRING, time_stamp LONG, cpu_util_percent DOUBLE, mem_util_percent DOUBLE, mem_gps DOUBLE, "
           "mkpi DOUBLE, net_in DOUBLE, net_out DOUBLE, disk_io_percent DOUBLE")
# CSV column -> contract metric key, in the contract's metric order.
METRIC_COLUMNS = {"cpu": "cpu_util_percent", "mem": "mem_util_percent", "mem_gps": "mem_gps", "mkpi": "mkpi",
                  "net_in": "net_in", "net_out": "net_out", "disk_io": "disk_io_percent"}
assert list(METRIC_COLUMNS) == S.ALIBABA_METRICS, "METRIC_COLUMNS must follow contracts.schemas.ALIBABA_METRICS"

BASE_EPOCH = int(datetime.fromisoformat(S.TIME_BASE_UTC.replace("Z", "+00:00")).timestamp())
PARTITIONS = 32
UTC_FORMAT = "yyyy-MM-dd'T'HH:mm:ss'Z'"


def build_spark(app_name: str = "alibaba_to_parquet", driver_memory: str = "4g") -> SparkSession:
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
            .config("spark.driver.memory", driver_memory)
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def read_usage(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.csv(path.as_posix(), schema=CSV_DDL, header=False)


def first_machines(usage: DataFrame, n: int) -> list[str]:
    rows = usage.where(F.col("machine_id").isNotNull()).select("machine_id").distinct().orderBy("machine_id").limit(n)
    return [r["machine_id"] for r in rows.collect()]


def to_raw(usage: DataFrame) -> DataFrame:
    """Telemetry records with exactly the columns and order of LAYERS["raw"]."""
    pairs = [x for key, col in METRIC_COLUMNS.items() for x in (F.lit(key), F.col(col))]
    metrics = F.map_filter(F.create_map(*pairs), lambda _, v: v.isNotNull() & ~F.isnan(v))
    return (usage.where(F.col("machine_id").isNotNull() & F.col("time_stamp").isNotNull())
            .select(F.col("machine_id").alias("entity_id"),
                    F.timestamp_seconds(F.lit(BASE_EPOCH) + F.col("time_stamp")).alias("event_time"),
                    metrics.alias("metrics"),
                    F.create_map().cast("map<string,string>").alias("attrs"),
                    F.lit(None).cast("int").alias("label")))


def check_schema(out_dir: Path) -> None:
    """The written Parquet must match arrow_schema('raw'): same columns in the same order, tolerant types."""
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files written under {out_dir}")
    got = pq.read_schema(files[0])
    want = S.LAYERS["raw"]
    if got.names != list(want):
        raise ValueError(f"raw columns are {got.names}, contract says {list(want)}")
    for name, code in want.items():
        if not S.type_matches(code, got.field(name).type):
            raise ValueError(f"raw column '{name}' is {got.field(name).type}, contract says {code}")


def convert(input_path: Path, root: Path = ROOT, machines: int | None = None, spark: SparkSession | None = None,
            partitions: int = PARTITIONS) -> dict:
    spark = spark or build_spark()
    usage = read_usage(spark, input_path)
    if machines:
        keep = first_machines(usage, machines)
        if not keep:
            raise ValueError(f"no machine_id found in {input_path}")
        usage = usage.where(F.col("machine_id").isin(keep))
        print(f"subset: first {len(keep)} machines ({keep[0]} .. {keep[-1]})")

    out_dir = root / S.PATHS["raw"] / "source=alibaba"
    (to_raw(usage).repartition(partitions, "entity_id").sortWithinPartitions("entity_id", "event_time")
     .write.mode("overwrite").parquet(out_dir.as_posix()))
    check_schema(out_dir)

    row = (spark.read.parquet(out_dir.as_posix())
           .agg(F.count("*").alias("rows"), F.countDistinct("entity_id").alias("machines"),
                F.date_format(F.min("event_time"), UTC_FORMAT).alias("first"),   # formatted in Spark (UTC session)
                F.date_format(F.max("event_time"), UTC_FORMAT).alias("last"),
                F.sum((F.size("metrics") == 0).cast("int")).alias("empty_metrics")).first())
    summary = row.asDict()
    print(f"wrote {summary['rows']} rows, {summary['machines']} machines to {out_dir.relative_to(root).as_posix()} "
          f"({summary['first']} .. {summary['last']}, {summary['empty_metrics']} rows with no metric values)")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="machine_usage.csv (or .gz, or a folder of parts)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root that contains data/")
    parser.add_argument("--machines", type=int, default=None, help="keep only the first N machine_ids (sorted)")
    parser.add_argument("--partitions", type=int, default=PARTITIONS, help="output files (and sort partitions)")
    parser.add_argument("--driver-memory", default="4g", help="Spark driver memory, e.g. 2g on a small laptop")
    args = parser.parse_args()
    if args.machines is not None and args.machines < 1:
        parser.error("--machines must be at least 1")
    spark = build_spark(driver_memory=args.driver_memory)
    spark.sparkContext.setLogLevel("ERROR")
    try:
        convert(args.input.expanduser(), args.root, args.machines, spark, args.partitions)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
