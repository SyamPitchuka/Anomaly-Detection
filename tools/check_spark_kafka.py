"""Inspection gauge for WP-2: PySpark 3.5.1 can read telemetry.smd with the frozen schema.

Run from the repo root after tools/check_kafka.py has passed:  python tools/check_spark_kafka.py
The first run downloads the Kafka connector from Maven (needs internet, ~1 minute).
"""
import os
import sys
from pathlib import Path

# On Windows, Spark must start workers with this same venv Python.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyspark  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from contracts import schemas as S  # noqa: E402

PKG = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
if pyspark.__version__ != "3.5.1":
    print(f"FAIL PySpark is {pyspark.__version__}, contract says 3.5.1")
    sys.exit(1)

spark = (SparkSession.builder.appName("check_spark_kafka").master("local[2]")
         .config("spark.jars.packages", PKG)
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = (spark.read.format("kafka").option("kafka.bootstrap.servers", S.KAFKA_BOOTSTRAP)
      .option("subscribe", S.TOPIC_SMD).option("startingOffsets", "earliest").load())
parsed = (df.select(F.from_json(F.col("value").cast("string"), S.TELEMETRY_JSON_DDL).alias("r"))
          .select("r.*")
          .withColumn("event_time", F.to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ss.SSSX")))
total = parsed.count()
bad = parsed.filter(F.col("entity_id").isNull() | F.col("event_time").isNull()).count()
machines = parsed.select("entity_id").distinct().count()
spark.stop()

print(f"{'OK  ' if total > 0 else 'FAIL'} read {total} records from {S.TOPIC_SMD}")
print(f"{'OK  ' if bad == 0 else 'FAIL'} {bad} records failed to parse")
print(f"OK   {machines} distinct machines")
ok = total > 0 and bad == 0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
