"""Frozen data contracts for Team SPARK (revision B).

This file is the master copy of Part A of the shared document.
Do not edit without Syam's approval. Every module imports names from here.
"""
import pyarrow as pa

SOURCES = ("smd", "alibaba")

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC_SMD = "telemetry.smd"
TOPIC_ALIBABA = "telemetry.alibaba"
TOPIC_ANOMALIES = "anomalies"

SMD_METRICS = [f"m{i:02d}" for i in range(38)]
ALIBABA_METRICS = ["cpu", "mem", "mem_gps", "mkpi", "net_in", "net_out", "disk_io"]

# SMD rows get synthetic times: base + i minutes (train rows first, then test rows).
TIME_BASE_UTC = "2026-01-01T00:00:00.000Z"

DATA_ROOT = "data"
PATHS = {
    "raw": "data/raw",
    "features": "data/features",
    "scores": "data/scores",
    "incidents": "data/incidents",
    "smd_interpretation": "data/reference/smd_interpretation",
}

# Detector ladder (revision B). Models are trained offline and saved here (git-ignored).
MODELS = {
    "kmeans": "models/kmeans_<source>",   # Tier 2, Spark MLlib KMeans
    "pca": "models/pca_<source>",         # Tier 2b, Spark MLlib PCA reconstruction error
    "ae": "models/ae_<source>",           # Tier 3, PyTorch dense autoencoder (model.pt + scaler.json)
}
# Window is anomalous if the autoencoder OR the robust z-score flags it.
# top_metrics = the 3 metrics with the largest autoencoder reconstruction error.
ANOMALY_RULE = "is_anomaly = ae_flag OR tier1_flag"

# Type codes used by the checks: ts = any timestamp, map_sd = map<string,double>,
# map_ss = map<string,string>, list_s = list<string>.
LAYERS = {
    "raw": {
        "entity_id": "string", "event_time": "ts", "metrics": "map_sd",
        "attrs": "map_ss", "label": "int",
    },
    "features": {
        "entity_id": "string", "window_start": "ts", "window_end": "ts",
        "metric": "string", "mean": "double", "max": "double", "std": "double",
        "slope": "double", "n": "long",
    },
    "scores": {
        "entity_id": "string", "window_start": "ts", "window_end": "ts",
        "tier1_score": "double", "tier1_flag": "bool", "tier2_score": "double",
        "tier2_flag": "bool", "pca_score": "double", "ae_score": "double",
        "ae_flag": "bool", "is_anomaly": "bool", "top_metrics": "list_s",
        "label": "int",
    },
    "incidents": {
        "incident_id": "string", "entity_id": "string", "start_time": "ts",
        "end_time": "ts", "n_windows": "long", "max_score": "double",
        "top_metrics": "list_s", "created_at": "ts",
    },
    "smd_interpretation": {
        "entity_id": "string", "start_idx": "long", "end_idx": "long",
        "metrics": "list_s",
    },
}

# Same contracts as Spark DDL strings, for spark.read.schema(...) and from_json(...).
TELEMETRY_JSON_DDL = (
    "source STRING, entity_id STRING, event_time STRING, "
    "metrics MAP<STRING,DOUBLE>, attrs MAP<STRING,STRING>, label INT"
)
SCORES_DDL = (
    "entity_id STRING, window_start TIMESTAMP, window_end TIMESTAMP, "
    "tier1_score DOUBLE, tier1_flag BOOLEAN, tier2_score DOUBLE, tier2_flag BOOLEAN, "
    "pca_score DOUBLE, ae_score DOUBLE, ae_flag BOOLEAN, is_anomaly BOOLEAN, top_metrics ARRAY<STRING>, label INT"
)

# Exact pyarrow types used when Python code (fake data, tests) writes Parquet.
_PA = {
    "string": pa.string(), "ts": pa.timestamp("us", tz="UTC"), "double": pa.float64(),
    "long": pa.int64(), "int": pa.int32(), "bool": pa.bool_(),
    "map_sd": pa.map_(pa.string(), pa.float64()),
    "map_ss": pa.map_(pa.string(), pa.string()), "list_s": pa.list_(pa.string()),
}


def arrow_schema(layer: str) -> pa.Schema:
    return pa.schema([(name, _PA[code]) for name, code in LAYERS[layer].items()])


def type_matches(code: str, t: pa.DataType) -> bool:
    """Tolerant type check: Spark and pyarrow may differ in timestamp unit or int width."""
    if code == "string":
        return pa.types.is_string(t) or pa.types.is_large_string(t)
    if code == "ts":
        return pa.types.is_timestamp(t)
    if code == "double":
        return pa.types.is_float64(t)
    if code == "long":
        return pa.types.is_int64(t)
    if code == "int":
        return pa.types.is_int32(t) or pa.types.is_int64(t)
    if code == "bool":
        return pa.types.is_boolean(t)
    if code == "map_sd":
        return pa.types.is_map(t) and pa.types.is_float64(t.item_type)
    if code == "map_ss":
        return pa.types.is_map(t) and pa.types.is_string(t.item_type)
    if code == "list_s":
        return (pa.types.is_list(t) or pa.types.is_large_list(t)) and pa.types.is_string(t.value_type)
    return False
