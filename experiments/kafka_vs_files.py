"""Do score_job --input files and --input kafka give the same windows for the same records?

Run from the repo root, with Kafka up:
    python experiments/kafka_vs_files.py subset            # first N replay records -> data/diag/subset_root
    python experiments/kafka_vs_files.py files             # score the subset from Parquet files
    python experiments/kafka_vs_files.py kafka --tag before   # replay the subset into Kafka, score it
    python experiments/kafka_vs_files.py compare files kafka_before

subset: the first --limit (20,000) records in replay order (event_time, entity_id) over every file of
data/raw/source=smd, exactly what `replay.py --limit 20000` sends, written back per input file into
data/diag/subset_root/data/raw/source=smd/<same file name>. So fake.parquet rows stay in fake.parquet, which
the files input skips (autoencoder.SKIP_FILES) and replay.py did not skip before the fix.

files / kafka: the score_job query (build_scores, same models, same per-machine state logic) with
trigger(availableNow), local[2]; every emitted window, train and test, is saved with entity_id, ws_us,
split, label, ae_score, ae_flag, is_anomaly -> data/diag/windows_<mode>.parquet. kafka recreates
telemetry.smd and runs `replay.py --root data/diag/subset_root --speedup 0` first.
compare: windows matched on (entity_id, ws_us); counts, is_anomaly rates, max |ae_score diff|, and the
windows present on one side only.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import schemas as S  # noqa: E402

DIAG = ROOT / "data" / "diag"
SUBSET_ROOT = DIAG / "subset_root"
SUBSET_RAW = SUBSET_ROOT / S.PATHS["raw"] / "source=smd"
COLS = ["entity_id", "ws_us", "split", "label", "ae_score", "ae_flag", "is_anomaly"]


def make_subset(limit: int) -> None:
    raw = ROOT / S.PATHS["raw"] / "source=smd"
    tables = []
    for f in sorted(raw.glob("*.parquet")):
        t = pq.read_table(f, columns=["entity_id", "event_time", "metrics", "attrs", "label"])
        tables.append(t.append_column("file", pa.array([f.name] * t.num_rows)))
    t = pa.concat_tables(tables)
    idx = pc.sort_indices(t, sort_keys=[("event_time", "ascending"), ("entity_id", "ascending")]).slice(0, limit)
    sub = t.take(idx)
    shutil.rmtree(SUBSET_ROOT, ignore_errors=True)
    SUBSET_RAW.mkdir(parents=True)
    files = sub["file"].to_pylist()
    for name in sorted(set(files)):
        part = sub.filter(pc.equal(sub["file"], name)).drop(["file"])
        pq.write_table(part.cast(S.arrow_schema("raw")), SUBSET_RAW / name)
        print(f"  {name}: {part.num_rows} rows")
    print(f"subset: {sub.num_rows} records in {len(set(files))} files -> {SUBSET_RAW.relative_to(ROOT).as_posix()}")


def score(mode: str, tag: str) -> Path:
    from pyspark.sql import functions as F

    from src.streaming import score_job as SJ
    from src.streaming.features_job import build_spark, read_telemetry

    name = f"windows_{mode}" + (f"_{tag}" if tag else "")
    ckpt = ROOT / "checkpoints" / "diag" / name
    shutil.rmtree(ckpt, ignore_errors=True)
    if mode == "kafka":
        from experiments.e2e_latency_runs import recreate_topic
        recreate_topic(S.TOPIC_SMD)
        r = subprocess.run([sys.executable, "src/producer/replay.py", "--source", "smd", "--speedup", "0",
                            "--root", str(SUBSET_ROOT)], cwd=ROOT, capture_output=True, text=True)
        print("  replay:", (r.stdout.strip().splitlines() or ["?"])[-1], r.stderr.strip()[-300:])
    spark = build_spark(f"diag_{name}", with_kafka=mode == "kafka", master="local[2]", shuffle_partitions=2)
    spark.sparkContext.setLogLevel("ERROR")
    if mode == "kafka":
        telemetry = read_telemetry(spark)
    else:
        from src.detection import autoencoder as AE
        files = sorted(p.name for p in SUBSET_RAW.glob("*.parquet") if p.name not in AE.SKIP_FILES)
        telemetry = (spark.readStream.schema(SJ.RAW_DDL).option("maxFilesPerTrigger", 1)
                     .option("pathGlobFilter", "{" + ",".join(files) + "}")
                     .parquet(SUBSET_RAW.as_posix()).withWatermark("event_time", SJ.WATERMARK))
    parts = []

    def keep(batch_df, batch_id):
        pdf = batch_df.select(*COLS).toPandas()
        if len(pdf):
            parts.append(pdf.assign(batch_id=batch_id))

    q = (SJ.build_scores(telemetry).writeStream.foreachBatch(keep)
         .option("checkpointLocation", ckpt.as_posix()).trigger(availableNow=True).start())
    q.awaitTermination()
    spark.stop()
    out = DIAG / f"{name}.parquet"
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS + ["batch_id"])
    df.to_parquet(out, index=False)
    print(f"{name}: {len(df)} windows ({int((df['split'] == 'test').sum())} test), "
          f"is_anomaly {int(df['is_anomaly'].sum())} -> {out.relative_to(ROOT).as_posix()}")
    return out


def _rate(df: pd.DataFrame) -> str:
    n = len(df)
    k = int(df["is_anomaly"].astype(bool).sum())
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "0/0"


def compare(a_name: str, b_name: str) -> str:
    a = pd.read_parquet(DIAG / f"windows_{a_name}.parquet")
    b = pd.read_parquet(DIAG / f"windows_{b_name}.parquet")
    lines = [f"| | {a_name} | {b_name} |", "| --- | ---: | ---: |",
             f"| windows | {len(a):,} | {len(b):,} |",
             f"| test windows | {int((a['split'] == 'test').sum())} | {int((b['split'] == 'test').sum())} |",
             f"| is_anomaly, all windows | {_rate(a)} | {_rate(b)} |",
             f"| is_anomaly, train windows | {_rate(a[a['split'] == 'train'])} | {_rate(b[b['split'] == 'train'])} |",
             f"| is_anomaly, test windows | {_rate(a[a['split'] == 'test'])} | {_rate(b[b['split'] == 'test'])} |"]
    m = a.merge(b, on=["entity_id", "ws_us"], suffixes=("_a", "_b"), how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both["ae_score_a"] - both["ae_score_b"]).abs()
    split_diff = both[both["split_a"] != both["split_b"]]
    flag_diff = both[both["is_anomaly_a"].astype(bool) != both["is_anomaly_b"].astype(bool)]
    lines += [f"| matched windows | {len(both):,} | |",
              f"| only in {a_name} / only in {b_name} | {int((m['_merge'] == 'left_only').sum())} "
              f"| {int((m['_merge'] == 'right_only').sum())} |",
              f"| max abs ae_score diff (matched) | {np.nanmax(diff.to_numpy()) if len(diff) else float('nan'):.3g} | |",
              f"| windows with ae_score diff > 1e-6 | {int((diff > 1e-6).sum())} | |",
              f"| windows whose split differs | {len(split_diff)} | |",
              f"| windows whose is_anomaly differs | {len(flag_diff)} | |"]
    bad = both[(diff > 1e-6) | (both["split_a"] != both["split_b"])]
    if len(bad):
        g = bad.groupby("entity_id").agg(windows=("ws_us", "size"), first=("ws_us", "min"), last=("ws_us", "max"))
        t0 = pd.Timestamp("2026-01-01", tz="UTC")
        for c in ("first", "last"):
            g[c] = ((pd.to_datetime(g[c], unit="us", utc=True) - t0).dt.total_seconds() // 60).astype(int)
        lines += ["", "Windows that differ, per machine (window start in minutes since 2026-01-01 00:00):", "",
                  "| machine | windows | first | last |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {e} | {r['windows']} | {r['first']} | {r['last']} |" for e, r in g.iterrows()]
    text = "\n".join(lines)
    print(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("subset", "files", "kafka", "compare"))
    ap.add_argument("names", nargs="*", help="compare: two run names, e.g. files kafka_before")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    if args.cmd == "subset":
        make_subset(args.limit)
    elif args.cmd in ("files", "kafka"):
        score(args.cmd, args.tag)
    else:
        compare(*args.names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
