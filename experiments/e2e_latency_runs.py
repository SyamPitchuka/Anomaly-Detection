"""End-to-end latency runs: replay.py -> Kafka -> score_job --input kafka, one run per replay speed.

Run from the repo root, with Kafka up (python tools/check_spark_kafka.py):
    python experiments/e2e_latency_runs.py                         # speedups 100, 500, 0 (defaults below)
    python experiments/e2e_latency_runs.py --runs 100:56000         # speedup:limit pairs
    python experiments/e2e_latency_runs.py --master "local[4]"      # score_job master (default local[4])
    python experiments/e2e_latency_runs.py --runs 100:56000 --trigger "5 seconds" --tag t5
then  python src/streaming/latency_report.py

Per run (speedup S, limit N):
  1. delete and recreate telemetry.smd (partitions from contracts/topics.yaml), so score_job, which reads from
     the earliest offset, sees only this run's records
  2. start score_job --input kafka --reset --max-seconds CAP --idle-stop IDLE (OMP_NUM_THREADS=1) and wait for
     its "score_job started" line
  3. start replay.py --source smd --speedup S --limit N
  4. every 5 s: broker end offsets, committed offsets in the checkpoint, free RAM -> lag.csv; if free RAM
     drops below --min-free-gb (1) both processes are killed and the run is marked out of memory
  5. score_job stops by itself (idle or cap); the latency side output, the logs and meta.json are kept in
     data/latency_runs/<speedup_S[_tag]>/
"""
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from contracts import schemas as S  # noqa: E402
from src.streaming import latency_report as LR  # noqa: E402
from src.streaming.score_job import CHECKPOINT_DIR, LATENCY_DIR, PROGRESS_LOG  # noqa: E402

RUNS_DIR = ROOT / "data" / "latency_runs"
# event minutes in ~20 min of wall time at 28 records per event minute; speedup 0 is bounded by scoring
DEFAULT_RUNS = ["100:56000", "500:280000", "0:400000"]
CAP_S = 25 * 60
SAMPLE_S = 5
MB = 2 ** 20
HARDWARE = "Laptop: i5-13420H (12 threads), 16 GB RAM, Windows 11, Kafka 3.7.0 in Docker Desktop."


def recreate_topic(topic: str) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic
    cfg = yaml.safe_load((ROOT / "contracts" / "topics.yaml").read_text())["topics"][topic]
    admin = AdminClient({"bootstrap.servers": S.KAFKA_BOOTSTRAP})
    if topic in admin.list_topics(timeout=10).topics:
        admin.delete_topics([topic])[topic].result()
        for _ in range(60):
            if topic not in admin.list_topics(timeout=10).topics:
                break
            time.sleep(1)
    for attempt in range(30):
        try:
            admin.create_topics([NewTopic(topic, num_partitions=cfg["partitions_laptop"],
                                          replication_factor=cfg["replication_laptop"])])[topic].result()
            break
        except Exception as e:  # noqa: BLE001  (topic still being deleted)
            if attempt == 29:
                raise
            print(f"  waiting to recreate {topic}: {e}", flush=True)
            time.sleep(2)
    parts = 0
    for _ in range(30):   # metadata can lag the create call
        meta = admin.list_topics(timeout=10).topics
        if topic in meta and len(meta[topic].partitions) == cfg["partitions_laptop"]:
            parts = len(meta[topic].partitions)
            break
        time.sleep(1)
    if not parts:
        raise RuntimeError(f"{topic} not visible after recreate")
    print(f"  {topic} recreated with {parts} partitions", flush=True)


def end_offsets(consumer, topic: str, partitions: int) -> dict[int, int]:
    from confluent_kafka import TopicPartition
    return {p: consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=5)[1] for p in range(partitions)}


def _kill_tree(proc) -> None:
    import psutil
    try:
        root = psutil.Process(proc.pid)
        for p in root.children(recursive=True) + [root]:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def run_once(speedup: float, limit: int, trigger: str, tag: str, order: int, min_free_mb: float,
             idle_s: float, master: str = "local[4]", shuffle_partitions: int = 4,
             replay_root: Path | None = None) -> dict:
    import psutil
    from confluent_kafka import Consumer

    name = f"speedup_{speedup:g}" + (f"_{tag}" if tag else "")
    out = RUNS_DIR / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    meta = {"order": order, "speedup": speedup, "limit": limit, "trigger": trigger, "master": master,
            "shuffle_partitions": shuffle_partitions, "status": "ok",
            "hardware": HARDWARE, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": []}
    print(f"run {name}: speedup {speedup:g}, limit {limit:,}, trigger {trigger}", flush=True)
    free = psutil.virtual_memory().available / MB
    if free < min_free_mb:
        meta.update(status="skipped: free RAM below limit before start", free_mb=free)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    recreate_topic(S.TOPIC_SMD)
    partitions = yaml.safe_load((ROOT / "contracts" / "topics.yaml").read_text())["topics"][S.TOPIC_SMD][
        "partitions_laptop"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONUNBUFFERED": "1"}
    py = sys.executable
    job_log = (out / "score_job.log").open("w", encoding="utf-8")
    job = subprocess.Popen([py, "src/streaming/score_job.py", "--input", "kafka", "--reset", "--trigger", trigger,
                            "--max-seconds", str(CAP_S), "--idle-stop", str(idle_s),
                            "--master", master, "--shuffle-partitions", str(shuffle_partitions)],
                           cwd=ROOT, stdout=job_log, stderr=subprocess.STDOUT, env=env)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 300:
        if "score_job started" in (out / "score_job.log").read_text(encoding="utf-8", errors="replace"):
            break
        if job.poll() is not None:
            break
        time.sleep(1)
    else:
        meta["notes"].append("score_job did not report started within 300 s")
    if job.poll() is not None:
        meta["status"] = f"failed: score_job exited early with code {job.returncode}"
        job_log.close()
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta
    time.sleep(5)   # first (empty) trigger

    rep_log = (out / "replay.log").open("w", encoding="utf-8")
    replay_cmd = [py, "src/producer/replay.py", "--source", "smd", "--speedup", str(speedup), "--limit", str(limit)]
    replay_cmd += ["--root", str(replay_root)] if replay_root else []
    replay = subprocess.Popen(replay_cmd, cwd=ROOT, stdout=rep_log, stderr=subprocess.STDOUT, env=env)
    start = time.monotonic()
    consumer = Consumer({"bootstrap.servers": S.KAFKA_BOOTSTRAP, "group.id": "e2e-latency-lag-probe",
                         "enable.auto.commit": False})
    min_free = float("inf")
    with (out / "lag.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "high", "committed", "lag", "free_mb"])
        while True:
            t = time.monotonic() - start
            try:
                high = end_offsets(consumer, S.TOPIC_SMD, partitions)
                committed = LR.committed_offsets(CHECKPOINT_DIR, S.TOPIC_SMD)
                w.writerow([round(t, 1), sum(high.values()), sum(committed.values()), LR.lag(high, committed),
                            round(psutil.virtual_memory().available / MB)])
                f.flush()
            except Exception as e:  # noqa: BLE001
                meta["notes"].append(f"lag sample failed at {t:.0f} s: {e}")
            free = psutil.virtual_memory().available / MB
            min_free = min(min_free, free)
            if free < min_free_mb:
                meta["status"] = f"out of memory: free RAM {free:.0f} MB at {t:.0f} s, run killed"
                _kill_tree(replay)
                _kill_tree(job)
                break
            if job.poll() is not None:
                break
            if t > CAP_S + 180:
                meta["notes"].append("score_job did not stop within cap + 180 s, killed")
                _kill_tree(job)
                break
            time.sleep(SAMPLE_S)
    consumer.close()
    if replay.poll() is None:
        meta["notes"].append("replay still running when score_job stopped (cap reached), killed")
        _kill_tree(replay)
    replay.wait()
    job.wait()
    job_log.close()
    rep_log.close()
    meta.update(wall_s=round(time.monotonic() - start, 1), min_free_mb=round(min_free),
                score_job_exit=job.returncode, replay_exit=replay.returncode)
    rlog = (out / "replay.log").read_text(encoding="utf-8", errors="replace")
    done = [ln for ln in rlog.splitlines() if ln.startswith("done:")]
    meta["records_sent"] = int(done[-1].split()[1].replace(",", "")) if done else 0
    meta["replay_done"] = done[-1] if done else None
    if LATENCY_DIR.exists():
        shutil.copytree(LATENCY_DIR, out / "latency")
    if PROGRESS_LOG.exists():
        shutil.copy2(PROGRESS_LOG, out / "progress.jsonl")
    if "stopping score_job: --max-seconds" in (out / "score_job.log").read_text(encoding="utf-8", errors="replace"):
        meta["notes"].append("score_job stopped at the 25-minute cap")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    s = LR.load_run(out)["summary"]
    print(f"  {meta['status']}: {meta['records_sent']:,} records, {s['windows']:,} windows, total p95 "
          f"{s['total_p95_s']:.1f} s, processing p95 {s['proc_p95_s']:.1f} s, wall {meta['wall_s']:.0f} s",
          flush=True)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="speedup:limit pairs")
    ap.add_argument("--trigger", default="30 seconds")
    ap.add_argument("--tag", default="", help="suffix of the run folder names")
    ap.add_argument("--order-start", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    ap.add_argument("--idle-stop", type=float, default=120.0)
    ap.add_argument("--master", default="local[4]",
                    help="score_job Spark master (default local[4]: local[*] = 12 Python workers, each with its own "
                         "PyTorch + models, ran the 16 GB laptop below 1 GB free)")
    ap.add_argument("--shuffle-partitions", type=int, default=4)
    ap.add_argument("--replay-root", type=Path, default=None,
                    help="replay.py --root (a folder with data/raw/source=smd); replay loads every file there "
                         "into memory, so a small subset saves RAM")
    args = ap.parse_args(argv)
    print(platform.platform(), flush=True)
    for i, spec in enumerate(args.runs):
        sp, lim = spec.split(":")
        meta = run_once(float(sp), int(lim), args.trigger, args.tag, args.order_start + i,
                        args.min_free_gb * 1024, args.idle_stop, args.master, args.shuffle_partitions,
                        args.replay_root)
        if meta["status"].startswith("out of memory"):
            print("stopping: out of memory", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
