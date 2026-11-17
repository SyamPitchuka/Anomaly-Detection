"""infra/cluster/drain_monitor.py parsing / summary, and the option handling of drain_test.sh and submit.sh -m
(no Kafka, Spark or VMs needed)."""
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "infra" / "cluster"
if str(CLUSTER) not in sys.path:
    sys.path.insert(0, str(CLUSTER))

import drain_monitor as D  # noqa: E402

TOPIC = "telemetry.smd"


def _batch(bid, ts, rows, ms, offsets, as_string=False):
    end = {TOPIC: {str(p): o for p, o in enumerate(offsets)}}
    return {"batchId": bid, "timestamp": ts, "numInputRows": rows, "durationMs": {"triggerExecution": ms},
            "sources": [{"description": "KafkaV2[Subscribe[telemetry.smd]]",
                         "endOffset": json.dumps(end) if as_string else end}]}


# 3 data batches of 100 rows (2 partitions), then an empty one: 10:00:00.000 + 4 s, +4.5 s, +5 s
BATCHES = [_batch(0, "2026-11-15T10:00:00.000Z", 100, 4000, [50, 50]),
           _batch(1, "2026-11-15T10:00:04.000Z", 100, 2000, [100, 100]),
           _batch(2, "2026-11-15T10:00:06.000Z", 100, 3000, [150, 150], as_string=True),
           _batch(3, "2026-11-15T10:00:09.000Z", 0, 100, [150, 150])]


def test_parse_iso_ms():
    assert D.parse_iso_ms("1970-01-01T00:00:01.500Z") == 1500
    assert D.parse_iso_ms("2026-11-15T10:00:04.000Z") - D.parse_iso_ms("2026-11-15T10:00:00.000Z") == 4000


def test_topic_offset_object_string_and_other_topic():
    assert D.topic_offset(BATCHES[0], TOPIC) == 100
    assert D.topic_offset(BATCHES[2], TOPIC) == 300                     # endOffset given as a JSON string
    assert D.topic_offset(BATCHES[0], "telemetry.alibaba") is None
    assert D.topic_offset({"sources": [{"endOffset": "not json"}]}, TOPIC) is None
    assert D.topic_offset({}, TOPIC) is None
    assert D.consumed_offset(BATCHES, TOPIC) == 300
    assert D.consumed_offset([], TOPIC) == 0


def test_read_progress_filters_dedupes_and_skips_bad_lines(tmp_path):
    p = tmp_path / "progress.jsonl"
    lines = [json.dumps({"logged_ts_ms": 500, **BATCHES[0]}),            # before the job started: ignored
             json.dumps({"logged_ts_ms": 1000, **BATCHES[1]}),
             json.dumps({"logged_ts_ms": 1100, **BATCHES[0]}),
             json.dumps({"logged_ts_ms": 1200, **{**BATCHES[1], "numInputRows": 99}}),   # newest line per batch wins
             '{"logged_ts_ms": 1300, "batch']                                            # still being written
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    got = D.read_progress(p, since_ms=1000)
    assert [b["batchId"] for b in got] == [0, 1]
    assert got[1]["numInputRows"] == 99
    assert D.read_progress(tmp_path / "missing.jsonl") == []


def test_percentile_nearest_rank():
    assert D.percentile([], 95) != D.percentile([], 95)                 # NaN
    assert D.percentile([3.0], 95) == 3.0
    assert D.percentile(list(range(1, 21)), 95) == 19                   # ceil(0.95 * 20) = 19th value
    assert D.percentile([5, 1, 4, 2, 3], 50) == 3


def test_summarise_drain():
    s = D.summarise(BATCHES, 300, wall_s=42.0)
    assert s["batches"] == 3 and s["input_rows"] == 300                 # the empty batch is not counted
    assert s["drain_s"] == pytest.approx(9.0)                           # 10:00:00 -> 10:00:06 + 3 s
    assert s["records_per_s"] == pytest.approx(300 / 9)
    assert s["steady_records_per_s"] == pytest.approx(200 / 5)          # without batch 0 (4 s)
    assert s["batch_mean_s"] == pytest.approx(3.0)
    assert s["batch_p95_s"] == pytest.approx(4.0)
    assert s["first_batch_s"] == pytest.approx(4.0)
    assert s["wall_s"] == 42.0
    empty = D.summarise([BATCHES[3]], 0)
    assert empty["batches"] == 0 and empty["drain_s"] != empty["drain_s"]


def test_executor_tracker():
    t = D.ExecutorTracker()
    assert t.result() == {"executors_seen": None, "executors_max_active": None, "executor_cores_max": None}
    t.add([{"id": "driver", "totalCores": 0}, {"id": "0", "totalCores": 2, "isActive": True}])
    t.add([{"id": "driver"}, {"id": "0", "totalCores": 2}, {"id": "1", "totalCores": 2}])
    t.add([{"id": "0", "totalCores": 2, "isActive": False}, {"id": "1", "totalCores": 2}, {"id": "2", "totalCores": 2}])
    assert t.result() == {"executors_seen": 3, "executors_max_active": 2, "executor_cores_max": 4}
    assert D.parse_executors([{"id": "driver", "totalCores": 4}]) == (set(), 0, 0)


def test_parse_meta():
    assert D.parse_meta(["executors=2", "cores=4", "replicate=", "trigger=0_seconds"]) == \
        {"executors": 2, "cores": 4, "replicate": None, "trigger": "0_seconds"}
    with pytest.raises(ValueError):
        D.parse_meta(["executors"])


def test_build_and_write_result(tmp_path):
    t = D.ExecutorTracker()
    t.add([{"id": "0", "totalCores": 2}, {"id": "1", "totalCores": 2}])
    meta = D.parse_meta(["executors=2", "cores=2", "max_offsets_per_trigger=100", "replicate=",
                         "topic_partitions=32", "shuffle_partitions=8"])
    r1 = D.build_result("NaN_run", "drained", meta, 300, BATCHES, TOPIC, t, 30.0)
    assert r1["total_cores"] == 4 and r1["consumed"] == 300 and r1["executors_seen"] == 2
    assert [b["offset"] for b in r1["batches_detail"]] == [100, 200, 300, 300]
    r2 = D.build_result("run2", "timeout", meta, 300, [], TOPIC, D.ExecutorTracker(), None)
    js, cs = D.write_result(r1, tmp_path)
    D.write_result(r2, tmp_path)

    saved = json.loads(js.read_text(encoding="utf-8"))
    assert saved["run_id"] == "NaN_run"                                  # NaN inside a string is untouched
    assert saved["records_per_s"] == pytest.approx(300 / 9)
    assert json.loads((tmp_path / "run2.json").read_text(encoding="utf-8"))["drain_s"] is None

    with cs.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_id"] for r in rows] == ["NaN_run", "run2"]            # header written once
    assert list(rows[0]) == D.CSV_FIELDS
    assert rows[0]["drain_s"] == "9.0" and rows[0]["batches"] == "3" and rows[0]["replicate"] == ""
    assert rows[0]["topic_partitions"] == "32" and rows[0]["shuffle_partitions"] == "8"
    assert saved["topic_partitions"] == 32 and saved["shuffle_partitions"] == 8
    assert rows[1]["status"] == "timeout" and rows[1]["drain_s"] == "" and rows[1]["executors_seen"] == ""


def test_partition_count():
    from types import SimpleNamespace as NS
    meta = NS(topics={TOPIC: NS(partitions={i: object() for i in range(32)}), "empty": NS(partitions={})})
    assert D.partition_count(meta, TOPIC) == 32
    assert D.partition_count(meta, "empty") == 0 and D.partition_count(meta, "missing") == 0


# ---- shell scripts (bash only; lib.sh reads cluster.env.example) -----------------------------------------

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _bash(args):
    # settings go in through 'export' on stdin, as in test_cluster_tools.py: on Windows 'bash' may be WSL's,
    # which does not inherit the environment of this process
    return subprocess.run(["bash"], input=f"export CLUSTER_ENV=cluster.env.example; bash {shlex.join(args)}",
                          cwd=CLUSTER, capture_output=True, text=True, timeout=60, env=os.environ)


@needs_bash
def test_drain_test_help_and_bad_options():
    r = _bash(["drain_test.sh", "--help"])
    assert r.returncode == 0 and "backlog" in r.stdout and "--max-offsets-per-trigger" in r.stdout
    for bad, msg in ((["-n", "0"], "RECORDS"), (["-m", "abc"], "MAX_OFFSETS"), (["-e", "x"], "EXECUTORS"),
                     (["-r", "0"], "--replicate"), (["-P", "0"], "TOPIC_PARTITIONS"),
                     (["-p", "x"], "SHUFFLE"), (["--bogus"], "unknown option")):
        r = _bash(["drain_test.sh", *bad])
        assert r.returncode != 0 and msg in r.stderr, (bad, r.stderr)


def _config(*args):
    r = _bash(["drain_test.sh", "--print-config", *args])
    assert r.returncode == 0, r.stderr
    return dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)


@needs_bash
def test_drain_test_defaults_shuffle_to_twice_the_cores_and_32_topic_partitions():
    cfg = _config("-e", "4", "-c", "2")
    assert cfg["shuffle_partitions"] == "16" and cfg["topic_partitions"] == "32" and cfg["replicate"] == ""
    cfg = _config("-e", "3", "-c", "2", "-r", "4")
    assert cfg["shuffle_partitions"] == "12" and cfg["replicate"] == "4"
    cfg = _config("-e", "1", "-c", "2", "-p", "40", "-P", "16")
    assert cfg["shuffle_partitions"] == "40" and cfg["topic_partitions"] == "16"
    cfg = _config()                                    # cluster.env.example: 2 workers x 1 executor x 2 cores
    assert (cfg["executors"], cfg["cores"], cfg["shuffle_partitions"]) == ("2", "2", "8")


@needs_bash
def test_submit_dry_run_passes_max_offsets_per_trigger_only_when_given():
    r = _bash(["submit.sh", "--dry-run", "-e", "1", "-c", "2", "-m", "200000", "--", "--reset"])
    assert r.returncode == 0, r.stderr
    cmd = " ".join(r.stdout.splitlines()[-1].split())
    assert "--max-offsets-per-trigger 200000 --reset" in cmd and cmd.endswith("--reset")
    r = _bash(["submit.sh", "--dry-run", "-e", "1", "-c", "2"])
    assert r.returncode == 0 and "--max-offsets-per-trigger" not in r.stdout.splitlines()[-1]
    r = _bash(["submit.sh", "--dry-run", "-m", "0"])
    assert r.returncode != 0 and "--max-offsets-per-trigger" in r.stderr


def test_write_result_upgrades_an_older_csv_header(tmp_path):
    old_fields = D.CSV_FIELDS[:D.CSV_FIELDS.index("tag")]
    with (tmp_path / D.CSV_NAME).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old_fields)
        w.writeheader()
        w.writerow({"run_id": "old", "status": "drained", "executors": 2})
    meta = D.parse_meta(["executors=2", "cores=2", "job_args=--read-only", "checkpoint_dir=/tmp/ckpt-r",
                         "allow_local_checkpoint=1", "tag=A"])
    D.write_result(D.build_result("new", "drained", meta, 300, BATCHES, TOPIC, D.ExecutorTracker(), 30.0), tmp_path)
    with (tmp_path / D.CSV_NAME).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == D.CSV_FIELDS and [r["run_id"] for r in rows] == ["old", "new"]
    assert rows[0]["executors"] == "2" and rows[0]["job_args"] == ""
    assert (rows[1]["job_args"], rows[1]["checkpoint_dir"], rows[1]["allow_local_checkpoint"], rows[1]["tag"]) == (
        "--read-only", "/tmp/ckpt-r", "1", "A")


@needs_bash
def test_drain_test_passes_the_experiment_switches():
    cfg = _config("--run-id", "r7", "-k", "/tmp/ckpt-{run_id}", "--allow-local-checkpoint", "--tag", "A",
                  "--", "--read-only", "--state-store", "rocksdb")
    assert cfg["checkpoint_dir"] == "/tmp/ckpt-r7" and cfg["allow_local_checkpoint"] == "1" and cfg["tag"] == "A"
    assert cfg["job_args"] == "--read-only --state-store rocksdb"
    cfg = _config()
    assert (cfg["checkpoint_dir"], cfg["allow_local_checkpoint"], cfg["tag"], cfg["job_args"]) == ("", "0", "", "")
    r = _bash(["drain_test.sh", "--tag", "a/b"])
    assert r.returncode != 0 and "--tag" in r.stderr


@needs_bash
def test_submit_refuses_a_local_checkpoint_unless_allowed():
    r = _bash(["submit.sh", "--dry-run", "-e", "1", "-c", "2", "-k", "/tmp/ckpt-x"])
    assert r.returncode != 0 and "--allow-local-checkpoint" in r.stderr
    r = _bash(["submit.sh", "--dry-run", "-e", "1", "-c", "2", "-k", "/tmp/ckpt-x", "--allow-local-checkpoint",
               "--", "--read-only"])
    assert r.returncode == 0, r.stderr
    assert "recovery after a" in r.stderr and "NOT guaranteed" in r.stderr
    assert "/tmp/ckpt-x/score/source=smd" in r.stdout and r.stdout.splitlines()[-1].rstrip().endswith("--read-only")
