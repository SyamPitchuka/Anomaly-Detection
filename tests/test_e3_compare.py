"""experiments/e3_compare.py: loss / duplicate counting on tiny fixtures (no Kafka, no Spark)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments import e3_compare as E


def _write(dirpath, name, rows):
    dirpath.mkdir(parents=True, exist_ok=True)
    t = pa.table({"entity_id": [r[0] for r in rows],
                  "window_start": pa.array([r[1] for r in rows], type=pa.timestamp("us", tz="UTC")),
                  "ae_score": [r[2] for r in rows], "is_anomaly": [r[3] for r in rows]})
    pq.write_table(t, dirpath / name)


def _ts(minute):
    import datetime as dt
    return dt.datetime(2026, 1, 1, 0, minute, tzinfo=dt.timezone.utc)


def _topic(path, rows):
    path.write_text("\n".join(json.dumps({"entity_id": e, "window_start": f"2026-01-01T00:{m:02d}:00.000Z",
                                          "ae_score": s, "is_anomaly": True}) for e, m, s in rows) + "\n")


def test_identical_runs_pass(tmp_path):
    rows = [("m1", _ts(0), 0.1, False), ("m1", _ts(5), 0.9, True)]
    _write(tmp_path / "c", "batch-000000.parquet", rows)
    _write(tmp_path / "f", "batch-000000.parquet", rows)
    _topic(tmp_path / "c.jsonl", [("m1", 5, 0.9)])
    _topic(tmp_path / "f.jsonl", [("m1", 5, 0.9)])
    p = E.compare(E.load_scores(tmp_path / "c"), E.load_scores(tmp_path / "f"))
    t = E.compare(E.topic_rows(tmp_path / "c.jsonl"), E.topic_rows(tmp_path / "f.jsonl"))
    assert (p["lost"], p["extra"], p["fault_duplicates"], p["mismatch"]) == (0, 0, 0, 0)
    assert E.verdict(p, t)


def test_lost_and_duplicate_rows_are_counted(tmp_path):
    _write(tmp_path / "c", "batch-000000.parquet", [("m1", _ts(0), 0.1, False), ("m1", _ts(5), 0.9, True)])
    _write(tmp_path / "f", "batch-000000.parquet", [("m1", _ts(0), 0.1, False)])
    _write(tmp_path / "f", "batch-000001.parquet", [("m1", _ts(0), 0.1, False)])   # same window again
    p = E.compare(E.load_scores(tmp_path / "c"), E.load_scores(tmp_path / "f"))
    assert p["lost"] == 1 and p["fault_duplicates"] == 1 and p["extra"] == 0


def test_topic_duplicates_are_reported_but_not_lost(tmp_path):
    _topic(tmp_path / "c.jsonl", [("m1", 5, 0.9)])
    _topic(tmp_path / "f.jsonl", [("m1", 5, 0.9), ("m1", 5, 0.9)])
    t = E.compare(E.topic_rows(tmp_path / "c.jsonl"), E.topic_rows(tmp_path / "f.jsonl"))
    assert t["fault_duplicates"] == 1 and t["lost"] == 0 and t["extra"] == 0


ROOT = Path(__file__).resolve().parents[1]


def _run_script(*args):
    env = {**os.environ, "CLUSTER_ENV": str(ROOT / "infra/cluster/cluster.env.example")}
    return subprocess.run([shutil.which("bash"), "experiments/e3_faults.sh", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_e3_script_help_and_argument_validation():
    r = _run_script("--help")
    assert r.returncode == 0 and "Steps" in r.stdout
    r = _run_script("--records", "abc")
    assert r.returncode != 0 and "positive integer" in r.stderr
    r = _run_script("--bogus")
    assert r.returncode != 0 and "unknown option" in r.stderr


def test_committed_offsets_use_newest_commit(tmp_path):
    for d in ("offsets", "commits"):
        (tmp_path / d).mkdir()
    for n in (0, 1):
        (tmp_path / "offsets" / str(n)).write_text(f"v1\n{{}}\n{json.dumps({'t': {'0': 10 * (n + 1)}})}\n")
    (tmp_path / "commits" / "0").write_text("{}")          # batch 1 planned, not committed
    assert E.committed_end_offsets(tmp_path) == {"t": {"0": 10}}
