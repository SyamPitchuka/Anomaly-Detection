"""Pure functions of infra/cluster/soak_sampler.py (no Kafka needed)."""
import json
import sys
from pathlib import Path

CLUSTER = Path(__file__).resolve().parents[1] / "infra" / "cluster"
if str(CLUSTER) not in sys.path:
    sys.path.insert(0, str(CLUSTER))

import soak_sampler as M  # noqa: E402


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rec(ts, rows):
    return json.dumps({"logged_ts_ms": ts, "numInputRows": rows})


def test_progress_since_counts_batches_in_the_half_open_interval(tmp_path):
    p = tmp_path / "progress.jsonl"
    _write(p, [_rec(1000, 10), _rec(2000, 20), _rec(3000, 30), _rec(4000, 40)])
    assert M.progress_since(p, 1000, 3000) == (2, 50)      # (1000, 3000]: 1000 excluded, 3000 included
    assert M.progress_since(p, 0, 10_000) == (4, 100)
    assert M.progress_since(p, 4000, 5000) == (0, 0)


def test_progress_since_missing_file_and_bad_lines(tmp_path):
    assert M.progress_since(tmp_path / "nope.jsonl", 0, 10**12) == (0, 0)
    p = tmp_path / "progress.jsonl"
    _write(p, [_rec(1000, 5), '{"logged_ts_ms": 2000, "numInp', _rec(3000, 7)])   # a line still being written
    assert M.progress_since(p, 0, 10_000) == (2, 12)


def test_progress_since_tolerates_missing_fields(tmp_path):
    p = tmp_path / "progress.jsonl"
    _write(p, [json.dumps({"logged_ts_ms": 1500}), json.dumps({"numInputRows": 9})])
    assert M.progress_since(p, 0, 10_000) == (1, 0)         # no numInputRows -> 0 rows; no timestamp -> not counted


def _row(produced, job):
    return {"produced_rec_per_s": produced, "job_rows_per_s": job}


def test_summarise_excludes_first_minute_from_steady_figures():
    lines = M.summarise([_row(100, 10), _row(1000, 200), _row(3000, 400)])
    assert len(lines) == 2
    assert "produced" in lines[0] and "min 1,000  mean 2,000  max 3,000" in lines[0] and "2 steady minutes" in lines[0]
    assert "score_job" in lines[1] and "min 200  mean 300  max 400" in lines[1]


def test_summarise_single_minute_uses_that_minute_and_empty_gives_nothing():
    lines = M.summarise([_row(500, 0)])
    assert "min 500  mean 500  max 500" in lines[0] and "1 steady minutes" in lines[0]
    assert "min 0  mean 0  max 0" in lines[1]
    assert M.summarise([]) == []
