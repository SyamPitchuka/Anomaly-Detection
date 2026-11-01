"""experiments/e2_local_scaling.py: log parsing, medians, speed-up/efficiency, out-of-memory runs and outputs
(no Spark needed)."""
import json

import pytest

from experiments import e2_local_scaling as E

LOG = """reset: removed stream output and checkpoint
score_job started: 2 machine files (machine-1-1, machine-1-2) -> data/scores_stream/source=smd
batch 0: input rows 600, windows scored 120 (test 60, anomalies 3), duration 3000 ms, state rows 1
some warning line
batch 1: input rows 400, windows scored 80 (test 40, anomalies 0), duration 2000 ms, state rows 2
"""


def _run(rps: float, wall: float = 10.0, rss: float = 2048, free: float = 8192) -> dict:
    return {"status": "ok", "wall_s": wall, "batch_s": 1000 / rps, "records": 1000, "rps": rps,
            "peak_state": 2, "peak_rss_mb": rss, "free_before_mb": free, "min_free_mb": free - rss}


def _oom(rss: float = 9000) -> dict:
    return {"status": E.OOM, "wall_s": 50.0, "batches_done": 1, "peak_rss_mb": rss, "free_before_mb": 8192,
            "min_free_mb": 900}


def test_parse_and_summarise():
    batches = E.parse_batches(LOG)
    assert [b["batch"] for b in batches] == [0, 1]
    assert batches[1] == {"batch": 1, "rows": 400, "ms": 2000, "state": 2}
    s = E.summarise_run(batches, 12.5)
    assert s == {"wall_s": 12.5, "batch_s": 5.0, "records": 1000, "batches": 2, "rps": 200.0, "peak_state": 2}


def test_aggregate_medians_speedup_efficiency():
    runs = {2: [_run(150), _run(170)], 1: [_run(90), _run(110)], 4: [_run(200, rss=3072), _run(200, rss=4096)]}
    table = E.aggregate(runs)
    assert [r["cores"] for r in table] == [1, 2, 4]
    assert [r["shuffle_partitions"] for r in table] == [2, 4, 8]
    assert [r["rps"] for r in table] == [100, 160, 200]
    assert [r["speedup"] for r in table] == pytest.approx([1.0, 1.6, 2.0])
    assert [r["efficiency"] for r in table] == pytest.approx([1.0, 0.8, 0.5])
    assert table[2]["peak_rss_mb"] == 3584
    md = E.markdown_table(table)
    assert md.count("\n") == 4 and "| 4 | 8 |" in md and "| 3.50 | 8.0 | 2/2 |" in md
    assert "peak memory" not in E.markdown_table(table, memory=False)


def test_out_of_memory_runs():
    table = E.aggregate({1: [_run(100), _run(100)], 4: [_run(300), _oom()], 8: [_oom(), _oom(9500)]})
    assert table[1]["status"] == "ok" and table[1]["rps"] == 300 and table[1]["ok"] == 1
    assert table[2]["status"] == E.OOM and table[2]["peak_rss_all_mb"] == 9500
    md = E.markdown_table(table)
    assert "| 1/2 (1 out of memory) |" in md
    assert "| 8 | 16 | – | – | out of memory | – | – | – | 9.28 | 8.0 | 0/2 (2 out of memory) |" in md


def test_outputs_from_json(tmp_path):
    meta = {"machines": 2, "records": 1000, "repeats": 2, "files_per_trigger": 8, "batches": 1, "cpu_count": 12,
            "ram_gb": 15.6, "min_free_gb": 1.0, "host": "test", "date": "2026-11-01"}
    saved = {"meta": meta, "runs": {"1": [_run(100), _run(100)], "2": [_run(180), _run(180)], "8": [_oom()]}}
    src = tmp_path / "in.json"
    src.write_text(json.dumps(saved))
    out = tmp_path / "out"
    assert E.main(["--from-json", str(src), "--out-dir", str(out), "--extra-json"]) == 0
    md = (out / "e2_local.md").read_text(encoding="utf-8")
    assert "cluster version of E2 replaces it" in md and "| 2 | 4 |" in md and "## Main result" not in md
    assert "| 8 | 16 | – | – | out of memory |" in md
    assert (out / "e2_local.png").stat().st_size > 1000

    base = {"meta": {**meta, "repeats": 1, "files_per_trigger": 1, "title": "Before: flat",
                     "explanation": "One key per batch."},
            "runs": {"1": [{"wall_s": 10.0, "batch_s": 10.0, "records": 1000, "rps": 100, "peak_state": 2}],
                     "2": [{"wall_s": 10.0, "batch_s": 10.5, "records": 1000, "rps": 95, "peak_state": 2}]}}
    (tmp_path / "base.json").write_text(json.dumps(base))
    notes = tmp_path / "notes.md"
    notes.write_text("Scaling flattens because of reasons.")
    assert E.main(["--from-json", str(src), "--out-dir", str(out), "--extra-json", str(tmp_path / "base.json"),
                   str(tmp_path / "missing.json"), "--notes", str(notes)]) == 0
    md = (out / "e2_local.md").read_text(encoding="utf-8")
    assert md.index("## Before: flat") < md.index("One key per batch.") < md.index("## Main result")
    assert "| 2 | 4 | 10.0 | 10.5 | 95 | 2 | 0.95 | 0.47 |\n" in md   # no memory columns in the old table
    assert md.rstrip().endswith("Scaling flattens because of reasons.")
