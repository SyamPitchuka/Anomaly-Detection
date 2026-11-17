"""experiments/e2_batch_model.py: score_job.log parsing, steady-batch selection and the T(n) = a + b/n fit on
synthetic logs (no cluster needed)."""
import json
import math

import pytest

from experiments import e2_batch_model as B

A_TRIGGER, B_TRIGGER = 3.2, 9.6     # s, s x n
A_STEP, B_STEP = 2.8, 9.4


def _log(n, batches=5, rows=200_000, first_s=30.0, read_only=False):
    """A score_job.log as score_job.py prints it, with T(n) = a + b/n per batch (first batch slow, last partial)."""
    lines = ["score_entry: Kafka spark-master:9092, checkpoint /x", "score_job started: telemetry.smd -> x",
             "batch 0: input rows 0, windows scored 0 (test 0, anomalies 0), duration 900 ms, state rows 0",
             '  durationMs {"triggerExecution": 900}']
    for i in range(1, batches + 2):
        last = i == batches + 1
        trig = first_s if i == 1 else A_TRIGGER + B_TRIGGER / n
        step = first_s - 1 if i == 1 else A_STEP + B_STEP / n
        r = rows // 3 if last else rows
        done = f"rows counted {r}" if read_only else f"windows scored 10 (test 5, anomalies 0)"
        lines.append(f"batch {i}: input rows {r}, {done}, duration {round(trig * 1000)} ms, state rows 112")
        lines.append("  durationMs " + json.dumps({"addBatch": round((trig - 0.2) * 1000), "getBatch": 0,
                                                   "latestOffset": 10, "triggerExecution": round(trig * 1000)}))
        name = "count(read only)" if read_only else "count(stateful scoring)"
        steps = {name: round(step * 1000)} if read_only else {name: round(step * 1000), "toPandas(test)": 50}
        lines.append("  foreachBatch steps ms " + json.dumps(steps))
    lines.append("stopping score_job")
    return "\n".join(lines) + "\n"


def test_parse_log_and_steady_batches():
    batches = B.parse_log(_log(2))
    assert [b["batch"] for b in batches] == list(range(7))
    assert batches[1]["steps"]["count(stateful scoring)"] == 29000 and batches[0]["steps"] == {}
    steady = B.steady_batches(batches)
    assert [b["batch"] for b in steady] == [2, 3, 4, 5]        # no empty, no first, no partial last
    assert B.steady_batches([]) == []
    assert B.parse_log("garbage\n  durationMs {}\n") == []


def test_fit_recovers_a_and_b():
    ns = [1, 2, 3, 4, 5]
    f = B.fit_amdahl(ns, [A_STEP + B_STEP / n for n in ns])
    assert f["a"] == pytest.approx(A_STEP) and f["b"] == pytest.approx(B_STEP)
    assert f["serial_fraction"] == pytest.approx(A_STEP / (A_STEP + B_STEP)) and f["r2"] == pytest.approx(1.0)
    assert math.isnan(B.fit_amdahl([2, 2], [1.0, 2.0])["a"])   # one distinct n -> no fit


def test_executors_from_path(tmp_path):
    assert B.executors_from_path(tmp_path / "logs" / "e2_A_20261117T000000Z_e4c2_rep1" / "score_job.log") == 4
    assert B.executors_from_path(tmp_path / "score_job.log") is None


def _write_sweep(tmp_path, read_only=False):
    runs = []
    for n in (1, 2, 3, 4, 5):
        for rep in (1, 2):
            rid = f"e2_A_20261117T000000Z_e{n}c2_rep{rep}"
            d = tmp_path / "scaling" / "logs" / rid
            d.mkdir(parents=True)
            (d / "score_job.log").write_text(_log(n, read_only=read_only), encoding="utf-8")
            runs.append({"run_id": rid, "executors": n, "cores": 2, "status": "drained"})
    runs.append({"run_id": "missing_e9c2_rep1", "executors": 9, "cores": 2, "status": "failed"})
    sweep = tmp_path / "e2_cluster_A.json"
    sweep.write_text(json.dumps({"meta": {"tag": "A"}, "runs": runs}), encoding="utf-8")
    return sweep


def test_main_from_sweep_json_writes_table_and_fits(tmp_path, capsys):
    sweep = _write_sweep(tmp_path)
    out = tmp_path / "model.md"
    assert B.main(["--sweep-json", str(sweep), "--run-dir", str(tmp_path / "scaling"), "--out", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    assert "| executors | runs | batches |" in md and "step `count(stateful scoring)`" in md
    assert "| 3 | 2 | 8 | 200,000 |" in md                     # 2 reps x 4 steady batches
    printed = capsys.readouterr()
    assert "step.count(stateful scoring): T(n) = 2.80 s + 9.40 s / n" in printed.out
    assert "durationMs.triggerExecution: T(n) = 3.20 s + 9.60 s / n" in printed.out
    assert "missing_e9c2_rep1" in printed.err                  # a run without log is reported, not fatal
    summary = B.summarise(B.load_runs(B.runs_from_sweep(sweep, tmp_path / "scaling"), 0.9))
    f = summary["fits"]["step.count(stateful scoring)"]
    assert f["a"] == pytest.approx(A_STEP, abs=0.01) and f["b"] == pytest.approx(B_STEP, abs=0.01)
    assert summary["fits"]["durationMs.addBatch"]["a"] == pytest.approx(A_TRIGGER - 0.2, abs=0.01)


def test_main_from_log_paths_read_only_and_default_out(tmp_path, monkeypatch, capsys):
    _write_sweep(tmp_path, read_only=True)
    logs = sorted((tmp_path / "scaling" / "logs").glob("*/score_job.log"))
    monkeypatch.setattr(B, "OUT_DIR", tmp_path / "results")
    assert B.main([str(p) for p in logs]) == 0
    assert "step.count(read only): T(n) = 2.80 s" in capsys.readouterr().out
    assert (tmp_path / "results" / "e2_batch_model.md").exists()
    with pytest.raises(SystemExit):
        B.main([str(tmp_path / "score_job.log")])               # no run id in the path
    with pytest.raises(SystemExit):
        B.main([])
