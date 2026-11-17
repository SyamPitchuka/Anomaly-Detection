"""experiments/e2_scaling_sweep.py: configs, commands, aggregation (speed-up, efficiency, min-max) and the
report / plot (no cluster needed)."""
import json

import pytest

from experiments import e2_scaling_sweep as E


def _run(e, c, rps, rep=1, status="drained", records=1_000_000, **kw):
    return {"run_id": f"s_e{e}c{c}_rep{rep}", "status": status, "executors": e, "cores": c, "records": records,
            "records_per_s": rps, "steady_records_per_s": rps * 1.1 if rps else None, "drain_s": records / rps if rps else None,
            "batches": 5, "batch_mean_s": 2.0, "batch_p95_s": 3.0, "first_batch_s": 6.0,
            "executors_seen": e, "replicate": 4, "topic_partitions": 32, "shuffle_partitions": 2 * e * c, **kw}


RUNS = [_run(1, 2, 1000, 1), _run(1, 2, 1200, 2), _run(1, 2, 1100, 3),
        _run(2, 2, 2000, 1), _run(2, 2, 2200, 2), _run(2, 2, None, 3, status="timeout"),
        _run(4, 2, 3300, 1), _run(4, 1, 3000, 1)]


def test_parse_config_and_errors():
    assert E.parse_config("4x2") == (4, 2) and E.parse_config("1X8") == (1, 8)
    for bad in ("4", "x2", "0x2", "2x0", "ax2"):
        with pytest.raises(ValueError):
            E.parse_config(bad)


def test_schedule_interleaves_repetitions():
    assert E.schedule([(1, 2), (2, 2)], 2) == [(1, (1, 2)), (1, (2, 2)), (2, (1, 2)), (2, (2, 2))]


def test_replicate_factor_for_weak_scaling():
    configs = [(1, 2), (2, 2), (4, 2)]
    assert [E.replicate_factor(c, configs) for c in configs] == [1, 2, 4]
    with pytest.raises(ValueError):
        E.replicate_factor((3, 1), [(1, 2), (3, 1)])


def test_drain_command():
    cmd = E.drain_command((2, 4), 500, 100, "rid", E.RUN_DIR, replicate=3, extra=["-p", "32"])
    assert cmd[:2] == ["bash", E.DRAIN.as_posix()]
    s = " ".join(cmd)
    assert "-e 2 -c 4 -n 500 -m 100 --run-id rid" in s and s.endswith("-r 3 -p 32")
    assert "-r" not in E.drain_command((1, 1), 1, 1, "x", E.RUN_DIR)
    with_parts = " ".join(E.drain_command((1, 2), 1, 1, "x", E.RUN_DIR, replicate=4, topic_partitions=32))
    assert with_parts.endswith("-r 4 -P 32")
    assert "-P" not in E.drain_command((1, 1), 1, 1, "x", E.RUN_DIR)


def test_run_replicate_strong_vs_weak():
    configs = [(1, 2), (2, 2), (4, 2)]
    assert [E.run_replicate(c, configs, False, 4) for c in configs] == [4, 4, 4]     # strong: --replicate
    assert [E.run_replicate(c, configs, True, 4) for c in configs] == [1, 2, 4]      # weak: cores / base cores
    assert E.run_replicate((1, 2), configs, False, None) is None


def test_aggregate_mean_minmax_speedup_efficiency():
    table = E.aggregate(RUNS)
    assert [(r["executors"], r["cores"]) for r in table] == [(1, 2), (2, 2), (4, 1), (4, 2)]   # ties: fewer executors first
    base, e2c2, e4c1, e4c2 = table
    assert base["records_per_s"] == pytest.approx(1100)
    assert (base["records_per_s_min"], base["records_per_s_max"]) == (1000, 1200)
    assert base["speedup"] == pytest.approx(1.0) and base["ideal"] == 1.0
    assert (e2c2["runs"], e2c2["ok"]) == (3, 2)                     # the timeout run is not averaged
    assert e2c2["records_per_s"] == pytest.approx(2100)
    assert e2c2["speedup"] == pytest.approx(2100 / 1100)
    assert e2c2["speedup_min"] == pytest.approx(2000 / 1100) and e2c2["speedup_max"] == pytest.approx(2200 / 1100)
    assert e2c2["ideal"] == 2 and e2c2["efficiency"] == pytest.approx(2100 / 1100 / 2)
    assert e4c2["ideal"] == 4 and e4c2["efficiency"] == pytest.approx(3300 / 1100 / 4)
    assert e4c1["total_cores"] == 4 and e4c1["executors_seen"] == (4, 4)
    assert base["batch_p95_s"] == 3.0


def test_aggregate_all_failed_config():
    table = E.aggregate([_run(1, 2, 1000), _run(2, 2, None, status="failed (exit 1, no result file)")])
    failed = table[1]
    assert failed["ok"] == 0 and failed["records_per_s"] != failed["records_per_s"]   # NaN
    assert E.markdown_table(table).splitlines()[3].count("–") >= 5


def test_markdown_tables():
    table = E.aggregate(RUNS)
    md = E.markdown_table(table)
    lines = md.splitlines()
    assert lines[0].startswith("| executors × cores | total cores | replicate | topic partitions | "
                               "shuffle partitions | runs drained |")
    assert "| 1 × 2 | 2 | 4 | 32 | 4 | 3/3 |" in lines[2] and "1,100 (1,000–1,200)" in lines[2]
    assert "| 2 × 2 | 4 | 4 | 32 | 8 | 2/3 |" in md
    assert "| 4 × 2 | 8 | 4 | 32 | 16 | 1/1 |" in md
    runs_md = E.runs_table(RUNS)
    assert runs_md.splitlines()[0].startswith("| run | status | replicate | topic partitions | shuffle partitions |")
    assert "s_e2c2_rep3 | timeout | 4 | 32 | 8 |" in runs_md and len(runs_md.splitlines()) == 2 + len(RUNS)


def test_settings_that_differ_or_are_missing():
    runs = [_run(1, 2, 1000, 1), _run(1, 2, 1000, 2, shuffle_partitions=16),
            _run(2, 2, 2000, 1, replicate=None, topic_partitions="", shuffle_partitions=None)]
    base, other = E.aggregate(runs)
    assert base["shuffle_partitions"] == "4/16" and base["replicate"] == "4" and base["topic_partitions"] == "32"
    assert (other["replicate"], other["topic_partitions"], other["shuffle_partitions"]) == ("–", "–", "–")
    assert "| r | ? | – | – | – |" in E.runs_table([{"run_id": "r", "executors": 1, "cores": 1}])


def test_write_outputs_md_png_json(tmp_path):
    meta = {"date": "2026-11-15", "configs": ["1x2", "2x2", "4x2", "4x1"], "repeats": 3, "records": 1_000_000,
            "max_offsets_per_trigger": 200_000, "weak": False, "cluster": "4 workers x 4 vCPU", "argv": "",
            "replicate": 4, "topic_partitions": 32}
    table = E.write_outputs(RUNS, meta, tmp_path)
    assert len(table) == 4
    md = (tmp_path / "e2_cluster.md").read_text(encoding="utf-8")
    assert "![Throughput, speed-up and efficiency vs total cores](e2_cluster.png)" in md
    assert "strong scaling" in md and "4 workers x 4 vCPU" in md and "## Every run" in md
    assert "`replicate = 4` copies of every machine (112 keys)" in md
    assert "recreated with **32 partitions**" in md and "2 × total cores" in md
    assert "| replicate | topic partitions | shuffle partitions |" in md
    png = (tmp_path / "e2_cluster.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000
    saved = json.loads((tmp_path / "e2_cluster.json").read_text(encoding="utf-8"))
    assert saved["meta"] == meta and len(saved["runs"]) == len(RUNS)

    weak_md = tmp_path / "weak"
    E.write_outputs(RUNS, {**meta, "weak": True}, weak_md)
    assert "weak scaling" in (weak_md / "e2_cluster.md").read_text(encoding="utf-8")


def test_from_json_redraw_and_dry_run(tmp_path, capsys):
    meta = {"date": "2026-11-15", "repeats": 1, "records": 10, "max_offsets_per_trigger": 5, "weak": False}
    src = tmp_path / "saved.json"
    src.write_text(json.dumps({"meta": meta, "runs": RUNS}), encoding="utf-8")
    assert E.main(["--from-json", str(src), "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "e2_cluster.png").exists()
    capsys.readouterr()

    assert E.main(["--dry-run", "--configs", "1x2", "2x2", "--repeats", "2", "--records", "100", "--weak"]) == 0
    cmds = capsys.readouterr().out.splitlines()
    assert len(cmds) == 4
    assert "-e 1 -c 2 -n 100 " in cmds[0] and "-r 1 -P 32" in cmds[0]
    assert "-e 2 -c 2 -n 200 " in cmds[1] and "-r 2 -P 32" in cmds[1] and "_rep1" in cmds[1]
    assert "_rep2" in cmds[3]

    # strong scaling (default): every run replays 4 copies, same backlog, 32 topic partitions
    assert E.main(["--dry-run", "--configs", "1x2", "4x2", "--repeats", "1", "--records", "100"]) == 0
    cmds = capsys.readouterr().out.splitlines()
    assert all("-n 100 " in c and c.endswith("-r 4 -P 32") for c in cmds) and len(cmds) == 2
    assert E.main(["--dry-run", "--configs", "1x2", "--repeats", "1", "--replicate", "1", "--topic-partitions", "8",
                   "--", "-p", "64"]) == 0
    assert capsys.readouterr().out.strip().endswith("-r 1 -P 8 -p 64")
    with pytest.raises(SystemExit):
        E.main(["--dry-run", "--replicate", "0"])


# ---- per-run timeout ----------------------------------------------------------------------------------

def test_run_drain_returns_exit_code_or_kills_on_timeout(tmp_path):
    import sys
    import time
    assert E.run_drain([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, timeout=30) == (3, False)
    t = time.monotonic()
    rc, timed_out = E.run_drain([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, timeout=1,
                                kill_grace=10)
    assert timed_out and rc != 0 and time.monotonic() - t < 15


@pytest.mark.skipif(__import__("os").name != "posix", reason="process groups are POSIX only")
def test_run_drain_timeout_kills_the_whole_process_group(tmp_path):
    # the run ignores SIGTERM and leaves a grandchild: both must be gone after the SIGKILL that follows the grace
    import os
    import sys
    pidfile = tmp_path / "child.pid"
    script = (f"import signal, subprocess, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
              f"open({str(pidfile)!r}, 'w').write(str(p.pid)); time.sleep(60)")
    rc, timed_out = E.run_drain([sys.executable, "-c", script], tmp_path, timeout=2, kill_grace=1)
    assert timed_out and rc == -9
    child = int(pidfile.read_text())
    for _ in range(50):                       # the orphaned grandchild is reaped by init shortly after SIGKILL
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        __import__("time").sleep(0.1)
    else:
        pytest.fail("grandchild survived the group kill")


def test_main_records_a_timed_out_run_as_failed_and_continues(tmp_path, monkeypatch):
    calls = []

    def fake_run_drain(cmd, cwd, timeout, kill_grace=180):
        rid = cmd[cmd.index("--run-id") + 1]
        calls.append((rid, timeout))
        if len(calls) == 1:                    # 1st run hangs: no result file
            return -15, True
        if len(calls) == 2:                    # 2nd run drained, then hung on stop: stays a success
            (tmp_path / "runs" / f"{rid}.json").write_text(json.dumps(_run(2, 2, 2000, run_id=rid)))
            return -15, True
        (tmp_path / "runs" / f"{rid}.json").write_text(json.dumps(_run(4, 2, 3500, run_id=rid)))
        return 0, False

    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(E, "run_drain", fake_run_drain)
    rc = E.main(["--configs", "1x2", "2x2", "4x2", "--repeats", "1", "--run-timeout", "5",
                 "--run-dir", str(tmp_path / "runs"), "--out-dir", str(tmp_path / "out")])
    assert rc == 1 and len(calls) == 3 and all(t == 5 for _, t in calls)
    runs = json.loads((tmp_path / "out" / "e2_cluster.json").read_text(encoding="utf-8"))["runs"]
    assert runs[0]["status"] == "failed (timeout after 5 s, no result file)"
    assert runs[1]["status"] == "drained" and runs[1]["stop_timed_out"] is True
    assert runs[2]["status"] == "drained" and "stop_timed_out" not in runs[2]
    with pytest.raises(SystemExit):
        E.main(["--run-timeout", "0"])


def test_load_run_timeout_with_undrained_result(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_run(1, 2, None, status="timeout")))
    r = E._load_run(p, "r", (1, 2), -15, timed_out=True, timeout=1800)
    assert r["status"] == "failed (timeout after 1800 s, timeout)"
    assert E._load_run(tmp_path / "none.json", "r", (1, 2), 2)["status"] == "failed (exit 2, no result file)"


# ---- bottleneck switches (--tag, --job-args, --checkpoint-dir, --allow-local-checkpoint) ------------------

def test_experiment_args_and_output_name():
    assert E.experiment_args() == [] and E.output_name(None) == "e2_cluster" and E.output_name("A") == "e2_cluster_A"
    assert E.experiment_args("/tmp/ckpt-{run_id}", True, "B", ["--state-store", "rocksdb"]) == [
        "-k", "/tmp/ckpt-{run_id}", "--allow-local-checkpoint", "--tag", "B", "--", "--state-store", "rocksdb"]


def test_dry_run_passes_the_experiment_switches(capsys):
    assert E.main(["--dry-run", "--configs", "1x2", "--repeats", "1", "--tag", "A", "--job-args=--read-only",
                   "--", "-p", "64"]) == 0
    cmd = capsys.readouterr().out.strip()
    assert "--run-id e2_A_" in cmd and cmd.endswith("-r 4 -P 32 -p 64 --tag A -- --read-only")
    assert E.main(["--dry-run", "--configs", "2x2", "--repeats", "1", "--tag", "B2",
                   "--job-args=--state-store rocksdb"]) == 0
    assert capsys.readouterr().out.strip().endswith("--tag B2 -- --state-store rocksdb")
    assert E.main(["--dry-run", "--configs", "2x2", "--repeats", "1", "--checkpoint-dir", "/x/ckpt-{run_id}",
                   "--allow-local-checkpoint"]) == 0
    assert capsys.readouterr().out.strip().endswith("-k '/x/ckpt-{run_id}' --allow-local-checkpoint")
    with pytest.raises(SystemExit):
        E.main(["--dry-run", "--tag", "a b"])


def test_tagged_outputs_and_experiment_line(tmp_path):
    meta = {"date": "2026-11-17", "repeats": 1, "records": 10, "max_offsets_per_trigger": 5, "weak": False,
            "tag": "A", "job_args": "--read-only", "checkpoint_dir": "/tmp/ckpt-{run_id}",
            "allow_local_checkpoint": True}
    E.write_outputs(RUNS, meta, tmp_path)
    md = (tmp_path / "e2_cluster_A.md").read_text(encoding="utf-8")
    assert "(e2_cluster_A.png)" in md and (tmp_path / "e2_cluster_A.json").exists()
    assert "**Experiment A:** score_job extra args `--read-only`; checkpoint root `/tmp/ckpt-{run_id}`" in md
    assert not (tmp_path / "e2_cluster.md").exists()
    assert E.experiment_line({}) == ""
    # --from-json keeps the saved tag
    src = tmp_path / "saved.json"
    src.write_text(json.dumps({"meta": meta, "runs": RUNS}), encoding="utf-8")
    assert E.main(["--from-json", str(src), "--out-dir", str(tmp_path / "re")]) == 0
    assert (tmp_path / "re" / "e2_cluster_A.md").exists()
