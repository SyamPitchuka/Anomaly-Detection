"""experiments/e2_scaling_sweep.py: configs, commands, aggregation (speed-up, efficiency, min-max) and the
report / plot (no cluster needed)."""
import json

import pytest

from experiments import e2_scaling_sweep as E


def _run(e, c, rps, rep=1, status="drained", records=1_000_000, **kw):
    return {"run_id": f"s_e{e}c{c}_rep{rep}", "status": status, "executors": e, "cores": c, "records": records,
            "records_per_s": rps, "steady_records_per_s": rps * 1.1 if rps else None, "drain_s": records / rps if rps else None,
            "batches": 5, "batch_mean_s": 2.0, "batch_p95_s": 3.0, "first_batch_s": 6.0,
            "executors_seen": e, **kw}


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
    assert lines[0].startswith("| executors × cores | total cores | runs drained |")
    assert "| 1 × 2 | 2 | 3/3 |" in lines[2] and "1,100 (1,000–1,200)" in lines[2]
    assert "| 2 × 2 | 4 | 2/3 |" in md
    runs_md = E.runs_table(RUNS)
    assert "s_e2c2_rep3 | timeout" in runs_md and len(runs_md.splitlines()) == 2 + len(RUNS)


def test_write_outputs_md_png_json(tmp_path):
    meta = {"date": "2026-11-15", "configs": ["1x2", "2x2", "4x2", "4x1"], "repeats": 3, "records": 1_000_000,
            "max_offsets_per_trigger": 200_000, "weak": False, "cluster": "4 workers x 4 vCPU", "argv": ""}
    table = E.write_outputs(RUNS, meta, tmp_path)
    assert len(table) == 4
    md = (tmp_path / "e2_cluster.md").read_text(encoding="utf-8")
    assert "![Throughput, speed-up and efficiency vs total cores](e2_cluster.png)" in md
    assert "strong scaling" in md and "4 workers x 4 vCPU" in md and "## Every run" in md
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
    assert "-e 1 -c 2 -n 100 " in cmds[0] and "-r 1" in cmds[0]
    assert "-e 2 -c 2 -n 200 " in cmds[1] and "-r 2" in cmds[1] and "_rep1" in cmds[1]
    assert "_rep2" in cmds[3]
