"""experiments/e6_cap.py: run on the simulated 2-node cluster (--dry-run), parsers and report rendering."""
import json

import pytest

from experiments import e6_cap as E


def test_required_replicas_and_percentile():
    assert [E.required_replicas(cl, 2) for cl in ("ONE", "QUORUM", "ALL")] == [1, 2, 2]
    assert [E.required_replicas(cl, 3) for cl in ("ONE", "QUORUM", "ALL")] == [1, 2, 3]
    assert E.percentile([], 50) is None
    assert E.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert E.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_parse_nodetool_status_and_hints():
    status = """Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address     Load        Tokens  Owns (effective)  Host ID                               Rack
UN  172.16.0.6  104.3 KiB   16      100.0%            2c3d4e5f-0000-4000-8000-000000000001  rack1
DN  172.17.0.6  98.1 KiB    16      100.0%            2c3d4e5f-0000-4000-8000-000000000002  rack1
"""
    assert E.parse_nodetool_status(status) == {"172.16.0.6": "UN", "172.17.0.6": "DN"}
    assert E.parse_pending_hints("This node does not have any pending hints") == 0
    hints = """          Host ID                               Address     Rack   DC           Status  Total files  Newest  Oldest
2c3d4e5f-0000-4000-8000-000000000002  172.17.0.6  rack1  datacenter1  DOWN    1            1731  1731
"""
    assert E.parse_pending_hints(hints) == 1


def test_dry_run_writes_the_report_and_matches_cap(tmp_path, capsys):
    rc = E.main(["--dry-run", "--hosts", "spark-w2,spark-w5", "--rows", "20", "--out", str(tmp_path)])
    assert rc == 0
    result = json.loads((tmp_path / "e6_cap.json").read_text(encoding="utf-8"))
    assert result["dry_run"] is True and result["victim"] == "spark-w5" and result["rf"] == 2
    by = {(p["phase"], r["cl"]): r for p in result["phases"] for r in p["results"]}
    for cl in ("ONE", "QUORUM", "ALL"):
        assert by[("both_up", cl)]["write"]["success_rate"] == 1.0
        assert by[("both_up", cl)]["read"]["found"] == 20
    assert by[("one_down", "ONE")]["write"]["ok"] == 20 and by[("one_down", "ONE")]["read"]["found"] == 20
    for cl in ("QUORUM", "ALL"):
        assert by[("one_down", cl)]["write"]["ok"] == 0
        assert by[("one_down", cl)]["write"]["errors"] == {"Unavailable": 20}
        assert by[("one_down", cl)]["write"]["p50_ms"] is None
    rec = result["recovery"]
    assert rec["expected_rows"] == 4 * 20                      # 3 levels both up + ONE with a node down
    assert rec["pending_hints_before_restart"] == 1 and rec["hints_delivered"]
    assert rec["victim_count_after_hints"] == 80
    assert rec["replica_counts_after_repair"] == {"spark-w2": 80, "spark-w5": 80}
    assert rec["converged"] is True

    md = (tmp_path / "e6_cap.md").read_text(encoding="utf-8")
    assert md.startswith("# E6: CAP behaviour of the Cassandra serving store (DRY RUN, simulated)")
    assert "| one_down (spark-w2 up) | QUORUM | 2 | 0/20 (0.0%) | - | - |" in md
    assert "write Unavailable 20, read Unavailable 20" in md
    assert "**ONE** (needs 1 of 2 replicas) stayed available" in md and "(AP)" in md
    assert "**ALL** (needs 2 of 2 replicas) refused every op" in md and "(CP)" in md
    assert "QUORUM = floor(2/2)+1 = 2 replicas = ALL" in md
    assert "UNEXPECTED" not in md
    assert "converged: **yes**" in md


def test_interpretation_flags_a_deviation():
    sim = E.SimBackend(["a", "b"])
    result = E.run_experiment(sim, ["a", "b"], "b", 5, log=lambda *_: None)
    # pretend QUORUM had stayed available with a node down: the report must not call that 'as expected'
    q = next(r for p in result["phases"] if p["phase"] == "one_down" for r in p["results"] if r["cl"] == "QUORUM")
    q["write"].update(ok=5, success_rate=1.0)
    q["read"].update(ok=5, success_rate=1.0)
    lines = E.interpret(result)
    assert "UNEXPECTED" in next(line for line in lines if line.startswith("**QUORUM**"))
    assert "UNEXPECTED" not in next(line for line in lines if line.startswith("**ONE**"))


def test_without_isolated_counts_uses_the_all_count(tmp_path):
    rc = E.main(["--dry-run", "--hosts", "a,b", "--victim", "a", "--rows", "3", "--no-isolated-counts",
                 "--out", str(tmp_path)])
    assert rc == 0
    rec = json.loads((tmp_path / "e6_cap.json").read_text(encoding="utf-8"))["recovery"]
    assert "replica_counts_after_repair" not in rec and rec["count_all_after_repair"] == 12 and rec["converged"]
    assert rec["survivor"] == "b"


def test_bad_arguments_are_refused():
    for argv in (["--dry-run", "--hosts", "a"], ["--dry-run", "--hosts", "a,b", "--victim", "c"],
                 ["--dry-run", "--hosts", "a,b", "--rows", "0"]):
        with pytest.raises(SystemExit):
            E.main(argv)
