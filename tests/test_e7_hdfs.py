"""E7 (HDFS checkpoints): experiments/e7_report.py parsing / compare / report with fake files, and the hdfs:// URI
handling of submit.sh, drain_test.sh, e2_scaling_sweep.py, lib.sh, 50_hdfs.sh, deploy.sh and health_check.sh.
No Hadoop, Spark, Kafka or VMs needed."""
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments import e2_scaling_sweep as SW
from experiments import e7_report as R

CLUSTER = Path(__file__).resolve().parents[1] / "infra" / "cluster"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

FSCK = """Connecting to namenode via http://spark-master:9870/fsck?ugi=hdfs&path=%2Fx
FSCK started by hdfs (auth:SIMPLE) from /10.0.0.4 for path /x at Wed Nov 18 10:00:00 UTC 2026

Status: {status}
 Number of data-nodes:\t{dn}
 Number of racks:\t\t1
 Total dirs:\t\t\t40
 Total symlinks:\t\t0

Replicated Blocks:
 Total size:\t1234567 B
 Total files:\t300
 Total blocks (validated):\t{blocks} (avg. block size 4115 B)
 Minimally replicated blocks:\t{blocks} (100.0 %)
 Over-replicated blocks:\t0 (0.0 %)
 Under-replicated blocks:\t{under} (10.0 %)
 Mis-replicated blocks:\t\t0 (0.0 %)
 Default replication factor:\t2
 Average block replication:\t1.9
 Missing blocks:\t\t{missing}
 Corrupt blocks:\t\t0
 Missing replicas:\t\t{under} (5.0 %)

Erasure Coded Block Groups:
 Total size:\t0 B
 Under-replicated block groups:\t7 (0.0 %)
 Missing block groups:\t\t0
FSCK ended at Wed Nov 18 10:00:01 UTC 2026 in 20 milliseconds


The filesystem under path '/x' is {status}
"""
REPORT = "Configured Capacity: 1 (1 B)\n-------------------------------------------------\nLive datanodes ({live}):\n\nName: 10.0.0.5:9866\n"


def sample_text(ts, phase, live=3, under=0, missing=0, blocks=300, status="HEALTHY"):
    return (f"# e7-sample ts_ms={ts} phase={phase}\n" + REPORT.format(live=live) + "==== fsck\n"
            + FSCK.format(status=status, dn=live, blocks=blocks, under=under, missing=missing))


# ---- parsing ----------------------------------------------------------------------------------------------
def test_parse_sample_takes_the_replicated_block_counts_not_the_erasure_coded_ones():
    s = R.parse_sample(sample_text(1000, "outage", live=2, under=30, blocks=300))
    assert (s["ts_ms"], s["phase"], s["live"]) == (1000, "outage", 2)
    assert (s["status"], s["total_blocks"], s["under_replicated"], s["missing"], s["corrupt"]) == ("HEALTHY", 300, 30, 0, 0)
    assert R.parse_sample("# e7-sample ts_ms=5 phase=before\n==== fsck\n")["under_replicated"] is None
    assert R.parse_fsck(FSCK.format(status="CORRUPT", dn=3, blocks=9, under=0, missing=2))["missing"] == 2


def test_sample_cli_field(tmp_path, capsys):
    f = tmp_path / "1_outage.txt"
    f.write_text(sample_text(1, "outage", live=2, under=4), encoding="utf-8")
    for field, want in (("live", "2"), ("under", "4"), ("missing", "0"), ("status", "HEALTHY")):
        assert R.main(["sample", str(f), "--field", field]) == 0
        assert capsys.readouterr().out.strip() == want


def test_event_and_batches(tmp_path, capsys):
    ev = tmp_path / "events.jsonl"
    R.append_event(ev, "datanode_stopped", ["host=spark-w5", "after_batches=3"], now_ms=42)
    assert R.read_events(ev) == [{"name": "datanode_stopped", "ts_ms": 42, "host": "spark-w5", "after_batches": 3}]
    with pytest.raises(ValueError):
        R.append_event(ev, "bad", ["novalue"])
    prog = tmp_path / "progress.jsonl"
    prog.write_text("\n".join(json.dumps({"logged_ts_ms": 10, "batchId": i, "numInputRows": n})
                              for i, n in enumerate([200000, 200000, 0, 5])) + "\n{half", encoding="utf-8")
    assert R.batches_with_rows(prog) == 3
    assert R.main(["batches", "--progress", str(tmp_path / "missing.jsonl")]) == 0
    assert capsys.readouterr().out.strip() == "0"


# ---- compare + report ---------------------------------------------------------------------------------------
def write_run(d: Path, rows, topic_rows):
    (d / "scores").mkdir(parents=True)
    t = pa.table({"entity_id": [r[0] for r in rows],
                  "window_start": pa.array([datetime(2026, 1, 1, 0, r[1], tzinfo=timezone.utc) for r in rows],
                                           pa.timestamp("us", tz="UTC")),
                  "ae_score": [r[2] for r in rows], "is_anomaly": [r[3] for r in rows]})
    pq.write_table(t, d / "scores" / "batch-0.parquet")
    (d / "anomalies.jsonl").write_text("".join(
        json.dumps({"entity_id": e, "window_start": f"2026-01-01T00:{m:02d}:00.000Z", "ae_score": s, "is_anomaly": True})
        + "\n" for e, m, s in topic_rows), encoding="utf-8")


ROWS = [("m1", 0, 0.1, False), ("m1", 5, 9.0, True), ("m2", 0, 0.2, False)]
TOPIC = [("m1", 5, 9.0)]


def test_compare_identical_and_different_runs(tmp_path):
    write_run(tmp_path / "clean", ROWS, TOPIC)
    write_run(tmp_path / "fault", ROWS, TOPIC + TOPIC)            # a re-published anomaly is not a failure
    res = R.compare_runs(tmp_path / "clean", tmp_path / "fault")
    assert res["pass"] and res["clean_anomalies"] == res["fault_anomalies"] == 1
    assert res["topic"]["fault_duplicates"] == 1 and res["parquet"]["fault_rows"] == 3
    write_run(tmp_path / "bad", ROWS[:2] + [("m2", 0, 0.5, False)], TOPIC)
    assert R.main(["compare", "--clean", str(tmp_path / "clean"), "--fault", str(tmp_path / "bad"),
                   "--out", str(tmp_path / "c.json")]) == 1
    assert json.loads((tmp_path / "c.json").read_text())["parquet"]["mismatch"] == 1


def run_json(rid, rps=20000.0, status="drained", records=1416825):
    return {"run_id": rid, "status": status, "records": records, "consumed": records if status == "drained" else 5,
            "batches": 8, "drain_s": records / rps, "records_per_s": rps, "steady_records_per_s": rps * 1.3,
            "batch_mean_s": 9.0, "batch_p95_s": 26.0, "first_batch_s": 26.0, "executors": 5, "cores": 2,
            "checkpoint_dir": f"hdfs://spark-master:9000/spark-anomaly/checkpoints/e7/{rid}"}


def fake_out(tmp_path, under_during_outage=30, healed=True):
    out = tmp_path / "e7_x"
    (out / "runs").mkdir(parents=True)
    for rid, rps in (("e7_x_clean", 18000), ("e7_x_fault", 17000), ("e7_x_hdfs_rep1", 22000), ("e7_x_hdfs_rep2", 23000)):
        (out / "runs" / f"{rid}.json").write_text(json.dumps(run_json(rid, rps)), encoding="utf-8")
    write_run(out / "clean", ROWS, TOPIC)
    write_run(out / "fault", ROWS, TOPIC)
    (out / "compare.json").write_text(json.dumps(R.compare_runs(out / "clean", out / "fault")), encoding="utf-8")
    smp = out / "fault" / "samples"
    smp.mkdir(parents=True)
    t0 = 1_000_000
    samples = [(t0 - 5000, "before", 3, 0), (t0 + 5000, "outage", 3, 0), (t0 + 95000, "outage", 2, under_during_outage),
               (t0 + 160000, "outage", 2, max(0, under_during_outage - 20)),
               (t0 + 175000, "restarted", 3, 4), (t0 + 185000, "restarted", 3, 0 if healed else 4)]
    for ts, phase, live, under in samples:
        (smp / f"{ts}_{phase}.txt").write_text(sample_text(ts, phase, live=live, under=under), encoding="utf-8")
    with (out / "events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"name": "datanode_stopped", "ts_ms": t0, "host": "spark-w5", "after_batches": 3}) + "\n")
        f.write(json.dumps({"name": "datanode_started", "ts_ms": t0 + 170000, "host": "spark-w5"}) + "\n")
    baseline = tmp_path / "e2.json"
    baseline.write_text(json.dumps({"meta": {"date": "2026-11-15", "records": 2000000, "replicate": 4},
                                    "runs": [run_json("b1", 24000, records=2000000) | {"executors": 5},
                                             run_json("b2", 25000, records=2000000),
                                             run_json("other", 99999, records=2000000) | {"executors": 1}]}),
                        encoding="utf-8")
    return out, baseline


def test_report_pass_with_timeline_throughput_and_caveats(tmp_path):
    out, baseline = fake_out(tmp_path)
    md, js = tmp_path / "e7.md", tmp_path / "e7.json"
    assert R.main(["report", "--out-dir", str(out), "--baseline", str(baseline), "--md", str(md), "--json", str(js)]) == 0
    rep = json.loads(js.read_text(encoding="utf-8"))
    b = rep["b_fault"]
    assert rep["pass"] and b["pass"] and rep["a_clean"]["anomalies"] == 1 and rep["a_clean"]["score_rows"] == 3
    assert (b["stop_host"], b["stopped_after_batches"], b["max_under_replicated"], b["max_missing"]) == ("spark-w5", 3, 30, 0)
    assert (b["dead_detected_after_s"], b["first_under_replicated_after_s"]) == (95.0, 95.0)
    assert (b["restart_after_s"], b["under_replicated_at_restart"], b["healed_after_restart_s"]) == (170.0, 10, 15.0)
    c = rep["c_throughput"]
    assert c["hdfs"]["drained"] == 2 and c["hdfs"]["records_per_s"]["mean"] == 22500
    assert c["nfs_baseline"]["runs"] == 2 and c["nfs_baseline"]["records_per_s"]["mean"] == 24500   # 5x2 runs only
    assert c["hdfs_vs_nfs_baseline"] == round(22500 / 24500, 3) and "nfs_rerun" not in c
    text = md.read_text(encoding="utf-8")
    assert "single point of failure" in text and "HA NameNode" in text and "future work" in text
    assert "**(b): PASS**" in text and "**Result: PASS**" in text and "spark-w5" in text
    assert "| 95.0 | outage | 2 | HEALTHY | 300 | 30 | 0 |" in text


def test_report_fails_without_under_replication_or_healing_or_on_missing_blocks(tmp_path):
    out, baseline = fake_out(tmp_path / "a", under_during_outage=0)
    rep = R.build_report(out, baseline)
    assert not rep["pass"] and not rep["b_fault"]["checks"]["fsck reported under-replicated blocks while the DataNode was down"]
    out, baseline = fake_out(tmp_path / "b", healed=False)
    rep = R.build_report(out, baseline)
    assert not rep["b_fault"]["pass"] and rep["b_fault"]["healed_after_restart_s"] is None
    out, baseline = fake_out(tmp_path / "c")
    f = sorted((out / "fault" / "samples").glob("*_outage.txt"))[-1]
    f.write_text(f.read_text(encoding="utf-8").replace("Missing blocks:\t\t0", "Missing blocks:\t\t2"), encoding="utf-8")
    rep = R.build_report(out, baseline)
    assert rep["b_fault"]["max_missing"] == 2 and not rep["pass"]


def test_report_throughput_only_and_thin_timeline(tmp_path):
    out = tmp_path / "e7_c"
    (out / "runs").mkdir(parents=True)
    (out / "runs" / "e7_c_hdfs_rep1.json").write_text(json.dumps(run_json("e7_c_hdfs_rep1", 22000)), encoding="utf-8")
    (out / "runs" / "e7_c_nfs_rep1.json").write_text(json.dumps(run_json("e7_c_nfs_rep1", 24000)), encoding="utf-8")
    rep = R.build_report(out, None)
    assert rep["pass"] and rep["a_clean"] is None and rep["b_fault"] is None
    assert rep["c_throughput"]["hdfs_vs_nfs_rerun"] == round(22000 / 24000, 3)
    assert "NFS (re-run today)" in R.render_md(rep)
    tl = [{"t_s": i, "phase": "outage", "live": 2, "status": "HEALTHY", "under": i, "missing": 0} for i in range(100)]
    thin = R.thin_timeline(tl)
    assert len(thin) <= 30 and thin[0] is tl[0] and thin[-1] is tl[-1]
    same = [dict(tl[0], t_s=i) for i in range(10)]
    assert R.thin_timeline(same) == [same[0], same[-1]]


# ---- sweep / drain_test / submit pass-through ---------------------------------------------------------------
def test_sweep_passes_an_hdfs_checkpoint_uri_through(capsys):
    uri = "hdfs://spark-master:9000/spark-anomaly/checkpoints/e7/{run_id}"
    assert SW.experiment_args(uri, False, "e7") == ["-k", uri, "--tag", "e7"]
    assert SW.main(["--dry-run", "--configs", "5x2", "--repeats", "1", "--checkpoint-dir", uri, "--tag", "e7"]) == 0
    assert capsys.readouterr().out.strip().endswith(f"-k '{uri}' --tag e7")


def _bash(script, **env):
    settings = {"CLUSTER_ENV": "cluster.env.example", **env}
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in settings.items())
    clean = {k: v for k, v in os.environ.items() if k.lower() not in {"http_proxy", "https_proxy", "no_proxy"}}
    return subprocess.run(["bash"], input=f"export {exports}; {script}", cwd=CLUSTER, capture_output=True,
                          text=True, timeout=60, env=clean)


def WITH(*lines):
    """Shell prefix: CLUSTER_ENV = cluster.env.example plus the given lines (they win, as in a real cluster.env)."""
    add = " ".join(shlex.quote(l) for l in lines)
    return f'tmp_env="$(mktemp)"; cat cluster.env.example > "$tmp_env"; printf "%s\\n" {add} >> "$tmp_env"; export CLUSTER_ENV="$tmp_env"; '


@needs_bash
def test_drain_test_passes_an_hdfs_checkpoint_uri_with_the_run_id():
    r = _bash("bash drain_test.sh --print-config --run-id e7_x_fault -k 'hdfs://spark-master:9000/a/e7/{run_id}'")
    assert r.returncode == 0, r.stderr
    cfg = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    assert cfg["checkpoint_dir"] == "hdfs://spark-master:9000/a/e7/e7_x_fault" and cfg["allow_local_checkpoint"] == "0"


@needs_bash
def test_submit_accepts_hdfs_uris_and_sets_the_client_replication():
    r = _bash("bash submit.sh --dry-run -e 1 -c 2 -k hdfs://spark-master:9000/spark-anomaly/checkpoints/e7/r1/ -- --reset")
    assert r.returncode == 0, r.stderr
    assert "hdfs://spark-master:9000/spark-anomaly/checkpoints/e7/r1/score/source=smd (HDFS, replication 2)" in r.stdout
    cmd = r.stdout.splitlines()[-1].split()
    i = cmd.index("--conf")
    assert cmd[i + 1] == "spark.hadoop.dfs.replication=2" and cmd.index("--conf") < cmd.index(
        "/srv/spark-shared/Anomaly-Detection/infra/cluster/score_entry.py") and cmd[-1] == "--reset"
    assert "--allow-local-checkpoint" not in r.stderr
    r = _bash(WITH("HDFS_REPLICATION=3") + "bash submit.sh --dry-run -e 1 -c 2 -k hdfs://nn:9000/x")
    assert r.returncode == 0 and "spark.hadoop.dfs.replication=3" in r.stdout
    r = _bash("bash submit.sh --dry-run -e 1 -c 2")                 # default: NFS, unchanged command
    assert r.returncode == 0 and "--conf" not in r.stdout.splitlines()[-1] and "(HDFS" not in r.stdout
    for bad, msg in (("hdfs://spark-master:9000", "is not hdfs://"), ("hdfs://spark-master:9000/a/../b", "is not hdfs://"),
                     ("s3a://bucket/x", "only hdfs://")):
        r = _bash(f"bash submit.sh --dry-run -e 1 -c 2 -k '{bad}'")
        assert r.returncode != 0 and msg in r.stderr, (bad, r.stderr)


@needs_bash
def test_lib_sh_hdfs_defaults_and_uri_helpers():
    r = _bash('source ./lib.sh && echo "$HDFS_URI|$HDFS_REPLICATION|$(hdfs_hosts | paste -sd, -)|$HADOOP_INSTALL_DIR"; '
              'u=hdfs://nn.x:9000/a/b; echo "$(hdfs_uri_host $u) $(hdfs_uri_port $u) $(hdfs_uri_path $u)"; '
              'echo "$(hdfs_uri_port hdfs://nn/a)"; is_hdfs_uri /srv/x || echo local; render_runtime_env | grep HDFS_',
              HADOOP_HOME="C:/Users/x/hadoop")                     # a laptop's winutils HADOOP_HOME is not used
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "hdfs://spark-master:9000|2|spark-master,spark-w2,spark-w5|/opt/hadoop"
    assert lines[1:4] == ["nn.x 9000 /a/b", "8020", "local"]
    assert "HDFS_URI=hdfs://spark-master:9000" in lines and "HDFS_REPLICATION=2" in lines
    # a cluster.env without the HDFS lines still gets the defaults from lib.sh
    r = _bash('tmp="$(mktemp)"; grep -vE "^(HDFS_|HADOOP_)" cluster.env.example > "$tmp"; export CLUSTER_ENV="$tmp"; '
              'source ./lib.sh && echo "$HDFS_DATANODE_HOSTS|$HDFS_NAMENODE_HEAP|$HDFS_DATANODE_HEAP"')
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "spark-master spark-w2 spark-w5|1g|512m"


@needs_bash
def test_hdfs_print_config_phase_and_health_parsers():
    r = _bash("bash ./50_hdfs.sh --print-config")
    assert r.returncode == 0, r.stderr
    cfg = dict(l.split("=", 1) for l in r.stdout.strip().splitlines())
    assert cfg["HADOOP_URL"] == "https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz"
    assert (cfg["HDFS_NAMENODE_HOST"], cfg["HDFS_DATANODE_HOSTS"]) == ("spark-master", "spark-master spark-w2 spark-w5")
    assert (cfg["HDFS_NAMENODE_HEAP"], cfg["HDFS_DATANODE_HEAP"], cfg["HDFS_REPLICATION"]) == ("1g", "512m", "2")
    assert cfg["HDFS_URI"] == "hdfs://spark-master:9000" and cfg["NAME_DIR"] == "/var/lib/hadoop-hdfs/name"
    r = _bash(WITH("HDFS_REPLICATION=4") + "bash ./50_hdfs.sh --print-config")
    assert r.returncode != 0 and "exceeds" in r.stderr
    deploy = (CLUSTER / "deploy.sh").read_text(encoding="utf-8")
    assert "hdfs) phase_hdfs ;;" in deploy and "phases=(ssh env proxy prereqs nfs repo kafka spark topics health)" in deploy
    health = (CLUSTER / "health_check.sh").read_text(encoding="utf-8")
    assert "hdfs) check_hdfs ;;" in health and "all) check_kafka; check_spark; check_storage; check_memory ;;" in health
    fsck = FSCK.format(status="HEALTHY", dn=2, blocks=12, under=3, missing=0).replace("\t", "  ")
    r = _bash("source ./lib.sh; eval \"$(sed -n '/^live_datanodes()/p; /^fsck_count()/p' health_check.sh)\"; "
              f"live_datanodes {shlex.quote(REPORT.format(live=2))}; "
              f"fsck_count {shlex.quote(fsck)} 'Under-replicated blocks'; fsck_count {shlex.quote(fsck)} 'Missing blocks'")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-3:] == ["2", "3", "0"]


def test_e7_script_options_and_defaults():
    text = (Path(__file__).resolve().parents[1] / "experiments" / "e7_hdfs.sh").read_text(encoding="utf-8")
    for want in ("STOP_HOST=spark-w5", "STOP_AFTER=3", "RECORDS=1416825", "TPUT_RECORDS=2000000", "TPUT_REPLICATE=4",
                 'systemctl stop hadoop-datanode', 'systemctl start hadoop-datanode', "hdfs_on hdfs fsck",
                 '-k "${HDFS_E7_ROOT}/{run_id}"', "trap cleanup EXIT"):
        assert want in text, want
