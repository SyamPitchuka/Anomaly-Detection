"""Tests for the pure-Python parts of infra/cluster (no Kafka, Spark or VMs needed)."""
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "infra" / "cluster"
if str(CLUSTER) not in sys.path:
    sys.path.insert(0, str(CLUSTER))

import cluster_common  # noqa: E402
import create_topics_cluster  # noqa: E402

# The bigger layout (what the scripts were first written for), applied on top of cluster.env.example.
FULL_LAYOUT = (
    'tmp_env="$(mktemp)"; cat cluster.env.example > "$tmp_env"; '
    'printf "%s\\n" "VM_PROFILE=vm_8gb_4cpu" "KAFKA_BROKER_COUNT=3" '
    '"SPARK_WORKER_HOSTS=(spark-w1 spark-w2 spark-w3 spark-w4)" >> "$tmp_env"; '
    'export CLUSTER_ENV="$tmp_env"; '
)


def _lib(script, **env):
    settings = {"CLUSTER_ENV": "cluster.env.example", **env}
    exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in settings.items())
    return subprocess.run(
        ["bash"],
        input=f"export {exports}; {script}",
        cwd=CLUSTER, capture_output=True, text=True, timeout=30, env=os.environ,
    )


def _meta(partitions_by_topic):
    """Fake confluent_kafka ClusterMetadata: {topic: [(leader, n_replicas), ...]}."""
    topics = {
        name: SimpleNamespace(partitions={
            i: SimpleNamespace(leader=leader, replicas=list(range(n_rep)))
            for i, (leader, n_rep) in enumerate(parts)
        })
        for name, parts in partitions_by_topic.items()
    }
    return SimpleNamespace(topics=topics)


def test_describe_counts_partitions_replication_and_leaderless():
    meta = _meta({"t": [(1, 3), (2, 3), (-1, 2)]})
    assert create_topics_cluster.describe(meta, "t") == (3, 2, 1)
    assert create_topics_cluster.describe(meta, "missing") == (0, 0, 0)


def test_replication_is_capped_at_the_number_of_brokers():
    # contracts/topics.yaml says replication_cluster = 3; a one-broker cluster can only hold 1
    assert cluster_common.effective_replication(3, 1) == 1
    assert cluster_common.effective_replication(3, 2) == 2
    assert cluster_common.effective_replication(3, 3) == 3
    assert cluster_common.effective_replication(3, 5) == 3
    assert cluster_common.effective_replication(1, 3) == 1
    assert cluster_common.effective_replication(3, 0) == 1      # never below 1


def test_broker_count_reads_runtime_env_and_falls_back():
    assert cluster_common.broker_count({"KAFKA_BROKER_COUNT": "1"}, 3) == 1
    assert cluster_common.broker_count({"KAFKA_BROKER_COUNT": "3"}, 1) == 3
    assert cluster_common.broker_count({}, 3) == 3
    assert cluster_common.broker_count({"KAFKA_BROKER_COUNT": "x"}, 3) == 3
    assert cluster_common.broker_count({"KAFKA_BROKER_COUNT": "0"}, 3) == 3


def test_load_runtime_env_reads_file_and_env_overrides(tmp_path, monkeypatch):
    f = tmp_path / "cluster.runtime.env"
    f.write_text("# comment\nKAFKA_BOOTSTRAP=a:9092,b:9092\nSPARK_WORKER_HOSTS=a,b\n\nSPARK_MASTER_URL=spark://m:7077\n")
    monkeypatch.setattr(cluster_common, "RUNTIME_ENV", f)
    monkeypatch.delenv("KAFKA_BOOTSTRAP", raising=False)
    monkeypatch.delenv("SPARK_MASTER_URL", raising=False)
    cfg = cluster_common.load_runtime_env()
    assert cfg["KAFKA_BOOTSTRAP"] == "a:9092,b:9092"
    assert cluster_common.split_list(cfg["SPARK_WORKER_HOSTS"]) == ["a", "b"]
    monkeypatch.setenv("KAFKA_BOOTSTRAP", "x:9092")
    assert cluster_common.load_runtime_env()["KAFKA_BOOTSTRAP"] == "x:9092"


def test_resolve_bootstrap_precedence_and_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_common, "RUNTIME_ENV", tmp_path / "missing.env")
    monkeypatch.delenv("KAFKA_BOOTSTRAP", raising=False)
    assert cluster_common.resolve_bootstrap("cli:9092") == "cli:9092"
    monkeypatch.setenv("KAFKA_BOOTSTRAP", "env:9092")
    assert cluster_common.resolve_bootstrap(None) == "env:9092"
    monkeypatch.delenv("KAFKA_BOOTSTRAP")
    with pytest.raises(SystemExit) as exc:
        cluster_common.resolve_bootstrap(None)
    assert exc.value.code == 2


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_lib_sh_derives_the_trial_layout_from_the_example_env():
    # default: 3 VMs (spark-master + spark-w1 + spark-w2), ONE Kafka broker on spark-master
    script = ('source ./lib.sh && validate_layout && echo '
              '"$KAFKA_BOOTSTRAP|$KAFKA_QUORUM_VOTERS|$EXECUTORS_PER_WORKER|$SPARK_CORES_MAX|$SPARK_MASTER_URL'
              '|$SPARK_WORKER_COUNT|$KAFKA_REPLICATION|$KAFKA_MIN_ISR|$VM_PROFILE"')
    out = _lib(script)
    assert out.returncode == 0, out.stderr
    bootstrap, voters, per_worker, cores_max, master, workers, rf, isr, profile = (
        out.stdout.strip().splitlines()[-1].split("|"))
    assert bootstrap == "spark-master:9092"
    assert voters == "1@spark-master:9093"
    assert workers == "2"
    assert per_worker == "1"          # 2-core workers, 2-core executors
    assert cores_max == "4"
    assert master == "spark://spark-master:7077"
    assert (rf, isr) == ("1", "1")    # one broker: replication 1, min ISR 1
    assert profile == "vm_8gb_2cpu"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_lib_sh_derives_the_full_layout_when_asked():
    # 4 workers of 4 cores and three brokers as Docker containers on one VM: distinct ports per broker
    script = (FULL_LAYOUT + 'source ./lib.sh && validate_layout && echo '
              '"$KAFKA_BOOTSTRAP|$KAFKA_QUORUM_VOTERS|$EXECUTORS_PER_WORKER|$SPARK_CORES_MAX|$KAFKA_REPLICATION|$KAFKA_MIN_ISR"')
    out = _lib(script)
    assert out.returncode == 0, out.stderr
    bootstrap, voters, per_worker, cores_max, rf, isr = out.stdout.strip().splitlines()[-1].split("|")
    assert bootstrap == "spark-master:9092,spark-master:9192,spark-master:9292"
    assert voters == "1@spark-master:9093,2@spark-master:9193,3@spark-master:9293"
    assert per_worker == "2"
    assert cores_max == "16"
    assert (rf, isr) == ("3", "2")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_lib_sh_rejects_a_bad_broker_count_and_an_empty_worker_list():
    for bad in ("0", "x"):
        out = _lib('source ./lib.sh && (validate_layout)', KAFKA_BROKER_COUNT=bad)
        assert out.returncode != 0, bad
        assert "KAFKA_BROKER_COUNT" in out.stderr
    # an env file with no workers
    script = ('tmp_env="$(mktemp)"; cat cluster.env.example > "$tmp_env"; echo "SPARK_WORKER_HOSTS=()" >> "$tmp_env"; '
              'export CLUSTER_ENV="$tmp_env"; source ./lib.sh && (validate_layout)')
    out = _lib(script)
    assert out.returncode != 0
    assert "SPARK_WORKER_HOSTS" in out.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_lib_sh_memory_budget_fits_8gb_vms_and_rejects_oversized_executors():
    base = 'source ./lib.sh && validate_layout && echo "$WORKER_NEED_MB|$MASTER_NEED_MB|$VM_MB"'
    ok = _lib(base)
    assert ok.returncode == 0, ok.stderr
    worker, master, vm = (int(x) for x in ok.stdout.strip().splitlines()[-1].split("|"))
    # worker (2 vCPU): 1 x (4g + 1536m) + 512m daemon + 1g OS = 7168 MB
    # master: 512m daemon + 1g driver + 384m overhead + 1g OS + 1 Kafka container x 1536m = 4480 MB
    assert worker == 7168 and worker <= vm == 8192
    assert master == 4480
    # the bigger layout: 2 x (2g + 768m) + 512m + 1g = 7168 on a worker, plus 3 x 1280m Kafka containers on the master
    full = _lib(FULL_LAYOUT + base)
    assert full.returncode == 0, full.stderr
    worker, master, vm = (int(x) for x in full.stdout.strip().splitlines()[-1].split("|"))
    assert worker == 7168 and master == 6784 and master <= vm == 8192
    # an executor size that does not fit an 8 GB VM must be refused by validate_layout
    oversized = _lib('source ./lib.sh && SPARK_EXECUTOR_MEMORY=6g && SPARK_WORKER_MEMORY=6g && (validate_layout)')
    assert oversized.returncode != 0
    assert "needs" in oversized.stderr
    # three brokers on the 2 vCPU profile still pass the check (7552 MB of 8192) but leave under 0.4 GB of real slack
    # on a ~7.7 GB kernel view; the profile header and the README warn against it
    three = _lib(base, KAFKA_BROKER_COUNT="3")
    assert three.returncode == 0, three.stderr
    assert int(three.stdout.strip().splitlines()[-1].split("|")[1]) == 7552


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_every_profile_in_profiles_dir_passes_validate_layout():
    names = sorted(p.stem for p in (CLUSTER / "profiles").glob("*.env"))
    assert {"vm_8gb_2cpu", "vm_8gb_4cpu", "vm_16gb_4cpu"} <= set(names)
    for name in names:
        out = _lib('source ./lib.sh && validate_layout && echo "$VM_MB|$WORKER_NEED_MB|$MASTER_NEED_MB"', VM_PROFILE=name)
        assert out.returncode == 0, f"{name}: {out.stderr}"
        vm, worker, master = (int(x) for x in out.stdout.strip().splitlines()[-1].split("|"))
        assert worker <= vm and master <= vm, name


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_cluster_env_value_wins_over_the_profile_and_unknown_profile_is_rejected():
    # cluster.env is sourced before the profile and the profile only sets unset variables
    out = _lib('SPARK_EXECUTOR_MEMORY=1g; source ./lib.sh && echo "$SPARK_EXECUTOR_MEMORY|$SPARK_EXECUTOR_CORES"')
    assert out.stdout.strip().splitlines()[-1] == "1g|2"
    out = _lib('source ./lib.sh && echo "$SPARK_EXECUTOR_MEMORY"', VM_PROFILE="no_such_profile")
    assert out.returncode != 0
    assert "profile" in out.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_runtime_env_carries_the_broker_count_for_the_python_tools():
    out = _lib('source ./lib.sh && render_runtime_env')
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert "KAFKA_BROKER_COUNT=1" in lines
    assert "KAFKA_BOOTSTRAP=spark-master:9092" in lines
    assert "SPARK_WORKER_HOSTS=spark-w1,spark-w2" in lines
    assert "SPARK_CORES_MAX=4" in lines
