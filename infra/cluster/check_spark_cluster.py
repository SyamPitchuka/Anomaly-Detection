"""Spark cluster gauge: every worker has its executors, tasks run on all of them, shared storage is shared.

The expected numbers come from cluster.runtime.env (default trial cluster: 2 workers x 1 executor x 2 cores).

Run ON THE SPARK MASTER VM as the 'spark' user, with the cluster venv, from the repo on the share:
    cd /srv/spark-shared/Anomaly-Detection
    sudo -u spark /opt/anomaly-venv/bin/python infra/cluster/check_spark_cluster.py

What it proves (ends with PASS or FAIL):
  1. an application registered on the standalone master gets WORKERS x EXECUTORS_PER_WORKER executors
     with SPARK_EXECUTOR_CORES cores each, spread evenly over the workers;
  2. tasks really run on every worker host (hostname returned by the tasks);
  3. every worker can write into <repo>/checkpoints/_gauge and the driver reads all files back
     (the shared checkpoint storage works across nodes);
  4. executors use the same Python minor version as the driver and can import the repo (contracts.schemas).
"""
from __future__ import annotations

import os
import socket
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from cluster_common import ROOT, load_runtime_env, split_list

cfg = load_runtime_env()
master_url = cfg.get("SPARK_MASTER_URL")
if not master_url:
    print("FAIL SPARK_MASTER_URL unknown; run: bash infra/cluster/deploy.sh env && deploy.sh repo")
    sys.exit(2)
workers = split_list(cfg.get("SPARK_WORKER_HOSTS"))
per_worker = int(cfg.get("EXECUTORS_PER_WORKER", "2"))
exec_cores = int(cfg.get("SPARK_EXECUTOR_CORES", "2"))
want_executors = len(workers) * per_worker
checkpoint_dir = Path(cfg.get("CHECKPOINT_DIR", str(ROOT / "checkpoints")))

if Path("/opt/spark/bin/spark-submit").exists():
    os.environ.setdefault("SPARK_HOME", "/opt/spark")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession  # noqa: E402

fails: list[str] = []


def report(ok: bool, text: str) -> None:
    print(f"{'OK  ' if ok else 'FAIL'} {text}")
    if not ok:
        fails.append(text)


spark = (SparkSession.builder.appName("check_spark_cluster").master(master_url)
         .config("spark.executor.cores", str(exec_cores))
         .config("spark.cores.max", str(want_executors * exec_cores))
         .config("spark.executorEnv.PYTHONPATH", str(ROOT))
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Executors register asynchronously: wait until the expected number is there (or give up after 90 s).
# The status tracker lists the driver as well, hence the "- 1".
deadline = time.time() + 90
while time.time() < deadline:
    if len(sc._jsc.sc().statusTracker().getExecutorInfos()) - 1 >= want_executors:
        break
    time.sleep(2)

n_slots = want_executors * exec_cores


def probe(_):
    import os as _os
    import socket as _socket
    import sys as _sys
    try:
        import contracts.schemas as _s  # noqa: F401
        importable = True
    except Exception:  # noqa: BLE001
        importable = False
    yield (_socket.gethostname(), "%d.%d" % _sys.version_info[:2], importable)


rows = sc.parallelize(range(n_slots * 8), n_slots * 4).mapPartitions(probe).collect()
by_host = Counter(h for h, _, _ in rows)
driver_py = "%d.%d" % sys.version_info[:2]

# Executors come from the status tracker (host:port per executor). It lists the driver too, which is
# dropped by its host name: the driver runs on the master VM and no worker runs there.
driver_host = sc.getConf().get("spark.driver.host", socket.gethostname())
exec_per_host = Counter(i.host() for i in sc._jsc.sc().statusTracker().getExecutorInfos() if i.host() != driver_host)
n_executors = sum(exec_per_host.values())

print(f"driver {driver_host}, python {driver_py}, master {master_url}")
for host in sorted(exec_per_host):
    print(f"     {host}: {exec_per_host[host]} executor(s), {by_host.get(host, 0)} probe tasks")

report(n_executors == want_executors, f"{n_executors} executors registered (want {want_executors} = {len(workers)} workers x {per_worker})")
report(set(by_host) == set(exec_per_host) and len(by_host) == len(workers),
       f"probe tasks ran on {len(by_host)} worker hosts (want {len(workers)}, every executor host used)")
report(all(n == per_worker for n in exec_per_host.values()) and len(exec_per_host) == len(workers),
       f"executors per worker: {dict(exec_per_host)} (want {per_worker} each)")
report({p for _, p, _ in rows} == {driver_py}, f"executor Python versions {sorted({p for _, p, _ in rows})} match driver {driver_py}")
report(all(i for _, _, i in rows), "executors can import contracts.schemas from the shared repo")

# shared checkpoint storage: every task host writes a file, driver (on another path through NFS) reads them back
run_id = uuid.uuid4().hex[:8]
gauge = checkpoint_dir / "_gauge" / run_id
gauge.mkdir(parents=True, exist_ok=True)


def write_marker(i):
    import socket as _socket
    from pathlib import Path as _Path
    p = _Path(str(gauge)) / f"{_socket.gethostname()}-{i}.txt"
    p.write_text(_socket.gethostname(), encoding="utf-8")
    yield _socket.gethostname()


writers = set(sc.parallelize(range(n_slots * 4), n_slots * 4).mapPartitionsWithIndex(lambda i, _it: write_marker(i)).collect())
time.sleep(2)  # NFS attribute cache
seen_hosts = {p.read_text(encoding="utf-8") for p in gauge.glob("*.txt")}
report(seen_hosts == writers and len(writers) == len(workers),
       f"driver read back files written by {len(seen_hosts)}/{len(workers)} worker hosts under {gauge}")
for p in gauge.glob("*.txt"):
    p.unlink()
try:
    gauge.rmdir()
    gauge.parent.rmdir()
except OSError:
    pass

spark.stop()
print("PASS" if not fails else "FAIL")
sys.exit(0 if not fails else 1)
