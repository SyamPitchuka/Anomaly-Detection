"""Run an existing repo job or check against the cluster without editing it.

    python infra/cluster/run_job.py src/streaming/features_job.py
    python infra/cluster/run_job.py src/streaming/score_job.py --input kafka
    python infra/cluster/run_job.py tools/check_spark_kafka.py        # PASS here = Spark executors read Kafka

Run it ON THE SPARK MASTER VM (the driver must be reachable from the executors), as the 'spark' user,
with the cluster venv:   sudo -u spark /opt/anomaly-venv/bin/python infra/cluster/run_job.py ...

The jobs hard-code SparkSession...master("local[*]") and contracts.schemas.KAFKA_BOOTSTRAP
(localhost:9092). Both are replaced IN MEMORY before the script starts:
  * SparkSession.Builder.master(...)  -> the standalone master (spark://spark-master:7077)
  * contracts.schemas.KAFKA_BOOTSTRAP -> the cluster's broker list
Everything else (checkpoint paths under <repo>/checkpoints, output under <repo>/data) is untouched; on the
cluster the repo lives on the NFS share, so those paths are the shared storage.

One application takes all cores by default (spark.cores.max = workers x worker cores; 4 on the 2-worker trial
cluster, 16 on the 4-worker one). To run the features job and the score job at the same time, cap each one,
e.g. on the trial cluster:  CLUSTER_CORES_MAX=2 python infra/cluster/run_job.py ...
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from cluster_common import ROOT, load_runtime_env

# Use /opt/spark (with conf/spark-defaults.conf written by 30_spark_node.sh), not the pip-bundled Spark.
if Path("/opt/spark/bin/spark-submit").exists():
    os.environ.setdefault("SPARK_HOME", "/opt/spark")
# One BLAS/OpenMP thread per Python process (1 Python worker per executor core).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)

cfg = load_runtime_env()
master_url = cfg.get("SPARK_MASTER_URL")
bootstrap = cfg.get("KAFKA_BOOTSTRAP")
if not master_url or not bootstrap:
    print("FAIL SPARK_MASTER_URL / KAFKA_BOOTSTRAP unknown; run: bash infra/cluster/deploy.sh env && deploy.sh repo")
    sys.exit(2)

from contracts import schemas as S  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

S.KAFKA_BOOTSTRAP = bootstrap


cores_cap = os.environ.get("CLUSTER_CORES_MAX")  # e.g. 2 -> one 2-core executor, so two jobs can share the cluster


def _cluster_master(self, master):  # noqa: ANN001
    builder = self.config("spark.master", master_url)
    if cores_cap:
        builder = builder.config("spark.cores.max", cores_cap)
    return builder


SparkSession.Builder.master = _cluster_master

script = Path(sys.argv[1])
if not script.is_absolute():
    script = (Path.cwd() / script) if (Path.cwd() / script).exists() else (ROOT / script)
print(f"run_job: {script.name} on {master_url}, Kafka {bootstrap}", flush=True)
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
