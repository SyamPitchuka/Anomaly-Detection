"""Run the unmodified tools/check_kafka.py against the cluster.

tools/check_kafka.py is a frozen gauge: it reads contracts.schemas.KAFKA_BOOTSTRAP (localhost:9092)
and compares partition counts with `partitions_laptop`. This wrapper changes both IN MEMORY ONLY
(bootstrap -> the cluster, partitions_laptop -> partitions_cluster) and then runs the real script,
so the round-trip logic and the final PASS/FAIL line are exactly the ones used on laptops.

Run from the repo root (data/sample/telemetry_smd.jsonl must exist: python tools/fake_data.py):
    python infra/cluster/check_kafka_cluster.py
    python infra/cluster/check_kafka_cluster.py --bootstrap spark-master:9092     # one broker; add :9192,:9292 for three
"""
from __future__ import annotations

import argparse
import runpy
import sys

import yaml

from cluster_common import ROOT, resolve_bootstrap

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--bootstrap", default=None, help="comma-separated broker list")
args = ap.parse_args()
bootstrap = resolve_bootstrap(args.bootstrap)

from contracts import schemas as S  # noqa: E402  (ROOT is on sys.path via cluster_common)

S.KAFKA_BOOTSTRAP = bootstrap

_orig_safe_load = yaml.safe_load


def _cluster_view(stream):
    """topics.yaml as check_kafka.py sees it: the cluster partition counts under the laptop keys."""
    cfg = _orig_safe_load(stream)
    if isinstance(cfg, dict) and "topics" in cfg:
        for topic in cfg["topics"].values():
            topic["partitions_laptop"] = topic["partitions_cluster"]
    return cfg


yaml.safe_load = _cluster_view
print(f"check_kafka.py against cluster: {bootstrap}")
sys.argv = [str(ROOT / "tools" / "check_kafka.py")]
runpy.run_path(sys.argv[0], run_name="__main__")  # ends with sys.exit(0 or 1)
