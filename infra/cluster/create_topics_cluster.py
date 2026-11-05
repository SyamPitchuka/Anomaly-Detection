"""Create the frozen topics on the cluster, using partitions_cluster / replication_cluster.

Run from the repo root (venv active, brokers reachable by name):
    python infra/cluster/create_topics_cluster.py
    python infra/cluster/create_topics_cluster.py --bootstrap spark-master:9092
    python infra/cluster/create_topics_cluster.py --bootstrap spark-master:9092,spark-master:9192,spark-master:9292

Safe to run again: existing topics are never altered, only verified. A topic that exists with a
different partition count or replication factor is reported as WRONG and the script exits 1
(Kafka cannot reduce partitions; delete and recreate the topic if you really want to change it).

contracts/topics.yaml asks for replication_cluster = 3. A cluster with fewer brokers cannot hold that, so the
replication factor is capped at KAFKA_BROKER_COUNT (from cluster.runtime.env; 3 if unknown), e.g. 1 on the
one-broker trial cluster. contracts/ itself is not edited.
Local variant for the laptop broker: infra/create_topics.py.
"""
from __future__ import annotations

import argparse
import sys
import time

import yaml
from confluent_kafka.admin import AdminClient, NewTopic

from cluster_common import ROOT, broker_count, effective_replication, load_runtime_env, resolve_bootstrap


def describe(meta, name: str) -> tuple[int, int, int]:
    """(partitions, min replication factor over partitions, partitions without a leader)."""
    topic = meta.topics.get(name)
    parts = len(topic.partitions) if topic is not None else 0
    rf = min((len(p.replicas) for p in topic.partitions.values()), default=0) if topic is not None else 0
    leaderless = sum(1 for p in topic.partitions.values() if p.leader < 0) if topic is not None else 0
    return parts, rf, leaderless


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bootstrap", default=None, help="comma-separated broker list")
    ap.add_argument("--brokers", type=int, default=None,
                    help="number of brokers in the cluster (default: KAFKA_BROKER_COUNT from cluster.runtime.env, else 3)")
    args = ap.parse_args(argv)
    bootstrap = resolve_bootstrap(args.bootstrap)
    status = 0

    cfg = yaml.safe_load((ROOT / "contracts" / "topics.yaml").read_text(encoding="utf-8"))
    n_brokers = args.brokers if args.brokers and args.brokers >= 1 else broker_count(load_runtime_env(), 3)
    want_rf = {name: effective_replication(t["replication_cluster"], n_brokers) for name, t in cfg["topics"].items()}
    admin = AdminClient({"bootstrap.servers": bootstrap})
    print(f"bootstrap: {bootstrap}")
    meta = admin.list_topics(timeout=15)
    need_rf = max(want_rf.values())
    print(f"brokers visible: {len(meta.brokers)} (cluster has {n_brokers}; largest replication factor wanted is {need_rf})")
    if len(meta.brokers) < need_rf:
        print(f"FAIL only {len(meta.brokers)} brokers up, replication factor {need_rf} needs {need_rf}")
        status = 1

    if status == 0:
        new = [
            NewTopic(name, num_partitions=t["partitions_cluster"], replication_factor=want_rf[name])
            for name, t in cfg["topics"].items() if name not in meta.topics
        ]
        for name, fut in (admin.create_topics(new).items() if new else []):
            try:
                fut.result()
                print(f"created {name}")
            except Exception as e:  # noqa: BLE001
                print(f"FAIL creating {name}: {e}")
                status = 1

    if status == 0:
        deadline = time.time() + 30
        while True:  # wait until every partition has a leader
            meta = admin.list_topics(timeout=15)
            if all(describe(meta, n)[2] == 0 and describe(meta, n)[0] > 0 for n in cfg["topics"]) or time.time() > deadline:
                break
            time.sleep(1)
        for name, t in cfg["topics"].items():
            parts, rf, leaderless = describe(meta, name)
            ok = parts == t["partitions_cluster"] and rf == want_rf[name] and leaderless == 0
            print(f"{name}: {parts} partitions (want {t['partitions_cluster']}), "
                  f"replication {rf} (want {want_rf[name]}), {leaderless} without leader "
                  f"[{'ok' if ok else 'WRONG'}]")
            if not ok:
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
