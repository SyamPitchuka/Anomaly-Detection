"""Create the frozen topics from contracts/topics.yaml on the laptop broker.

Run from the repo root:  python infra/create_topics.py
Safe to run again: existing topics are left alone.
"""
import sys
from pathlib import Path

import yaml
from confluent_kafka.admin import AdminClient, NewTopic

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "contracts" / "topics.yaml").read_text())
admin = AdminClient({"bootstrap.servers": cfg["bootstrap_laptop"]})

existing = set(admin.list_topics(timeout=10).topics)
new = [
    NewTopic(name, num_partitions=t["partitions_laptop"], replication_factor=t["replication_laptop"])
    for name, t in cfg["topics"].items() if name not in existing
]
for name, fut in (admin.create_topics(new).items() if new else []):
    try:
        fut.result()
        print(f"created {name}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL creating {name}: {e}")
        sys.exit(1)

meta = admin.list_topics(timeout=10).topics
for name, t in cfg["topics"].items():
    parts = len(meta[name].partitions) if name in meta else 0
    status = "ok" if parts == t["partitions_laptop"] else "WRONG"
    print(f"{name}: {parts} partitions [{status}]")
