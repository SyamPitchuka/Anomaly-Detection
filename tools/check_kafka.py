"""Inspection gauge for WP-2: Kafka is up, topics are right, records round-trip.

Run from the repo root (Kafka running, topics created):  python tools/check_kafka.py
Sends the fake SMD sample to telemetry.smd and reads it back. Ends with PASS or FAIL.
"""
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from confluent_kafka import Consumer, Producer  # noqa: E402
from confluent_kafka.admin import AdminClient  # noqa: E402

from contracts import schemas as S  # noqa: E402

fails = []
cfg = yaml.safe_load((ROOT / "contracts/topics.yaml").read_text())
conf = {"bootstrap.servers": S.KAFKA_BOOTSTRAP}

try:
    meta = AdminClient(conf).list_topics(timeout=10)
except Exception as e:  # noqa: BLE001
    print(f"FAIL cannot reach Kafka at {S.KAFKA_BOOTSTRAP}: {e}")
    sys.exit(1)

for name, t in cfg["topics"].items():
    parts = len(meta.topics[name].partitions) if name in meta.topics else 0
    ok = parts == t["partitions_laptop"]
    print(f"{'OK  ' if ok else 'FAIL'} topic {name}: {parts} partitions (want {t['partitions_laptop']})")
    if not ok:
        fails.append(name)

sample = ROOT / "data/sample/telemetry_smd.jsonl"
if not sample.exists():
    print("FAIL data/sample/telemetry_smd.jsonl missing; run python tools/fake_data.py first")
    sys.exit(1)

run_id = uuid.uuid4().hex[:8]
recs = [json.loads(line) for line in sample.open()]
p = Producer(conf)
for r in recs:
    r["attrs"] = {**r["attrs"], "run_id": run_id}
    p.produce(S.TOPIC_SMD, key=r["entity_id"].encode(), value=json.dumps(r, separators=(",", ":")).encode())
p.flush(10)
print(f"OK   produced {len(recs)} records (run_id={run_id})")

c = Consumer({**conf, "group.id": f"check-{run_id}", "auto.offset.reset": "earliest"})
c.subscribe([S.TOPIC_SMD])
seen, wrong_key, deadline = 0, 0, time.time() + 30
while seen < len(recs) and time.time() < deadline:
    m = c.poll(1.0)
    if m is None or m.error():
        continue
    v = json.loads(m.value())
    if v.get("attrs", {}).get("run_id") == run_id:
        seen += 1
        wrong_key += m.key().decode() != v["entity_id"]
c.close()

print(f"{'OK  ' if seen == len(recs) else 'FAIL'} consumed {seen}/{len(recs)} records back")
print(f"{'OK  ' if not wrong_key else 'FAIL'} message key equals entity_id ({wrong_key} wrong)")
if seen != len(recs) or wrong_key:
    fails.append("roundtrip")
print("PASS" if not fails else "FAIL")
sys.exit(0 if not fails else 1)
