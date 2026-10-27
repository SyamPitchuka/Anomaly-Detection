"""Toy end-to-end run of the whole SPARK pipeline, small enough to check by hand.

Run from the repo root:  python tools/toy_pipeline.py            (prints every stage)
                         python tools/toy_pipeline.py --json out.json

This is a teaching model, not the real system: plain Python, numpy and scikit-learn stand in
for Kafka and Spark, but every name, column and rule follows contracts/schemas.py. Each stage
prints what the real stage must produce, so teammates can compare their own output against it.
Needs: numpy, scikit-learn (pip install scikit-learn).
"""
import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
MACHINES = ["machine-1-1", "machine-1-2", "machine-2-1"]
METRICS = ["m00", "m01", "m02"]          # toy meaning: m00 = CPU %, m01 = network, m02 = disk
NAMES = {"m00": "cpu", "m01": "net", "m02": "disk"}
TRAIN_MIN, TEST_MIN, WIN = 600, 60, 5
PARTITIONS = 4
R = {}                                    # results for the web page


def murmur2(data: bytes) -> int:
    """Kafka's default partitioner hash (same constants as the Java client)."""
    seed, m, r = 0x9747B28C, 0x5BD1E995, 24
    length = len(data)
    h = (seed ^ length) & 0xFFFFFFFF
    i = 0
    while length - i >= 4:
        k = int.from_bytes(data[i:i + 4], "little")
        k = (k * m) & 0xFFFFFFFF
        k ^= k >> r
        k = (k * m) & 0xFFFFFFFF
        h = (h * m) & 0xFFFFFFFF
        h ^= k
        i += 4
    rest = length - i
    if rest == 3:
        h ^= data[i + 2] << 16
    if rest >= 2:
        h ^= data[i + 1] << 8
    if rest >= 1:
        h ^= data[i]
        h = (h * m) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * m) & 0xFFFFFFFF
    h ^= h >> 15
    return h


def partition_for(key: str) -> int:
    return (murmur2(key.encode()) & 0x7FFFFFFF) % PARTITIONS


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


# ---------------------------------------------------------------- 1-2. sources + adapter
def make_rows():
    rng = np.random.default_rng(7)
    rows = []
    for e in MACHINES:
        for i in range(TRAIN_MIN + TEST_MIN):
            test = i >= TRAIN_MIN
            j = i - TRAIN_MIN
            cpu = 50 + 25 * math.sin(i / 37) + rng.normal(0, 2)          # slow daily-like wave
            net = 100 * (cpu / 100) ** 2 + rng.normal(0, 1.5)              # net follows cpu on a curve
            label = 0 if test else None
            if test and e == "machine-1-2" and 20 <= j < 35:              # correlation break
                cpu, net, label = 60 + rng.normal(0, 1), 27 + rng.normal(0, 1), 1   # each value normal alone
            disk = 20 + 0.2 * cpu + rng.normal(0, 1)
            if test and e == "machine-2-1" and 40 <= j < 45:              # plain disk spike
                disk, label = 60 + rng.normal(0, 0.5), 1
            rows.append({"source": "smd", "entity_id": e, "event_time": BASE + timedelta(minutes=i),
                         "metrics": {"m00": round(cpu, 2), "m01": round(net, 2), "m02": round(disk, 2)},
                         "attrs": {"split": "test" if test else "train"}, "label": label})
    return rows


rows = make_rows()
sample = next(r for r in rows if r["entity_id"] == "machine-1-2" and r["attrs"]["split"] == "test"
              and r["event_time"] == BASE + timedelta(minutes=TRAIN_MIN + 22))
R["adapter"] = {**sample, "event_time": iso(sample["event_time"])}
R["sources"] = {"machines": len(MACHINES), "metrics": len(METRICS), "train_rows": TRAIN_MIN * len(MACHINES),
                "test_rows": TEST_MIN * len(MACHINES), "anomalous_rows": sum(1 for r in rows if r["label"] == 1)}

# ---------------------------------------------------------------- 3. replay into Kafka
parts = {e: partition_for(e) for e in MACHINES}
R["kafka"] = {"partitions": PARTITIONS, "assignment": parts,
              "speedups": [{"x": x, "trace_min": 60, "wall_s": round(3600 / x, 1)} for x in (1, 100, 1000)]}

# ---------------------------------------------------------------- 4. watermark
arrivals = [("00:21", 21), ("00:23", 23), ("00:09", 9), ("00:24", 24)]
wm, max_seen, wm_rows = None, -1, []
for label, minute in arrivals:
    max_seen = max(max_seen, minute)
    wm = max_seen - 10
    wm_rows.append({"arrives": label, "watermark": f"00:{wm:02d}", "kept": minute >= wm})
R["watermark"] = wm_rows

# ---------------------------------------------------------------- 5. windowed features
def windows(rs):
    by = defaultdict(list)
    for r in rs:
        t = r["event_time"]
        w = t.replace(minute=(t.minute // WIN) * WIN, second=0, microsecond=0)
        by[(r["entity_id"], w)].append(r)
    return by


feat = {}
for (e, w), rs in windows(rows).items():
    vals = {k: [r["metrics"][k] for r in rs] for k in METRICS}
    feat[(e, w)] = {"mean": {k: float(np.mean(v)) for k, v in vals.items()},
                    "max": {k: float(np.max(v)) for k, v in vals.items()},
                    "std": {k: float(np.std(v)) for k, v in vals.items()},
                    "slope": {k: (v[-1] - v[0]) / max(len(v) - 1, 1) for k, v in vals.items()},
                    "label": int(any(r["label"] == 1 for r in rs)) if rs[0]["attrs"]["split"] == "test" else None,
                    "split": rs[0]["attrs"]["split"], "n": len(rs)}
keys = sorted(feat)
train_keys = [k for k in keys if feat[k]["split"] == "train"]
test_keys = [k for k in keys if feat[k]["split"] == "test"]
show_w = BASE + timedelta(minutes=TRAIN_MIN + 20)
f = feat[("machine-1-2", show_w)]
R["features"] = {"window": "machine-1-2 · " + iso(show_w), "values": [
    {"metric": f"{k} ({NAMES[k]})", "mean": round(f["mean"][k], 2), "max": round(f["max"][k], 2),
     "std": round(f["std"][k], 2), "slope": round(f["slope"][k], 2), "n": f["n"]} for k in METRICS],
    "train_windows": len(train_keys), "test_windows": len(test_keys)}

X = lambda ks: np.array([[feat[k]["mean"][m] for m in METRICS] for k in ks])
Xtr, Xte = X(train_keys), X(test_keys)

# ---------------------------------------------------------------- 6. scoring ladder
# Tier 1: robust z per machine-metric
base = {}
for e in MACHINES:
    a = np.array([[feat[k]["mean"][m] for m in METRICS] for k in train_keys if k[0] == e])
    med = np.median(a, axis=0)
    mad = np.median(np.abs(a - med), axis=0)
    base[e] = (med, 1.4826 * mad)
z = np.array([np.abs((Xte[i] - base[k[0]][0]) / base[k[0]][1]) for i, k in enumerate(test_keys)])
t1 = z.max(axis=1)

mu, sd = Xtr.mean(0), Xtr.std(0)
Str, Ste = (Xtr - mu) / sd, (Xte - mu) / sd


def thr(train_scores):
    return float(np.percentile(train_scores, 99.5) * 1.1)


km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Str)
d_tr = km.transform(Str).min(1)
d_te = km.transform(Ste).min(1)
pca = PCA(n_components=1).fit(Str)
p_tr = ((Str - pca.inverse_transform(pca.transform(Str))) ** 2).sum(1)
p_te = ((Ste - pca.inverse_transform(pca.transform(Ste))) ** 2).sum(1)
ae = MLPRegressor(hidden_layer_sizes=(16, 1, 16), activation="tanh", max_iter=4000,
                  random_state=0, tol=1e-6).fit(Str, Str)
err_tr = (Str - ae.predict(Str)) ** 2
err_te = (Ste - ae.predict(Ste)) ** 2
a_tr, a_te = err_tr.sum(1), err_te.sum(1)

T = {"tier1": 3.0, "kmeans": thr(d_tr), "pca": thr(p_tr), "ae": thr(a_tr)}
scores = []
for i, k in enumerate(test_keys):
    s = {"entity_id": k[0], "window_start": k[1], "tier1_score": float(t1[i]), "tier1_flag": bool(t1[i] > T["tier1"]),
         "tier2_score": float(d_te[i]), "tier2_flag": bool(d_te[i] > T["kmeans"]),
         "pca_score": float(p_te[i]), "pca_flag": bool(p_te[i] > T["pca"]),
         "ae_score": float(a_te[i]), "ae_flag": bool(a_te[i] > T["ae"]), "label": feat[k]["label"],
         "top_metrics": [METRICS[j] for j in np.argsort(-err_te[i])[:3]]}
    s["is_anomaly"] = s["ae_flag"] or s["tier1_flag"]
    scores.append(s)

pick = next(s for s in scores if s["entity_id"] == "machine-1-2" and s["window_start"] == show_w)
expected_net = 100 * (f["mean"]["m00"] / 100) ** 2
R["ladder"] = {"window": "machine-1-2 · " + iso(show_w),
               "cpu": round(f["mean"]["m00"], 1), "net": round(f["mean"]["m01"], 1), "expected_net": round(expected_net, 1),
               "rows": [{"tier": "Tier 1 · robust z-score", "score": round(pick["tier1_score"], 2), "threshold": T["tier1"], "flag": pick["tier1_flag"]},
                        {"tier": "Tier 2 · KMeans", "score": round(pick["tier2_score"], 2), "threshold": round(T["kmeans"], 2), "flag": pick["tier2_flag"]},
                        {"tier": "Tier 2b · PCA", "score": round(pick["pca_score"], 2), "threshold": round(T["pca"], 2), "flag": pick["pca_flag"]},
                        {"tier": "Tier 3 · Autoencoder", "score": round(pick["ae_score"], 2), "threshold": round(T["ae"], 2), "flag": pick["ae_flag"]}],
               "top_metrics": [f"{m} ({NAMES[m]})" for m in pick["top_metrics"]], "is_anomaly": pick["is_anomaly"]}


# ---------------------------------------------------------------- evaluation (same rules as tools/check_eval.py)
def segs(y):
    out, s = [], None
    for i, v in enumerate(list(y) + [0]):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def prf(t, p):
    tp = sum(a and b for a, b in zip(t, p))
    fp = sum((not a) and b for a, b in zip(t, p))
    fn = sum(a and not b for a, b in zip(t, p))
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def event_prf(t, p):
    p = list(p)
    for s, e in segs(t):
        if any(p[s:e + 1]):
            p[s:e + 1] = [1] * (e - s + 1)
    return prf(t, p)


ev = []
for name, key in [("Tier 1 · z-score", "tier1_flag"), ("Tier 2 · KMeans", "tier2_flag"),
                  ("Tier 2b · PCA", "pca_flag"), ("Tier 3 · Autoencoder", "ae_flag"), ("Final rule (AE OR Tier 1)", "is_anomaly")]:
    tot_t, tot_p = [], []
    for e in MACHINES:
        ss = [s for s in scores if s["entity_id"] == e]
        tot_t += [s["label"] for s in ss] + [0]
        tot_p += [int(s[key]) for s in ss] + [0]
    pr, rc, f1 = event_prf(tot_t, tot_p)
    ev.append({"detector": name, "precision": round(pr, 2), "recall": round(rc, 2), "f1": round(f1, 2)})
R["eval"] = ev

# ---------------------------------------------------------------- 7. incidents
inc, cur = [], None
for s in sorted(scores, key=lambda s: (s["entity_id"], s["window_start"])) + [None]:
    if s and s["is_anomaly"] and cur and cur["entity_id"] == s["entity_id"] and \
            s["window_start"] - cur["end_time"] <= timedelta(minutes=WIN):
        cur["end_time"] = s["window_start"] + timedelta(minutes=WIN)
        cur["n_windows"] += 1
        cur["max_score"] = max(cur["max_score"], s["ae_score"])
        cur["tm"] += s["top_metrics"]
        continue
    if cur:
        cnt = defaultdict(int)
        for m in cur.pop("tm"):
            cnt[m] += 1
        cur["top_metrics"] = [f"{m} ({NAMES[m]})" for m, _ in sorted(cnt.items(), key=lambda x: -x[1])[:3]]
        cur["incident_id"] = f"{cur['entity_id']}_{cur['start_time'].strftime('%Y%m%d%H%M')}"
        inc.append(cur)
        cur = None
    if s and s["is_anomaly"]:
        cur = {"entity_id": s["entity_id"], "start_time": s["window_start"], "end_time": s["window_start"] + timedelta(minutes=WIN),
               "n_windows": 1, "max_score": s["ae_score"], "tm": list(s["top_metrics"])}
alerts = sum(s["is_anomaly"] for s in scores)
R["incidents"] = {"alerts": alerts, "incidents": len(inc), "reduction_pct": round(100 * (1 - len(inc) / alerts)) if alerts else 0,
                  "rows": [{"incident_id": i["incident_id"], "entity_id": i["entity_id"],
                            "start": i["start_time"].strftime("%H:%M"), "end": i["end_time"].strftime("%H:%M"),
                            "n_windows": i["n_windows"], "max_score": round(i["max_score"], 1), "top_metrics": i["top_metrics"]} for i in inc]}
R["timeline"] = [{"entity_id": s["entity_id"], "t": s["window_start"].strftime("%H:%M"),
                  "label": s["label"], "is_anomaly": s["is_anomaly"], "ae": round(s["ae_score"], 2), "t1": round(s["tier1_score"], 2)}
                 for s in sorted(scores, key=lambda s: (s["entity_id"], s["window_start"]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(R, fh, indent=1, default=str)
    print("1. Sources:", R["sources"])
    print("2. Adapter, one record:", json.dumps(R["adapter"], default=str))
    print("3. Kafka partition per machine (murmur2 % 4):", R["kafka"]["assignment"])
    print("4. Watermark:", R["watermark"])
    print("5. Features", R["features"]["window"])
    for v in R["features"]["values"]:
        print("   ", v)
    print("6. Ladder on", R["ladder"]["window"], f"cpu={R['ladder']['cpu']} net={R['ladder']['net']} expected net~{R['ladder']['expected_net']}")
    for v in R["ladder"]["rows"]:
        print("   ", v)
    print("   top_metrics:", R["ladder"]["top_metrics"], "is_anomaly:", R["ladder"]["is_anomaly"])
    print("   Event-level F1 on the toy test set:")
    for v in R["eval"]:
        print("   ", v)
    print("7. Incidents:", R["incidents"]["alerts"], "alerts ->", R["incidents"]["incidents"], "incidents")
    for v in R["incidents"]["rows"]:
        print("   ", v)


if __name__ == "__main__":
    main()
