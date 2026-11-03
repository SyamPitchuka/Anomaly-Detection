"""E1 report on a tiny synthetic scores table (2 machines x 20 windows)."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from contracts import schemas as S
from src.detection import e1_report as E
from src.detection.evaluate import (
    composite_metrics, detection_delays, event_metrics, false_alarms_per_day, point_metrics, pr_auc,
)

PCA_THR = 0.5
N = 20


def _machine(entity, labels, tier1, tier2, pca, ae, start):
    t0 = datetime(2026, 1, 20, tzinfo=timezone.utc) + timedelta(minutes=start)
    ws = [t0 + timedelta(minutes=5 * i) for i in range(N)]
    rng = np.random.default_rng(len(entity) + start)
    ae_flag = np.array(ae, bool)
    return pd.DataFrame({
        "entity_id": entity, "window_start": ws, "window_end": [w + timedelta(minutes=5) for w in ws],
        "tier1_score": rng.random(N), "tier1_flag": np.array(tier1, bool),
        "tier2_score": rng.random(N), "tier2_flag": np.array(tier2, bool),
        "pca_score": np.array(pca, float), "ae_score": rng.random(N) + ae_flag,
        "ae_flag": ae_flag, "is_anomaly": ae_flag.copy(), "top_metrics": [["m00", "m01", "m02"]] * N,
        "label": np.array(labels, np.int32),
    })


def _bits(*idx):
    a = np.zeros(N, int)
    a[list(idx)] = 1
    return a


def synthetic() -> pd.DataFrame:
    # machine a: faults at 3-5 and 12-13; machine b: fault at 0-1 (starts at the machine's first window)
    la, lb = _bits(3, 4, 5, 12, 13), _bits(0, 1)
    a = _machine("machine-a", la, tier1=np.ones(N), tier2=_bits(4, 9), pca=_bits(5, 6, 13) * 0.9 + 0.1,
                 ae=_bits(3, 12, 17), start=0)
    b = _machine("machine-b", lb, tier1=_bits(5), tier2=_bits(1), pca=_bits(0) * 0.9 + 0.1,
                 ae=_bits(1, 2), start=7)
    return pd.concat([a, b], ignore_index=True)


def write_layer(root, df):
    folder = root / S.PATHS["scores"] / "source=smd"
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(df, schema=S.arrow_schema("scores"), preserve_index=False),
                   folder / "part-0.parquet")
    (root / "models/pca_smd").mkdir(parents=True)
    (root / "models/pca_smd/thresholds.json").write_text(json.dumps({"mode": "global", "global": PCA_THR}))


def expected(df, flag, score):
    """The autoencoder.py scoring rule written out with evaluate.py only."""
    all_t, all_p, delays = [], [], []
    for _, g in df.assign(f=flag.astype(int)).sort_values(["entity_id", "window_start"]).groupby("entity_id"):
        t, p = g["label"].to_numpy(), g["f"].to_numpy()
        all_t += [0] + t.tolist()
        all_p += [0] + p.tolist()
        delays += detection_delays(t, p)
    t, p = df["label"].to_numpy(), flag.astype(int)
    return {"composite": composite_metrics(all_t, all_p), "point": point_metrics(all_t, all_p),
            "event": event_metrics(all_t, all_p), "fa_day": false_alarms_per_day(t, p, 5),
            "delay_min": float(np.median(delays)) * 5 if delays else float("nan"),
            "pr_auc": pr_auc(t, score)}


def test_numbers_match_evaluate(tmp_path):
    df = synthetic()
    write_layer(tmp_path, df)
    out = E.run(root=tmp_path, out_dir=tmp_path / "out", expected_windows=2 * N, expected_machines=2,
                verbose=False)
    rows = {r["key"]: r for r in out["rows"]}
    assert list(rows) == ["tier1", "kmeans", "pca", "ae_v2", "is_anomaly"]
    srt = df.sort_values(["entity_id", "window_start"]).reset_index(drop=True)
    cases = {"tier1": (srt["tier1_flag"], "tier1_score"), "kmeans": (srt["tier2_flag"], "tier2_score"),
             "pca": (srt["pca_score"] > PCA_THR, "pca_score"), "ae_v2": (srt["ae_flag"], "ae_score"),
             "is_anomaly": (srt["is_anomaly"], "ae_score")}
    for key, (flag, score_col) in cases.items():
        exp, r = expected(srt, flag.to_numpy(bool), srt[score_col].to_numpy()), rows[key]
        c = exp["composite"]
        assert r["faults"] == 3 and r["faults_caught"] == c["events_detected"], key
        assert r["composite_f1"] == pytest.approx(c["composite_f1"]), key
        assert r["alarm_precision"] == pytest.approx(c["alarm_precision"]), key
        assert r["false_alarms_per_machine_day"] == pytest.approx(exp["fa_day"]), key
        assert r["pr_auc"] == pytest.approx(exp["pr_auc"]), key
        assert r["point_f1"] == pytest.approx(exp["point"]["f1"]), key
        assert r["event_f1"] == pytest.approx(exp["event"]["f1"]), key
        assert r["windows_flagged"] == int(flag.sum()), key
        if np.isnan(exp["delay_min"]):
            assert np.isnan(r["median_delay_min"]), key
        else:
            assert r["median_delay_min"] == pytest.approx(exp["delay_min"]), key
    # hand-checked: AE catches all 3 faults (b's fault at its 2nd window), flags 5 windows, 3 of them anomalous
    assert rows["ae_v2"]["faults_caught"] == 3
    assert rows["ae_v2"]["alarm_precision"] == pytest.approx(3 / 5)
    assert rows["ae_v2"] == {**rows["is_anomaly"], "detector": rows["ae_v2"]["detector"], "key": "ae_v2"}
    # PCA flag = score > threshold: 4 windows above 0.5
    assert rows["pca"]["windows_flagged"] == 4
    pm = out["per_machine"]
    assert list(pm["machine"]) == ["machine-a", "machine-b"] and pm.shape == (2, 6)
    for name in ("e1.md", "e1.json", "e1.png"):
        assert (tmp_path / "out" / name).stat().st_size > 0
    doc = json.loads((tmp_path / "out/e1.json").read_text())
    assert doc["test_windows"] == 2 * N and len(doc["detectors"]) == 5


@pytest.mark.parametrize("breaker, message", [
    (lambda d: d.iloc[:-1], "39 test windows"),
    (lambda d: d.assign(entity_id="machine-a", window_start=pd.date_range(
        "2026-01-20", periods=len(d), freq="5min", tz="UTC")), "1 machines"),
    (lambda d: d.assign(tier2_score=np.nan), "tier2_score is null"),
    (lambda d: d.assign(pca_score=np.nan), "pca_score is null"),
    (lambda d: d.assign(is_anomaly=d["ae_flag"] | d["tier1_flag"]), "is_anomaly differs from ae_flag"),
])
def test_sanity_checks_fail_on_bad_input(breaker, message):
    good = synthetic()
    E.sanity_checks(good, 2 * N, 2, verbose=False)
    with pytest.raises(E.SanityError, match=message):
        E.sanity_checks(breaker(good), 2 * N, 2, verbose=False)


def test_missing_layer_is_a_clear_error(tmp_path):
    with pytest.raises(E.SanityError, match="does not exist"):
        E.run(root=tmp_path, out_dir=tmp_path / "out", verbose=False)
