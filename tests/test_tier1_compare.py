"""experiments/tier1_compare.py: parity numbers and Tier 1 on/off metrics on a tiny hand-made example."""
import numpy as np
import pandas as pd
import pytest

from experiments import tier1_compare as M

T0 = pd.Timestamp("2026-01-01", tz="UTC")


def _frames():
    # 2 machines x 6 test windows; one fault per machine (windows 2-3 on a, window 4 on b)
    ws = [T0 + pd.Timedelta(minutes=5 * i) for i in range(6)]
    keys = pd.DataFrame({"entity_id": ["a"] * 6 + ["b"] * 6, "window_start": ws * 2})
    label = [0, 0, 1, 1, 0, 0] + [0, 0, 0, 0, 1, 0]
    ae_flag = [False, False, False, True, False, False] + [False] * 6          # AE catches a's fault late
    t1_flag = [True, False, False, False, False, False] + [False] * 4 + [True, False]   # 1 FA on a, catches b
    t1_score = np.linspace(0.0, 5.0, 12)
    ae_score = np.linspace(1.0, 2.0, 12)
    ae = keys.assign(ae_score=ae_score, ae_flag=ae_flag, label=label)
    tier1 = keys.assign(tier1_score=t1_score, tier1_flag=t1_flag)
    stream = keys.assign(tier1_score=t1_score + 1e-9, tier1_flag=t1_flag, ae_score=ae_score,
                         ae_flag=pd.array(ae_flag, dtype="boolean"), is_anomaly=ae_flag)
    stream = pd.concat([stream.iloc[:-1], stream.iloc[[0]]], ignore_index=True)   # last window missing, 1 duplicate
    return stream, tier1, ae


def test_compare():
    stream, tier1, ae = _frames()
    r = M.compare(stream, tier1, ae)
    assert (r["duplicates"], r["batch_windows"], r["matched"]) == (1, 12, 11)
    assert r["max_tier1_diff"] == pytest.approx(1e-9, abs=1e-12)
    assert r["max_ae_diff"] == 0.0
    assert r["tier1_flag_identical"] == 1.0 and r["ae_flag_identical"] == 1.0
    assert r["is_anomaly_is_ae_flag"]
    assert (r["tier1_adds"], r["tier1_adds_on_anomalies"]) == (2, 1)
    off, on = r["rows"]["Tier 1 off (ae_flag)"], r["rows"]["Tier 1 on (ae_flag OR tier1_flag)"]
    assert off["faults_caught"] == "1/2" and on["faults_caught"] == "2/2"
    assert off["median_delay_min"] == 5.0                     # a's fault caught one window late
    assert on["median_delay_min"] == 2.5                       # delays 1 and 0 windows
    assert off["false_alarms_per_machine_day"] == 0.0 and on["false_alarms_per_machine_day"] > 0
    assert (off["flagged"], on["flagged"]) == (1, 3)


def test_abs_diff_nulls():
    d = M.abs_diff(pd.Series([1.0, np.nan, np.nan]), pd.Series([1.5, np.nan, 2.0]))
    assert d.tolist() == [0.5, 0.0, np.inf]
