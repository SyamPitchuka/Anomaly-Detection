"""Inspection gauge for WP-4: the dashboard data layer reads the fake scores and incidents correctly.

Run from the repo root after python tools/fake_data.py:  python tools/check_dashboard.py

Required in src/dashboard/data.py:
    load_scores(source="smd")    -> pandas DataFrame with all scores columns from contracts/schemas.py
    load_incidents(source="smd") -> pandas DataFrame with all incidents columns
Expected on the fake data: 36 score rows, 3 of them anomalous, and exactly 1 incident
(machine-1-2, 3 windows, incident_id machine-1-2_202601010020).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts import schemas as S  # noqa: E402

try:
    from src.dashboard.data import load_incidents, load_scores
except ImportError as e:
    print(f"FAIL cannot import from src/dashboard/data.py: {e}")
    sys.exit(1)

fails = 0


def check(ok, msg):
    global fails
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {msg}")


s, i = load_scores("smd"), load_incidents("smd")
check(set(S.LAYERS["scores"]) <= set(s.columns), "scores has every contract column")
check(set(S.LAYERS["incidents"]) <= set(i.columns), "incidents has every contract column")
check(len(s) == 36, f"36 score rows (found {len(s)})")
check(int(s["is_anomaly"].sum()) == 3, f"3 anomalous windows (found {int(s['is_anomaly'].sum())})")
check(len(i) == 1, f"1 incident (found {len(i)})")
if len(i) == 1:
    row = i.iloc[0]
    check(row["incident_id"] == "machine-1-2_202601010020", f"incident_id is {row['incident_id']}")
    check(int(row["n_windows"]) == 3, f"incident spans 3 windows (found {row['n_windows']})")
print("PASS" if not fails else "FAIL")
sys.exit(0 if not fails else 1)
