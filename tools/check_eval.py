"""Inspection gauge for WP-3 evaluation code: src/detection/evaluate.py must give these exact answers.

Run from the repo root:  python tools/check_eval.py

Required functions (inputs are equal-length lists or numpy arrays of 0/1):
    point_metrics(y_true, y_pred)  -> {"precision", "recall", "f1"}
    event_metrics(y_true, y_pred)  -> {"precision", "recall", "f1"}   point-adjusted:
        if any point of a true anomaly segment is predicted, the whole segment counts as detected
    detection_delay(y_true, y_pred) -> {"mean_delay", "n_detected", "n_events"}
        delay = first predicted index inside a segment minus the segment start, in points
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.detection.evaluate import detection_delay, event_metrics, point_metrics
except ImportError as e:
    print(f"FAIL cannot import from src/detection/evaluate.py: {e}")
    sys.exit(1)

Y_TRUE = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
Y_PRED = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
CASES = [
    ("point_metrics", point_metrics, {"precision": 0.5, "recall": 0.2, "f1": 0.2857}),
    ("event_metrics", event_metrics, {"precision": 0.75, "recall": 0.6, "f1": 0.6667}),
    ("detection_delay", detection_delay, {"mean_delay": 1.0, "n_detected": 1, "n_events": 2}),
]

fails = 0
for name, fn, want in CASES:
    got = fn(Y_TRUE, Y_PRED)
    for k, v in want.items():
        ok = k in got and abs(float(got[k]) - v) < 1e-3
        fails += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}[{k}] = {got.get(k)} (want {v})")

zero = point_metrics([0, 0, 1], [0, 0, 0])
ok = float(zero["f1"]) == 0.0
fails += not ok
print(f"{'OK  ' if ok else 'FAIL'} no predictions gives f1 = 0 without crashing")
print("PASS" if not fails else "FAIL")
sys.exit(0 if not fails else 1)
