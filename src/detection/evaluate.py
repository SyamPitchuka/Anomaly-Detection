"""Evaluation metrics for anomaly detectors (WP-3).

Inputs are equal-length lists or numpy arrays of 0/1, one entry per time point, in time order.
The expected answers are pinned by tools/check_eval.py.
"""
import numpy as np


def _as_binary(y_true, y_pred):
    t = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_pred, dtype=int).ravel()
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred differ in length: {t.size} vs {p.size}")
    return t != 0, p != 0


def _segments(t):
    """(start, end) index pairs, end exclusive, of each run of consecutive true points."""
    edges = np.diff(np.concatenate(([0], t.astype(int), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _prf(t, p):
    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def point_metrics(y_true, y_pred):
    """Point-wise precision, recall and F1. Any undefined ratio (zero denominator) is 0.0."""
    t, p = _as_binary(y_true, y_pred)
    return _prf(t, p)


def event_metrics(y_true, y_pred):
    """Point-adjusted precision, recall and F1.

    If any point of a true anomaly segment is predicted, every point of that segment counts as
    predicted; false positives outside segments are unchanged. Metrics are then computed point-wise.
    """
    t, p = _as_binary(y_true, y_pred)
    adjusted = p.copy()
    for start, end in _segments(t):
        if p[start:end].any():
            adjusted[start:end] = True
    return _prf(t, adjusted)


def detection_delay(y_true, y_pred):
    """Delay, in points, from each true segment's start to its first predicted point.

    mean_delay averages over detected segments only; it is nan when no segment is detected.
    """
    t, p = _as_binary(y_true, y_pred)
    segments = _segments(t)
    delays = []
    for start, end in segments:
        hits = np.flatnonzero(p[start:end])
        if hits.size:
            delays.append(int(hits[0]))
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    return {"mean_delay": mean_delay, "n_detected": len(delays), "n_events": len(segments)}
