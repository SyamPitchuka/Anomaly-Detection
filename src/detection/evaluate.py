"""Evaluation metrics for anomaly detectors (WP-3).

Inputs are equal-length lists or numpy arrays of 0/1, one entry per time point, in time order.
The expected answers are pinned by tools/check_eval.py.
"""
import numpy as np
from sklearn.metrics import average_precision_score


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


def composite_metrics(y_true, y_pred):
    """Event-wise recall combined with point-wise precision.

    event_recall is the share of true segments with at least one predicted point; alarm_precision
    is point-wise precision, so every false alarm counts. composite_f1 is their harmonic mean.
    """
    t, p = _as_binary(y_true, y_pred)
    segments = _segments(t)
    detected = sum(1 for start, end in segments if p[start:end].any())
    event_recall = detected / len(segments) if segments else 0.0
    alarm_precision = _prf(t, p)["precision"]
    total = event_recall + alarm_precision
    composite_f1 = 2 * event_recall * alarm_precision / total if total else 0.0
    return {
        "events": len(segments),
        "events_detected": detected,
        "event_recall": event_recall,
        "alarm_precision": alarm_precision,
        "composite_f1": composite_f1,
    }


def false_alarms_per_day(y_true, y_pred, window_minutes=5):
    """False-positive points per day; each point covers window_minutes. nan for an empty sequence."""
    t, p = _as_binary(y_true, y_pred)
    days = t.size * window_minutes / (24 * 60)
    return int(np.sum(~t & p)) / days if days else float("nan")


def pr_auc(y_true, scores):
    """Area under the precision-recall curve (average precision); nan if there are no positives."""
    t = np.asarray(y_true, dtype=int).ravel() != 0
    s = np.asarray(scores, dtype=float).ravel()
    if t.shape != s.shape:
        raise ValueError(f"y_true and scores differ in length: {t.size} vs {s.size}")
    if not t.any():
        return float("nan")
    return float(average_precision_score(t, s))


def detection_delays(y_true, y_pred):
    """Delay, in points, from each detected true segment's start to its first predicted point."""
    t, p = _as_binary(y_true, y_pred)
    delays = []
    for start, end in _segments(t):
        hits = np.flatnonzero(p[start:end])
        if hits.size:
            delays.append(int(hits[0]))
    return delays
