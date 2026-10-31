import math

import numpy as np
import pytest

from src.detection.evaluate import (
    composite_metrics,
    detection_delay,
    detection_delays,
    event_metrics,
    false_alarms_per_day,
    point_metrics,
    pr_auc,
)

# Same fixture as tools/check_eval.py: true segments at [2, 4] and [7, 8].
Y_TRUE = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
Y_PRED = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]


def test_point_metrics_known_answer():
    got = point_metrics(Y_TRUE, Y_PRED)
    assert got["precision"] == pytest.approx(0.5)
    assert got["recall"] == pytest.approx(0.2)
    assert got["f1"] == pytest.approx(2 * 0.5 * 0.2 / 0.7)


def test_event_metrics_point_adjusted():
    # The hit at index 3 marks all of segment [2, 4]: tp=3, fp=1, fn=2.
    got = event_metrics(Y_TRUE, Y_PRED)
    assert got["precision"] == pytest.approx(0.75)
    assert got["recall"] == pytest.approx(0.6)
    assert got["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_detection_delay_known_answer():
    assert detection_delay(Y_TRUE, Y_PRED) == {"mean_delay": 1.0, "n_detected": 1, "n_events": 2}


def test_numpy_input_matches_list_input():
    assert point_metrics(np.array(Y_TRUE), np.array(Y_PRED)) == point_metrics(Y_TRUE, Y_PRED)
    assert event_metrics(np.array(Y_TRUE), np.array(Y_PRED)) == event_metrics(Y_TRUE, Y_PRED)


def test_no_predictions_gives_zero_without_crashing():
    y_true, y_pred = [0, 0, 1, 1], [0, 0, 0, 0]
    for fn in (point_metrics, event_metrics):
        assert fn(y_true, y_pred) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    got = detection_delay(y_true, y_pred)
    assert math.isnan(got["mean_delay"])
    assert (got["n_detected"], got["n_events"]) == (0, 1)


def test_no_true_anomalies():
    y_true, y_pred = [0, 0, 0], [0, 1, 0]
    assert point_metrics(y_true, y_pred) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    got = detection_delay(y_true, y_pred)
    assert (got["n_detected"], got["n_events"]) == (0, 0)


def test_perfect_prediction():
    y = [1, 1, 0, 0, 1]
    assert point_metrics(y, y) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert event_metrics(y, y) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert detection_delay(y, y) == {"mean_delay": 0.0, "n_detected": 2, "n_events": 2}


def test_segments_at_array_edges():
    # Segments touch both ends; late hit in the first, early hit in the last.
    y_true = [1, 1, 1, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 0, 0, 1, 0]
    assert event_metrics(y_true, y_pred)["recall"] == pytest.approx(1.0)
    assert detection_delay(y_true, y_pred) == {"mean_delay": 1.0, "n_detected": 2, "n_events": 2}


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        point_metrics([0, 1], [0, 1, 0])


def test_composite_metrics_known_answer():
    # One of two segments hit (index 3); one of two flags is a false alarm (index 1).
    assert composite_metrics(Y_TRUE, Y_PRED) == {
        "events": 2,
        "events_detected": 1,
        "event_recall": 0.5,
        "alarm_precision": 0.5,
        "composite_f1": 0.5,
    }


def test_composite_one_flag_per_segment_is_perfect():
    y_true = [0, 1, 1, 1, 0, 0, 1, 1, 0]
    y_pred = [0, 0, 1, 0, 0, 0, 0, 1, 0]
    got = composite_metrics(y_true, y_pred)
    assert got["event_recall"] == 1.0
    assert got["alarm_precision"] == 1.0
    assert got["composite_f1"] == 1.0


def test_composite_zero_when_nothing_right():
    assert composite_metrics([0, 1, 1, 0], [1, 0, 0, 0])["composite_f1"] == 0.0
    assert composite_metrics([0, 0, 0], [0, 0, 0])["composite_f1"] == 0.0


def test_false_alarms_punish_composite_more_than_event_f1():
    # 1000 points, 4 segments of 50 points; one flag per segment plus 20 false alarms.
    y_true = np.zeros(1000, dtype=int)
    starts = [100, 300, 500, 700]
    for s in starts:
        y_true[s:s + 50] = 1
    clean = np.zeros(1000, dtype=int)
    clean[starts] = 1
    noisy = clean.copy()
    normal = np.flatnonzero(y_true == 0)
    noisy[np.random.default_rng(0).choice(normal, size=20, replace=False)] = 1

    assert composite_metrics(y_true, clean)["composite_f1"] == 1.0
    assert event_metrics(y_true, clean)["f1"] == 1.0
    # Event F1: tp=200, fp=20 -> P=10/11, R=1 -> F1=20/21. Composite: P=4/24, R=1 -> F1=2/7.
    event_f1 = event_metrics(y_true, noisy)["f1"]
    composite_f1 = composite_metrics(y_true, noisy)["composite_f1"]
    assert event_f1 == pytest.approx(20 / 21)
    assert composite_f1 == pytest.approx(2 / 7)
    assert 1.0 - event_f1 < 0.05
    assert 1.0 - composite_f1 > 0.7


def test_false_alarms_per_day_known_answer():
    # 1 false positive over 10 points x 5 min = 50 min -> 1440 / 50 = 28.8 per day.
    assert false_alarms_per_day(Y_TRUE, Y_PRED) == pytest.approx(28.8)
    # One day of 1-minute points with 3 false positives.
    y_true = np.zeros(1440, dtype=int)
    y_pred = y_true.copy()
    y_pred[[10, 20, 30]] = 1
    assert false_alarms_per_day(y_true, y_pred, window_minutes=1) == pytest.approx(3.0)
    assert math.isnan(false_alarms_per_day([], []))


def test_pr_auc_known_answer():
    # Ranked scores: pos, neg, pos -> AP = 1/2 * 1 + 1/2 * 2/3 = 5/6.
    assert pr_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(5 / 6)
    assert pr_auc([0, 1, 1], [0.1, 0.5, 0.9]) == pytest.approx(1.0)


def test_pr_auc_nan_without_positives():
    assert math.isnan(pr_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_detection_delays_lists_detected_segments_only():
    assert detection_delays(Y_TRUE, Y_PRED) == [1]
    assert detection_delays([1, 1, 0, 1, 1, 1], [0, 1, 0, 0, 0, 1]) == [1, 2]
    assert detection_delays([0, 1], [0, 0]) == []
