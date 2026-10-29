import math

import numpy as np
import pytest

from src.detection.evaluate import detection_delay, event_metrics, point_metrics

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
