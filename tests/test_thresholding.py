"""Tests for Fatal-class threshold application and selection."""

import numpy as np
import pytest

from road_accidents.thresholding import (
    apply_fatal_threshold,
    search_fatal_thresholds,
    select_fatal_f1_threshold,
    select_macro_f1_threshold,
    threshold_metrics,
)


def test_threshold_rule_uses_fatal_at_equality_then_nonfatal_argmax():
    probabilities = np.array(
        [
            [0.30, 0.20, 0.50],
            [0.29, 0.60, 0.11],
            [0.10, 0.45, 0.45],
        ]
    )
    assert apply_fatal_threshold(probabilities, 0.30).tolist() == [0, 1, 1]


def test_threshold_metrics_include_requested_values():
    y_true = np.array([0, 0, 1, 2])
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.4, 0.5, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    result = threshold_metrics(y_true, probabilities, 0.5)
    assert result["fatal_precision"] == pytest.approx(0.5)
    assert result["fatal_recall"] == pytest.approx(0.5)
    assert result["fatal_f1"] == pytest.approx(0.5)
    assert result["predicted_fatal_proportion"] == pytest.approx(0.5)
    assert 0 <= result["macro_f1"] <= 1


def _result(threshold, macro, fatal, precision=0.1, proportion=0.1):
    return {
        "threshold": threshold,
        "macro_f1": macro,
        "fatal_f1": fatal,
        "fatal_precision": precision,
        "fatal_recall": 0.1,
        "predicted_fatal_proportion": proportion,
    }


def test_selection_uses_documented_metrics_and_tie_breakers():
    results = [
        _result(0.2, 0.40, 0.20, proportion=0.2),
        _result(0.3, 0.40, 0.25, proportion=0.3),
        _result(0.4, 0.39, 0.30, precision=0.2),
    ]
    assert select_macro_f1_threshold(results)["threshold"] == 0.3
    assert select_fatal_f1_threshold(results)["threshold"] == 0.4


def test_search_refines_around_both_coarse_optima():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.35, 0.40, 0.25],
            [0.30, 0.60, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
            [0.05, 0.10, 0.85],
        ]
    )
    search = search_fatal_thresholds(y_true, probabilities)
    assert search["grid"]["threshold_count"] > 101
    assert 0 <= search["selected_for_macro_f1"]["threshold"] <= 1
    assert 0 <= search["selected_for_fatal_f1"]["threshold"] <= 1
