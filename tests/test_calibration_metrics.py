"""
Unit tests for src/evaluation/calibration_metrics.py.

Run with: pytest tests/test_calibration_metrics.py -v
"""

import numpy as np
import pytest

from src.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_data,
)


def test_brier_score_is_zero_for_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_is_worst_case_for_confidently_wrong_predictions():
    y_true = np.array([0, 1])
    y_prob = np.array([1.0, 0.0])  # maximally confident and maximally wrong
    assert brier_score(y_true, y_prob) == pytest.approx(1.0)


def test_reliability_diagram_bins_cover_all_data_points():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(size=1000)
    y_true = (rng.uniform(size=1000) < y_prob).astype(int)

    bins = reliability_diagram_data(y_true, y_prob, n_bins=10)

    assert sum(b.count for b in bins) == 1000


def test_reliability_diagram_empty_bin_has_nan_means_not_dropped():
    # All probabilities clustered near 0 — bins near 1.0 should be empty,
    # not silently absent from the returned list.
    y_prob = np.array([0.05, 0.06, 0.07, 0.04])
    y_true = np.array([0, 0, 1, 0])

    bins = reliability_diagram_data(y_true, y_prob, n_bins=10)

    assert len(bins) == 10  # all bins present, even empty ones
    high_bins = [b for b in bins if b.bin_lower >= 0.9]
    assert all(b.count == 0 for b in high_bins)
    assert all(np.isnan(b.mean_predicted) for b in high_bins)


def test_ece_is_near_zero_for_well_calibrated_predictions():
    """Construct predictions that ARE the true underlying probability, so
    the model is calibrated by construction — ECE should be small (not
    exactly 0, since it's still measured on a finite random sample).
    """
    rng = np.random.default_rng(0)
    n = 20000
    y_prob = rng.uniform(0, 1, size=n)
    y_true = (rng.uniform(size=n) < y_prob).astype(int)

    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.03


def test_ece_is_large_for_badly_calibrated_predictions():
    """Predictions systematically shrunk toward 0 relative to the true
    probability — the same failure mode observed in the real LightGBM run.
    """
    rng = np.random.default_rng(0)
    n = 20000
    true_prob = rng.uniform(0, 1, size=n)
    y_true = (rng.uniform(size=n) < true_prob).astype(int)
    raw_prob = true_prob * 0.3  # systematically too low

    ece = expected_calibration_error(y_true, raw_prob, n_bins=10)
    assert ece > 0.1


def test_ece_matches_hand_computed_value_on_a_tiny_example():
    """A fully worked, hand-checkable example: 4 points, 2 bins."""
    # Bin [0, 0.5): predictions 0.1, 0.2 -> mean predicted 0.15, labels 0,0 -> mean observed 0.0
    # Bin [0.5, 1.0]: predictions 0.8, 0.9 -> mean predicted 0.85, labels 1,1 -> mean observed 1.0
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 0, 1, 1])

    ece = expected_calibration_error(y_true, y_prob, n_bins=2)

    # weight 0.5 each: 0.5*|0.15-0.0| + 0.5*|0.85-1.0| = 0.5*0.15 + 0.5*0.15 = 0.15
    assert ece == pytest.approx(0.15, abs=1e-9)
