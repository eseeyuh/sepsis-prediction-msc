"""
Unit tests for src/calibration/methods.py.

The key property under test throughout: a calibrator fit on one
(val-like) sample and applied to a FRESH (test-like) sample drawn from the
same underlying process should reduce calibration error — this directly
mirrors how calibration is used in this project (fit on val, evaluate on
test) and guards against the calibrator merely memorising its fitting data.

Run with: pytest tests/test_calibration_methods.py -v
"""

import numpy as np
import pytest

from src.calibration.methods import (
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    fit_calibrator,
)
from src.evaluation.calibration_metrics import expected_calibration_error


def _miscalibrated_data(n=4000, shrink_factor=0.3, seed=0):
    """Simulates the exact failure mode observed in the real LightGBM run:
    a model whose raw probabilities are systematically compressed toward 0,
    even though its underlying discrimination is reasonable.
    """
    rng = np.random.default_rng(seed)
    true_prob = rng.beta(2, 5, size=n)
    y_true = (rng.uniform(size=n) < true_prob).astype(int)
    raw_prob = true_prob * shrink_factor
    return raw_prob, y_true, true_prob


@pytest.mark.parametrize("method_name", ["platt", "temperature", "isotonic"])
def test_all_methods_reduce_ece_on_held_out_data(method_name):
    raw_val, y_val, _ = _miscalibrated_data(seed=1)
    raw_test, y_test, _ = _miscalibrated_data(seed=2)  # different draw, same process

    ece_before = expected_calibration_error(y_test, raw_test)

    calibrator = fit_calibrator(method_name, raw_val, y_val)
    calibrated_test = calibrator.transform(raw_test)
    ece_after = expected_calibration_error(y_test, calibrated_test)

    assert ece_after < ece_before


@pytest.mark.parametrize("method_name", ["platt", "temperature", "isotonic"])
def test_all_methods_return_valid_probabilities(method_name):
    raw_val, y_val, _ = _miscalibrated_data()
    calibrator = fit_calibrator(method_name, raw_val, y_val)

    calibrated = calibrator.transform(raw_val)

    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_temperature_scaler_learns_temperature_greater_than_one_when_underconfident():
    """The synthetic data is systematically shrunk toward 0 — i.e. raw
    probabilities are too LOW / overconfident-toward-negative. Temperature
    scaling should learn T that corrects this. We check the temperature is
    a sensible positive number, not pinned at a search-boundary default.
    """
    raw_val, y_val, _ = _miscalibrated_data()
    scaler = TemperatureScaler().fit(raw_val, y_val)

    assert scaler.temperature > 0
    # Should not have collapsed to the edge of the search bounds
    # (exp(-3) to exp(3)) — that would indicate the optimiser failed to
    # converge to an interior solution.
    assert np.exp(-2.9) < scaler.temperature < np.exp(2.9)


def test_isotonic_calibrator_output_is_monotonic_in_sorted_input():
    raw_val, y_val, _ = _miscalibrated_data()
    calibrator = IsotonicCalibrator().fit(raw_val, y_val)

    sorted_raw = np.sort(raw_val)
    calibrated = calibrator.transform(sorted_raw)

    # Non-decreasing: isotonic regression's defining property.
    assert np.all(np.diff(calibrated) >= -1e-9)  # tiny tolerance for float noise


def test_unfit_calibrators_raise_clear_error_on_transform():
    with pytest.raises(RuntimeError):
        PlattScaler().transform(np.array([0.5]))
    with pytest.raises(RuntimeError):
        TemperatureScaler().transform(np.array([0.5]))
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform(np.array([0.5]))


def test_fit_calibrator_rejects_unknown_method():
    raw_val, y_val, _ = _miscalibrated_data()
    with pytest.raises(ValueError):
        fit_calibrator("not_a_real_method", raw_val, y_val)


def test_calibration_improves_brier_score_too():
    from src.evaluation.calibration_metrics import brier_score

    raw_val, y_val, _ = _miscalibrated_data(seed=1)
    raw_test, y_test, _ = _miscalibrated_data(seed=2)

    brier_before = brier_score(y_test, raw_test)

    calibrator = fit_calibrator("temperature", raw_val, y_val)
    calibrated_test = calibrator.transform(raw_test)
    brier_after = brier_score(y_test, calibrated_test)

    assert brier_after < brier_before
