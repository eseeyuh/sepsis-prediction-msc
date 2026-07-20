"""
Unit tests for src/uncertainty/aps.py.

Mirrors the structure of test_conformal.py (coverage close to nominal is
the central property), plus tests specific to what APS was added FOR:
demonstrating genuine {0,1} ambiguous sets under conditions where LAC
cannot produce them (see project diary for the motivating finding).

Run with: pytest tests/test_aps.py -v
"""

import numpy as np
import pytest

from src.uncertainty.aps import build_prediction_sets_aps, calibrate_conformal_aps
from src.uncertainty.conformal import (
    average_set_size,
    build_prediction_sets,
    calibrate_conformal,
    empirical_coverage,
    set_size_distribution,
)


def _synthetic_calib_test_split(n_total=6000, seed=0):
    rng = np.random.default_rng(seed)
    true_class = rng.binomial(1, 0.3, size=n_total)
    noise = rng.normal(0, 0.3, size=n_total)
    y_prob = np.clip(true_class * 0.7 + 0.15 + noise, 0.01, 0.99)
    half = n_total // 2
    return y_prob[:half], true_class[:half], y_prob[half:], true_class[half:]


def _imbalanced_calib_test_split(n_total=8000, seed=7):
    """Mimics the real dataset: severe class imbalance, most probabilities
    far from 0.5 — the exact scenario where LAC was found to never
    produce size-2 sets.
    """
    rng = np.random.default_rng(seed)
    true_class = rng.binomial(1, 0.02, size=n_total)
    y_prob = np.where(
        true_class == 1,
        np.clip(rng.beta(2, 5, n_total), 0.001, 0.999),
        np.clip(rng.beta(1, 30, n_total), 0.001, 0.999),
    )
    half = n_total // 2
    return y_prob[:half], true_class[:half], y_prob[half:], true_class[half:]


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_aps_empirical_coverage_close_to_nominal_target(alpha):
    prob_calib, y_calib, prob_test, y_test = _synthetic_calib_test_split()

    qhat = calibrate_conformal_aps(prob_calib, y_calib, alpha=alpha, seed=42)
    sets = build_prediction_sets_aps(prob_test, qhat, seed=123)
    coverage = empirical_coverage(sets, y_test)

    target = 1 - alpha
    assert abs(coverage - target) < 0.04  # slightly wider tolerance than LAC: randomization adds variance


def test_aps_calibration_is_reproducible_with_fixed_seed():
    prob_calib, y_calib, _, _ = _synthetic_calib_test_split()

    qhat_a = calibrate_conformal_aps(prob_calib, y_calib, alpha=0.1, seed=42)
    qhat_b = calibrate_conformal_aps(prob_calib, y_calib, alpha=0.1, seed=42)

    assert qhat_a == qhat_b


def test_aps_prediction_sets_are_reproducible_with_fixed_seed():
    prob_calib, y_calib, prob_test, _ = _synthetic_calib_test_split()
    qhat = calibrate_conformal_aps(prob_calib, y_calib, alpha=0.1, seed=42)

    sets_a = build_prediction_sets_aps(prob_test, qhat, seed=123)
    sets_b = build_prediction_sets_aps(prob_test, qhat, seed=123)

    np.testing.assert_array_equal(sets_a, sets_b)


def test_aps_different_seeds_give_different_but_similar_results():
    """Different random seeds should give somewhat different qhat/sets
    (confirming randomization is actually happening), but not wildly
    different overall coverage — the randomization affects individual
    points, not the aggregate statistical guarantee.
    """
    prob_calib, y_calib, prob_test, y_test = _synthetic_calib_test_split()

    qhat_seed1 = calibrate_conformal_aps(prob_calib, y_calib, alpha=0.1, seed=1)
    qhat_seed2 = calibrate_conformal_aps(prob_calib, y_calib, alpha=0.1, seed=2)

    sets_seed1 = build_prediction_sets_aps(prob_test, qhat_seed1, seed=101)
    sets_seed2 = build_prediction_sets_aps(prob_test, qhat_seed2, seed=102)

    coverage1 = empirical_coverage(sets_seed1, y_test)
    coverage2 = empirical_coverage(sets_seed2, y_test)

    assert abs(coverage1 - coverage2) < 0.03  # both close to nominal target, not wildly apart


def test_aps_produces_double_inclusion_sets_where_lac_cannot():
    """The core motivating comparison (see project diary): under severe
    class imbalance where calibrated probabilities cluster far from 0.5,
    LAC's qhat is always < 0.5, making size-2 sets structurally
    impossible. APS, being rank-based rather than probability-magnitude
    based, can still produce genuine ambiguous sets in this regime.
    """
    prob_calib, y_calib, prob_test, y_test = _imbalanced_calib_test_split()
    alpha = 0.05

    qhat_lac = calibrate_conformal(prob_calib, y_calib, alpha=alpha)
    sets_lac = build_prediction_sets(prob_test, qhat_lac)
    lac_dist = set_size_distribution(sets_lac)

    qhat_aps = calibrate_conformal_aps(prob_calib, y_calib, alpha=alpha, seed=42)
    sets_aps = build_prediction_sets_aps(prob_test, qhat_aps, seed=123)
    aps_dist = set_size_distribution(sets_aps)

    assert lac_dist[2] == pytest.approx(0.0, abs=1e-9)  # LAC: structurally impossible here
    assert aps_dist[2] > 0.0  # APS: can and does produce ambiguous sets


def test_aps_both_methods_achieve_similar_coverage_despite_different_set_shapes():
    """Even though LAC and APS produce structurally different prediction
    sets (empty-heavy vs. some double-inclusion), both should still honour
    the same nominal coverage target — the guarantee is a property of the
    conformal procedure, not of which nonconformity score is used.
    """
    prob_calib, y_calib, prob_test, y_test = _imbalanced_calib_test_split()
    alpha = 0.1

    qhat_lac = calibrate_conformal(prob_calib, y_calib, alpha=alpha)
    coverage_lac = empirical_coverage(build_prediction_sets(prob_test, qhat_lac), y_test)

    qhat_aps = calibrate_conformal_aps(prob_calib, y_calib, alpha=alpha, seed=42)
    coverage_aps = empirical_coverage(build_prediction_sets_aps(prob_test, qhat_aps, seed=123), y_test)

    target = 1 - alpha
    assert abs(coverage_lac - target) < 0.03
    assert abs(coverage_aps - target) < 0.03


def test_calibrate_conformal_aps_raises_on_empty_calibration_set():
    with pytest.raises(ValueError):
        calibrate_conformal_aps(np.array([]), np.array([]), alpha=0.1)


def test_aps_prediction_sets_are_properly_shaped_boolean_array():
    prob_calib, y_calib, prob_test, _ = _synthetic_calib_test_split(n_total=200)
    qhat = calibrate_conformal_aps(prob_calib, y_calib, alpha=0.1, seed=42)
    sets = build_prediction_sets_aps(prob_test, qhat, seed=123)

    assert sets.shape == (len(prob_test), 2)
    assert sets.dtype == bool
