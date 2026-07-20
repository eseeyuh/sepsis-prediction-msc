"""
Unit tests for src/uncertainty/conformal.py.

The central property under test throughout: the empirical coverage
achieved on a held-out (test-like) set should be close to the nominal
1 - alpha target, when calibration and test data are exchangeable (drawn
from the same synthetic process here). This is the whole point of
conformal prediction, so it gets the most scrutiny.

Run with: pytest tests/test_conformal.py -v
"""

import numpy as np
import pytest

from src.uncertainty.conformal import (
    average_set_size,
    build_prediction_sets,
    calibrate_conformal,
    empirical_coverage,
    set_size_distribution,
    singleton_accuracy,
)


def _synthetic_calib_test_split(n_total=6000, seed=0):
    """A reasonably good but imperfect classifier: probabilities correlate
    with the true class but aren't perfect. Split into a calibration half
    and a test half, drawn from the same process (exchangeable, as
    required by conformal prediction's validity assumption).
    """
    rng = np.random.default_rng(seed)
    true_class = rng.binomial(1, 0.3, size=n_total)
    noise = rng.normal(0, 0.3, size=n_total)
    y_prob = np.clip(true_class * 0.7 + 0.15 + noise, 0.01, 0.99)

    half = n_total // 2
    return (
        y_prob[:half], true_class[:half],   # calibration set
        y_prob[half:], true_class[half:],    # test set
    )


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_empirical_coverage_close_to_nominal_target(alpha):
    """Coverage should be close to 1 - alpha. We allow a generous
    tolerance since this is a finite-sample statistical guarantee, not an
    exact equality — occasional small undershoot/overshoot is expected,
    but it should not be wildly off.
    """
    prob_calib, y_calib, prob_test, y_test = _synthetic_calib_test_split()

    qhat = calibrate_conformal(prob_calib, y_calib, alpha=alpha)
    sets = build_prediction_sets(prob_test, qhat)
    coverage = empirical_coverage(sets, y_test)

    target = 1 - alpha
    assert abs(coverage - target) < 0.03


def test_larger_alpha_gives_smaller_average_set_size():
    """Loosening the coverage requirement (larger alpha) should make the
    method more willing to commit to a single class, on average — a
    smaller average set size. This is the core coverage-efficiency
    trade-off conformal prediction is built around.
    """
    prob_calib, y_calib, prob_test, _ = _synthetic_calib_test_split()

    qhat_strict = calibrate_conformal(prob_calib, y_calib, alpha=0.05)
    qhat_loose = calibrate_conformal(prob_calib, y_calib, alpha=0.3)

    size_strict = average_set_size(build_prediction_sets(prob_test, qhat_strict))
    size_loose = average_set_size(build_prediction_sets(prob_test, qhat_loose))

    assert size_loose < size_strict


def test_qhat_matches_hand_computed_value_on_tiny_example():
    """5 calibration points, alpha=0.4 -> hand-checkable quantile level."""
    y_prob_calib = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y_true_calib = np.array([1, 1, 1, 1, 1])  # all positive, all "correct" at varying confidence
    # nonconformity scores = 1 - p(true class) = [0.1, 0.2, 0.3, 0.4, 0.5]

    n = 5
    alpha = 0.4
    # q_level = ceil((5+1)*(1-0.4))/5 = ceil(3.6)/5 = 4/5 = 0.8
    qhat = calibrate_conformal(y_prob_calib, y_true_calib, alpha=alpha)

    # The 0.8 quantile (via "higher" interpolation) of [0.1,0.2,0.3,0.4,0.5] is 0.5.
    assert qhat == pytest.approx(0.5)


def test_perfect_confident_model_gives_mostly_singleton_correct_sets():
    rng = np.random.default_rng(1)
    n = 2000
    y_true = rng.binomial(1, 0.3, size=n)
    y_prob = np.where(y_true == 1, 0.97, 0.03)  # near-perfect, confident

    half = n // 2
    qhat = calibrate_conformal(y_prob[:half], y_true[:half], alpha=0.1)
    sets = build_prediction_sets(y_prob[half:], qhat)

    dist = set_size_distribution(sets)
    assert dist[1] > 0.9  # the vast majority should be confident singletons

    acc = singleton_accuracy(sets, y_true[half:])
    assert acc > 0.95


def test_noisy_uninformative_model_gives_more_abstentions():
    """A model whose probabilities carry little information about the
    true class should be forced into wider (more abstaining) sets to
    maintain coverage, compared to the well-behaved model above.
    """
    rng = np.random.default_rng(2)
    n = 2000
    y_true = rng.binomial(1, 0.3, size=n)
    y_prob = np.clip(0.5 + rng.normal(0, 0.05, size=n), 0.01, 0.99)  # near-random, ~0.5 always

    half = n // 2
    qhat = calibrate_conformal(y_prob[:half], y_true[:half], alpha=0.1)
    sets = build_prediction_sets(y_prob[half:], qhat)

    avg_size = average_set_size(sets)
    assert avg_size > 1.5  # mostly abstaining (set size close to 2)


def test_empty_prediction_sets_are_reported_not_hidden():
    """At a large alpha (loose target), the LAC score can legitimately
    produce empty prediction sets — this is documented, expected behaviour
    of this method, not a bug. The set_size_distribution function must
    surface this rather than silently patching it away.
    """
    prob_calib, y_calib, prob_test, y_test = _synthetic_calib_test_split()

    qhat = calibrate_conformal(prob_calib, y_calib, alpha=0.3)
    sets = build_prediction_sets(prob_test, qhat)
    dist = set_size_distribution(sets)

    assert 0 in dist  # the key must be present even if the rate is 0
    assert dist[0] >= 0.0


def test_empirical_coverage_treats_empty_sets_as_non_coverage():
    """A row with no included classes cannot possibly cover the true
    label — empirical_coverage must reflect that (not error, not silently
    exclude these rows from the denominator).
    """
    prediction_sets = np.array([
        [False, False],  # empty set
        [True, False],   # covers class 0
        [False, True],   # covers class 1
    ])
    y_true = np.array([0, 0, 1])

    coverage = empirical_coverage(prediction_sets, y_true)
    # row 0: true=0, set=empty -> not covered
    # row 1: true=0, set={0} -> covered
    # row 2: true=1, set={1} -> covered
    assert coverage == pytest.approx(2 / 3)


def test_calibrate_conformal_raises_on_empty_calibration_set():
    with pytest.raises(ValueError):
        calibrate_conformal(np.array([]), np.array([]), alpha=0.1)


def test_singleton_accuracy_returns_none_when_no_singletons():
    prediction_sets = np.array([[True, True], [True, True]])  # always abstaining
    y_true = np.array([0, 1])
    assert singleton_accuracy(prediction_sets, y_true) is None
