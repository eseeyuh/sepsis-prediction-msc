"""
Unit tests for src/evaluation/decision_curve.py.

Run with: pytest tests/test_decision_curve.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.decision_curve import (
    decision_curve_data,
    net_benefit,
    net_benefit_treat_all,
    net_benefit_treat_none,
    summarize_decision_curve,
)


def test_net_benefit_perfect_model_equals_prevalence():
    """A perfect model (TP = all positives, FP = 0) has net benefit
    exactly equal to prevalence, at any threshold — the FP term vanishes
    entirely, leaving just TP/n = prevalence.
    """
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # prevalence = 0.2
    y_prob = y_true.astype(float)  # perfect separation, prob=1 iff true positive

    nb = net_benefit(y_true, y_prob, threshold=0.1)
    assert nb == pytest.approx(0.2)


def test_net_benefit_raises_on_boundary_thresholds():
    y_true = np.array([1, 0])
    y_prob = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        net_benefit(y_true, y_prob, threshold=0.0)
    with pytest.raises(ValueError):
        net_benefit(y_true, y_prob, threshold=1.0)


def test_net_benefit_treat_all_crosses_zero_exactly_at_prevalence():
    """Known analytical property of decision curve analysis: NB_all(pt)
    equals exactly 0 when pt equals the outcome's prevalence in the data.
    This is a strong, exact correctness check — not just a direction check.
    """
    y_true = np.array([1] * 20 + [0] * 80)  # prevalence = 0.2 exactly
    prevalence = 0.2

    nb_at_prevalence = net_benefit_treat_all(y_true, threshold=prevalence)
    assert nb_at_prevalence == pytest.approx(0.0, abs=1e-9)


def test_net_benefit_treat_all_positive_below_prevalence_negative_above():
    y_true = np.array([1] * 20 + [0] * 80)  # prevalence = 0.2

    nb_below = net_benefit_treat_all(y_true, threshold=0.05)
    nb_above = net_benefit_treat_all(y_true, threshold=0.5)

    assert nb_below > 0
    assert nb_above < 0


def test_net_benefit_treat_none_is_always_zero():
    assert net_benefit_treat_none() == 0.0


def test_decision_curve_data_has_expected_columns_and_length():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.1, size=500)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.1, 500), 0, 1)

    thresholds = np.linspace(0.01, 0.2, 20)
    curve = decision_curve_data(y_true, y_prob, thresholds=thresholds)

    assert len(curve) == 20
    assert set(curve.columns) == {
        "threshold", "net_benefit_model", "net_benefit_treat_all", "net_benefit_treat_none"
    }


def test_decision_curve_treat_none_column_is_always_zero():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.1, size=200)
    y_prob = rng.uniform(size=200)

    curve = decision_curve_data(y_true, y_prob)
    assert (curve["net_benefit_treat_none"] == 0.0).all()


def test_good_model_beats_both_reference_strategies_somewhere():
    """A model with real signal should have SOME threshold range where it
    beats both 'treat all' and 'treat none' — otherwise it would be
    clinically pointless despite good AUROC.
    """
    rng = np.random.default_rng(1)
    n = 3000
    y_true = rng.binomial(1, 0.05, size=n)
    # Reasonably good, imperfect probabilities correlated with truth.
    y_prob = np.clip(y_true * 0.4 + 0.02 + rng.normal(0, 0.05, n), 0.001, 0.999)

    curve = decision_curve_data(y_true, y_prob)
    summary = summarize_decision_curve(curve)

    assert summary["useful_threshold_range"] is not None
    assert summary["fraction_of_range_useful"] > 0


def test_useless_random_model_has_little_or_no_useful_range():
    """A model whose probabilities carry no information about the true
    label should rarely, if ever, beat BOTH reference strategies —
    unlike the good-model case above.
    """
    rng = np.random.default_rng(2)
    n = 3000
    y_true = rng.binomial(1, 0.05, size=n)
    y_prob = rng.uniform(0.0, 0.1, size=n)  # no relationship to y_true

    curve = decision_curve_data(y_true, y_prob)
    summary = summarize_decision_curve(curve)

    # Should be small — a random model shouldn't reliably beat both
    # informed reference strategies across a wide threshold range.
    assert summary["fraction_of_range_useful"] < 0.3


def test_summarize_decision_curve_best_threshold_has_max_net_benefit():
    rng = np.random.default_rng(3)
    n = 2000
    y_true = rng.binomial(1, 0.05, size=n)
    y_prob = np.clip(y_true * 0.4 + 0.02 + rng.normal(0, 0.05, n), 0.001, 0.999)

    curve = decision_curve_data(y_true, y_prob)
    summary = summarize_decision_curve(curve)

    max_nb_in_curve = curve["net_benefit_model"].max()
    assert summary["best_net_benefit"] == pytest.approx(max_nb_in_curve)
