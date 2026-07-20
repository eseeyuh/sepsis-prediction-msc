"""
Unit tests for src/evaluation/metrics.py.

The utility score tests focus on SHAPE correctness (perfect predictions
score highest, inaction scores exactly 0 after normalisation, false alarms
are penalised, missing sepsis near onset is penalised more than missing it
early) rather than exact numeric equality with the official Challenge
script, since that script was not available to cross-check directly during
development. See the CAVEAT in metrics.py — cross-check against the
official script before quoting this metric in the dissertation.

Run with: pytest tests/test_metrics.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_auprc,
    compute_auroc,
    compute_challenge_utility,
    derive_t_sepsis,
    evaluate_predictions,
    find_best_threshold,
    patient_best_possible_utility,
    patient_inaction_utility,
    patient_utility,
)


# ---------------------------------------------------------------------------
# AUROC / AUPRC — thin wrappers, just confirm they behave sanely.
# ---------------------------------------------------------------------------


def test_auroc_is_1_for_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_auroc(y_true, y_prob) == pytest.approx(1.0)


def test_auroc_is_0point5_for_random_predictions():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert compute_auroc(y_true, y_prob) == pytest.approx(0.5)


def test_auprc_is_1_for_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_auprc(y_true, y_prob) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# t_sepsis derivation.
# ---------------------------------------------------------------------------


def test_derive_t_sepsis_for_septic_patient():
    hours = np.arange(1, 11)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # first septic hour = 5
    # t_sepsis = first septic hour + 6 = 11
    assert derive_t_sepsis(labels, hours) == 11.0


def test_derive_t_sepsis_is_none_for_non_septic_patient():
    hours = np.arange(1, 11)
    labels = np.zeros(10)
    assert derive_t_sepsis(labels, hours) is None


# ---------------------------------------------------------------------------
# Utility score — shape and normalisation correctness.
# ---------------------------------------------------------------------------


def _septic_patient_frame(n_hours=20, onset_label_hour=9):
    """A septic patient whose SepsisLabel turns 1 at `onset_label_hour`,
    so t_sepsis = onset_label_hour + 6.
    """
    hours = np.arange(1, n_hours + 1)
    labels = (hours >= onset_label_hour).astype(int)
    return pd.DataFrame({"patient_id": ["p1"] * n_hours, "ICULOS": hours, "SepsisLabel": labels})


def _nonseptic_patient_frame(n_hours=20):
    hours = np.arange(1, n_hours + 1)
    return pd.DataFrame({
        "patient_id": ["p2"] * n_hours,
        "ICULOS": hours,
        "SepsisLabel": np.zeros(n_hours, dtype=int),
    })


def test_always_positive_is_optimal_for_septic_patient():
    """For a septic patient, u_tp(tau) >= u_fn(tau) at every hour, so
    always predicting 1 should achieve the best possible score exactly.
    """
    df = _septic_patient_frame()
    hours = df["ICULOS"].to_numpy()
    labels = df["SepsisLabel"].to_numpy()

    always_positive = np.ones(len(df))
    achieved = patient_utility(hours, labels, always_positive)
    best = patient_best_possible_utility(hours, labels)

    assert achieved == pytest.approx(best)


def test_inaction_utility_matches_predicting_all_zero():
    df = _septic_patient_frame()
    hours = df["ICULOS"].to_numpy()
    labels = df["SepsisLabel"].to_numpy()

    always_negative = np.zeros(len(df))
    achieved = patient_utility(hours, labels, always_negative)
    inaction = patient_inaction_utility(hours, labels)

    assert achieved == pytest.approx(inaction)


def test_non_septic_patient_false_alarm_is_penalized():
    df = _nonseptic_patient_frame()
    hours = df["ICULOS"].to_numpy()
    labels = df["SepsisLabel"].to_numpy()

    always_positive = np.ones(len(df))  # constant false alarms
    always_negative = np.zeros(len(df))  # correct, quiet baseline

    score_false_alarms = patient_utility(hours, labels, always_positive)
    score_quiet = patient_utility(hours, labels, always_negative)

    assert score_false_alarms < score_quiet
    assert score_quiet == 0.0  # U_TN is 0 by definition


def test_missing_sepsis_at_onset_is_worse_than_missing_it_early():
    """A false negative right at/after onset (tau > dt_optimal) should cost
    more than a false negative far before onset (tau < dt_early), where
    the model isn't expected to have caught it yet.
    """
    hours = np.arange(1, 21)
    labels = (hours >= 9).astype(int)  # t_sepsis = 15

    # All-negative predictions throughout.
    predictions = np.zeros(20)

    from src.evaluation.metrics import _u_fn

    tau_far_before_onset = 1 - 15  # -14, before dt_early=-12
    tau_at_onset = 15 - 15  # 0, within the penalized ramp

    penalty_far_before = _u_fn(np.array([tau_far_before_onset]))[0]
    penalty_at_onset = _u_fn(np.array([tau_at_onset]))[0]

    assert penalty_at_onset < penalty_far_before  # more negative = worse


def test_normalized_utility_is_1_for_perfect_predictions_full_dataset():
    septic = _septic_patient_frame()
    nonseptic = _nonseptic_patient_frame()
    df = pd.concat([septic, nonseptic], ignore_index=True)

    # Optimal: predict 1 throughout for septic patient, 0 throughout for
    # non-septic patient.
    predictions = np.where(df["patient_id"] == "p1", 1, 0)

    result = compute_challenge_utility(df, predictions)
    assert result.normalized_score == pytest.approx(1.0)


def test_normalized_utility_is_0_for_inaction_on_full_dataset():
    septic = _septic_patient_frame()
    nonseptic = _nonseptic_patient_frame()
    df = pd.concat([septic, nonseptic], ignore_index=True)

    predictions = np.zeros(len(df))  # never alarm, for anyone

    result = compute_challenge_utility(df, predictions)
    assert result.normalized_score == pytest.approx(0.0)


def test_normalized_utility_is_negative_for_constant_false_alarms_only():
    """A model that only ever fires false alarms on a non-septic patient
    (and never catches the septic patient) should score below the
    inaction baseline — i.e. negative normalized utility.
    """
    nonseptic = _nonseptic_patient_frame()
    septic = _septic_patient_frame()
    df = pd.concat([septic, nonseptic], ignore_index=True)

    # False alarms on the non-septic patient, inaction (miss) on the septic one.
    predictions = np.where(df["patient_id"] == "p2", 1, 0)

    result = compute_challenge_utility(df, predictions)
    assert result.normalized_score < 0.0


# ---------------------------------------------------------------------------
# evaluate_predictions — integration of all metrics together.
# ---------------------------------------------------------------------------


def test_evaluate_predictions_returns_all_expected_keys():
    septic = _septic_patient_frame()
    nonseptic = _nonseptic_patient_frame()
    df = pd.concat([septic, nonseptic], ignore_index=True)

    # Assign probability based on the actual HOURLY label, not the patient
    # as a whole — this is what gives genuinely perfect AUROC/AUPRC, since
    # the septic patient's own pre-onset hours are legitimately negative.
    y_prob = np.where(df["SepsisLabel"] == 1, 0.9, 0.1)
    result = evaluate_predictions(df, y_prob, threshold=0.5)

    expected_keys = {
        "auroc", "auprc", "utility_normalized", "utility_raw",
        "utility_best", "utility_inaction", "threshold",
    }
    assert expected_keys.issubset(result.keys())
    assert result["auroc"] == pytest.approx(1.0)


def test_find_best_threshold_recovers_utility_lost_by_a_bad_fixed_threshold():
    """Regression test for the exact real-world finding this function
    exists to address: calibrated probabilities for a rare positive class
    can sit entirely below 0.5, making the model look useless at a fixed
    default threshold, even though a lower threshold recovers strong
    utility. find_best_threshold must find that lower threshold.
    """
    septic = _septic_patient_frame()
    nonseptic = _nonseptic_patient_frame()
    df = pd.concat([septic, nonseptic], ignore_index=True)

    # Probabilities that correctly RANK septic-hours higher, but are all
    # compressed below 0.5 — mirrors the real Platt/isotonic result.
    y_prob = np.where(df["SepsisLabel"] == 1, 0.30, 0.05)

    utility_at_default = evaluate_predictions(df, y_prob, threshold=0.5)["utility_normalized"]
    best_threshold, best_utility = find_best_threshold(df, y_prob)

    assert utility_at_default == pytest.approx(0.0)  # nothing crosses 0.5
    assert best_threshold < 0.30  # must find a threshold that DOES separate the classes
    assert best_utility > utility_at_default


def test_find_best_threshold_returns_value_within_requested_range():
    septic = _septic_patient_frame()
    nonseptic = _nonseptic_patient_frame()
    df = pd.concat([septic, nonseptic], ignore_index=True)

    y_prob = np.where(df["SepsisLabel"] == 1, 0.9, 0.1)
    thresholds = np.linspace(0.1, 0.9, 9)

    best_threshold, _ = find_best_threshold(df, y_prob, thresholds=thresholds)

    assert thresholds.min() <= best_threshold <= thresholds.max()
