"""
Split conformal prediction for binary sepsis classification.

Implements the "least ambiguous set-valued classifier" (LAC) nonconformity
score (Sadinle, Lei & Wasserman, 2019), following the presentation in
Angelopoulos & Bates (2021), "A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification" — the standard reference
for this project's uncertainty quantification method (see project
specification O4).

Unlike a single probability, conformal prediction returns a PREDICTION SET
for each patient-hour — {no sepsis}, {sepsis}, or {no sepsis, sepsis} —
with a formal, finite-sample marginal coverage guarantee: for a chosen
error rate alpha, the true label is contained in the predicted set with
probability >= 1 - alpha. A set of size 2 is the model's own "I don't
know" signal — the abstention behaviour the project's Aim calls for.

IMPORTANT — validity assumption: split conformal prediction requires the
calibration set and the set it is applied to be EXCHANGEABLE (informally:
drawn from the same distribution). This holds for the val -> test split
used within a single hospital population here. It should NOT be assumed
to hold across the Phase 6 cross-site shift experiment (train/calibrate
on hospitals A+B, evaluate on a held-out partition simulating hospital
C) — watching the coverage guarantee break down under shift is itself
one of the things Phase 6 is designed to measure, not a bug to fix here.

NOTE on the calibration set: for simplicity, this project reuses the
validation split — already used for early stopping (Phase 2) and
probability calibration (Phase 3) — as the conformal calibration set too.
A methodologically stricter design would reserve a fourth, disjoint split
purely for conformal calibration. This is a documented simplification
(see project diary), not an oversight.
"""

from typing import Optional

import numpy as np


def _nonconformity_score(y_prob_positive: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """LAC nonconformity score: 1 - predicted probability of the TRUE
    class. Low score = model was confident and correct; high score =
    model was confident and wrong, or genuinely unsure.
    """
    y_prob_positive = np.asarray(y_prob_positive)
    y_true = np.asarray(y_true)
    p_true_class = np.where(y_true == 1, y_prob_positive, 1 - y_prob_positive)
    return 1 - p_true_class


def calibrate_conformal(
    y_prob_calib: np.ndarray, y_true_calib: np.ndarray, alpha: float = 0.1
) -> float:
    """Compute qhat, the conformal threshold, from a calibration set.

    Uses the standard finite-sample correction (Angelopoulos & Bates,
    2021, Section 2): qhat is the ceil((n+1)(1-alpha))/n empirical
    quantile of the nonconformity scores. This specific correction — not
    a naive (1-alpha) quantile — is what gives the coverage guarantee
    exactly, not just asymptotically as n -> infinity.
    """
    y_prob_calib = np.asarray(y_prob_calib)
    y_true_calib = np.asarray(y_true_calib)
    n = len(y_true_calib)
    if n == 0:
        raise ValueError("Calibration set is empty.")

    scores = _nonconformity_score(y_prob_calib, y_true_calib)

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # guard for very small n
    qhat = np.quantile(scores, q_level, method="higher")
    return float(qhat)


def build_prediction_sets(y_prob_positive: np.ndarray, qhat: float) -> np.ndarray:
    """Construct a prediction set for every point: a class is included if
    excluding it would be "too surprising" given qhat.

    Returns an (n, 2) boolean array: column 0 = "no sepsis" included,
    column 1 = "sepsis" included. Each row can have 0, 1, or 2 True
    values:
      - 2 True (both included): the model isn't confident enough to rule
        out either class — an explicit abstention signal.
      - 1 True: the model commits to a single class.
      - 0 True: possible in principle for a very small qhat; reported
        explicitly (via set_size_distribution) rather than silently
        patched, since a non-trivial empty-set rate would itself be worth
        investigating rather than hiding.
    """
    y_prob_positive = np.asarray(y_prob_positive)
    p_negative = 1 - y_prob_positive
    p_positive = y_prob_positive

    include_negative = (1 - p_negative) <= qhat
    include_positive = (1 - p_positive) <= qhat

    return np.stack([include_negative, include_positive], axis=1)


def empirical_coverage(prediction_sets: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of points whose TRUE label fell inside its predicted set.
    Should be close to (at least) 1 - alpha if calibration and test data
    are exchangeable, per the conformal guarantee.
    """
    y_true = np.asarray(y_true).astype(int)
    included = prediction_sets[np.arange(len(y_true)), y_true]
    return float(np.mean(included))


def average_set_size(prediction_sets: np.ndarray) -> float:
    """Mean number of classes included per prediction set. Ranges from 0
    (degenerate) to 2 (always abstaining). Smaller is more informative,
    for the same coverage level — this is the efficiency/informativeness
    side of the coverage-efficiency trade-off central to conformal
    prediction.
    """
    return float(prediction_sets.sum(axis=1).mean())


def set_size_distribution(prediction_sets: np.ndarray) -> dict:
    """Fraction of points with prediction set size exactly 0, 1, or 2."""
    sizes = prediction_sets.sum(axis=1)
    total = len(sizes)
    if total == 0:
        return {0: float("nan"), 1: float("nan"), 2: float("nan")}
    return {size: float(np.mean(sizes == size)) for size in [0, 1, 2]}


def singleton_accuracy(prediction_sets: np.ndarray, y_true: np.ndarray) -> Optional[float]:
    """Among points where the model committed to a SINGLE class (set size
    1), what fraction were correct? Returns None if there are no
    singleton predictions in this batch.
    """
    y_true = np.asarray(y_true).astype(int)
    sizes = prediction_sets.sum(axis=1)
    singleton_mask = sizes == 1

    if singleton_mask.sum() == 0:
        return None

    predicted_class = prediction_sets[:, 1].astype(int)  # the class that IS included, when size==1
    correct = predicted_class[singleton_mask] == y_true[singleton_mask]
    return float(np.mean(correct))
