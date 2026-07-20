"""
Evaluation metrics for sepsis prediction models.

Includes:
  - AUROC, AUPRC — standard discrimination metrics (thin wrappers around
    scikit-learn; no custom logic, so little room for subtle bugs here).
  - Challenge utility score — the PhysioNet/CinC 2019 Challenge's own
    metric, which rewards early correct predictions of sepsis and
    penalises late/missed predictions and false alarms.

IMPORTANT CAVEAT on the utility score:
This is a reimplementation based on the published Challenge description
(Reyna et al., 2020, "Early Prediction of Sepsis from Clinical Data: The
PhysioNet/Computing in Cardiology Challenge 2019", Critical Care Medicine)
and the parameter values reproduced across several derivative works. It has
NOT been cross-checked line-by-line against the official scoring script
(physionetchallenges/evaluation-2019 on GitHub). Before reporting this
number in the dissertation, download the official script and verify it
gives the same result on a held-out sample — treat any discrepancy as
authoritative in favour of the official script, not this one.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# AUROC / AUPRC — standard metrics, sklearn-backed.
# ---------------------------------------------------------------------------


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the ROC curve. y_true: binary labels. y_prob: predicted
    probability of the positive class (sepsis).
    """
    return roc_auc_score(y_true, y_prob)


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the precision-recall curve. More informative than AUROC
    under class imbalance (sepsis-positive hours are a small minority here).
    """
    return average_precision_score(y_true, y_prob)


# ---------------------------------------------------------------------------
# Challenge utility score.
# ---------------------------------------------------------------------------

# Official Challenge parameters (see module docstring for provenance).
DT_EARLY = -12       # hours before onset where reward starts ramping up
DT_OPTIMAL = -6      # hours before onset = the single best time to predict
DT_LATE = 3          # hours after onset where reward/penalty ramp ends

MAX_U_TP = 1.0       # reward at the optimal prediction time
MIN_U_FN = -2.0      # penalty for missing sepsis at/after onset
U_FP = -0.05         # penalty for a false alarm on a non-septic patient
U_TN = 0.0           # no reward/penalty for correctly staying quiet


def _u_tp(tau: np.ndarray) -> np.ndarray:
    """Reward for predicting sepsis=1 at time offset `tau` hours relative
    to true onset, for a septic patient. Ramps 0 -> MAX_U_TP over
    [DT_EARLY, DT_OPTIMAL], then back down to 0 over [DT_OPTIMAL, DT_LATE].
    Zero outside that window.
    """
    tau = np.asarray(tau, dtype=float)
    reward = np.zeros_like(tau)

    rising = (tau >= DT_EARLY) & (tau <= DT_OPTIMAL)
    reward[rising] = MAX_U_TP * (tau[rising] - DT_EARLY) / (DT_OPTIMAL - DT_EARLY)

    falling = (tau > DT_OPTIMAL) & (tau <= DT_LATE)
    reward[falling] = MAX_U_TP * (DT_LATE - tau[falling]) / (DT_LATE - DT_OPTIMAL)

    return reward


def _u_fn(tau: np.ndarray) -> np.ndarray:
    """Penalty for predicting sepsis=0 (missing it) at time offset `tau`
    hours relative to true onset, for a septic patient. Zero before the
    optimal time (not yet expected to have caught it), ramping down to
    MIN_U_FN by DT_LATE, and staying at MIN_U_FN after that (still missing
    it well after onset is just as bad as missing it right at onset).
    """
    tau = np.asarray(tau, dtype=float)
    penalty = np.zeros_like(tau)

    ramping = (tau > DT_OPTIMAL) & (tau <= DT_LATE)
    penalty[ramping] = MIN_U_FN * (tau[ramping] - DT_OPTIMAL) / (DT_LATE - DT_OPTIMAL)

    late = tau > DT_LATE
    penalty[late] = MIN_U_FN

    return penalty


def derive_t_sepsis(sepsis_labels: np.ndarray, hours: np.ndarray) -> Optional[float]:
    """Recover the clinical onset hour t_sepsis from the hourly SepsisLabel
    column. The Challenge's SepsisLabel is 1 starting 6 hours BEFORE
    clinical onset (i.e. it already encodes the early-prediction target),
    so t_sepsis = (first hour where SepsisLabel==1) + 6.

    Returns None if the patient has no septic hour (i.e. is not septic).

    NOTE: this derivation assumes the standard Challenge labelling
    convention. If you change the prediction horizon in
    src/data/labeling.py (e.g. to 4h or 8h), update this function to match,
    or utility scores will be computed relative to the wrong onset time.
    """
    septic_hours = hours[sepsis_labels == 1]
    if len(septic_hours) == 0:
        return None
    return float(np.min(septic_hours)) + 6.0


def patient_utility(
    hours: np.ndarray,
    sepsis_labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Unnormalised Challenge utility score for a single patient's full stay.

    Args:
        hours: hourly time index for this patient (e.g. ICULOS), sorted ascending.
        sepsis_labels: binary SepsisLabel per hour, same length as `hours`.
        predictions: binary (0/1) model decisions per hour, same length.
            Must already be thresholded — this function does not threshold
            probabilities itself.

    Returns:
        Sum of per-hour utility for this patient.
    """
    hours = np.asarray(hours, dtype=float)
    sepsis_labels = np.asarray(sepsis_labels)
    predictions = np.asarray(predictions)

    t_sepsis = derive_t_sepsis(sepsis_labels, hours)

    if t_sepsis is None:
        # Non-septic patient: constant penalty/reward regardless of timing.
        return float(np.sum(np.where(predictions == 1, U_FP, U_TN)))

    tau = hours - t_sepsis
    reward_if_positive = _u_tp(tau)
    reward_if_negative = _u_fn(tau)
    per_hour = np.where(predictions == 1, reward_if_positive, reward_if_negative)
    return float(np.sum(per_hour))


def patient_best_possible_utility(hours: np.ndarray, sepsis_labels: np.ndarray) -> float:
    """Best achievable utility for this patient: predict 1 at every hour if
    septic (u_tp >= u_fn everywhere, so always-positive is optimal for a
    septic patient), predict 0 at every hour if not septic (u_tn > u_fp).
    """
    hours = np.asarray(hours, dtype=float)
    sepsis_labels = np.asarray(sepsis_labels)
    t_sepsis = derive_t_sepsis(sepsis_labels, hours)

    if t_sepsis is None:
        return 0.0  # always predicting 0 gives U_TN=0 at every hour

    tau = hours - t_sepsis
    return float(np.sum(_u_tp(tau)))


def patient_inaction_utility(hours: np.ndarray, sepsis_labels: np.ndarray) -> float:
    """Utility if the model always predicts 0 (never raises an alarm) —
    the baseline used to normalise the score, matching the official
    Challenge definition of "inaction"."""
    hours = np.asarray(hours, dtype=float)
    sepsis_labels = np.asarray(sepsis_labels)
    t_sepsis = derive_t_sepsis(sepsis_labels, hours)

    if t_sepsis is None:
        return 0.0  # non-septic, always predicting 0 -> U_TN=0 throughout

    tau = hours - t_sepsis
    return float(np.sum(_u_fn(tau)))


@dataclass
class UtilityScoreResult:
    raw_score: float
    inaction_score: float
    best_score: float
    normalized_score: float


def compute_challenge_utility(
    df: pd.DataFrame,
    predictions: np.ndarray,
    patient_col: str = "patient_id",
    hour_col: str = "ICULOS",
    label_col: str = "SepsisLabel",
) -> UtilityScoreResult:
    """Compute the normalised Challenge utility score across all patients.

    normalized_score = (raw - inaction) / (best - inaction)

    This bounds the score at roughly 0 (no better than never alerting) to 1
    (perfect, optimally-timed predictions for every patient). Scores can go
    below 0 if the model performs worse than doing nothing (e.g. constant
    false alarms), which is itself a meaningful and reportable result.

    Args:
        df: long-format DataFrame with one row per patient-hour, containing
            at least `patient_col`, `hour_col`, `label_col`.
        predictions: binary (0/1) array, same length and row order as `df`.
    """
    df = df.copy()
    df["_prediction"] = np.asarray(predictions)

    raw_total = 0.0
    inaction_total = 0.0
    best_total = 0.0

    for _, group in df.groupby(patient_col):
        hours = group[hour_col].to_numpy()
        labels = group[label_col].to_numpy()
        preds = group["_prediction"].to_numpy()

        raw_total += patient_utility(hours, labels, preds)
        inaction_total += patient_inaction_utility(hours, labels)
        best_total += patient_best_possible_utility(hours, labels)

    denominator = best_total - inaction_total
    normalized = (raw_total - inaction_total) / denominator if denominator != 0 else 0.0

    return UtilityScoreResult(
        raw_score=raw_total,
        inaction_score=inaction_total,
        best_score=best_total,
        normalized_score=normalized,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper — everything at once.
# ---------------------------------------------------------------------------


def evaluate_predictions(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    patient_col: str = "patient_id",
    hour_col: str = "ICULOS",
    label_col: str = "SepsisLabel",
) -> dict:
    """Run the full evaluation suite: AUROC, AUPRC (probability-based) and
    the normalised Challenge utility score (threshold-based).

    `threshold` controls the operating point used for the utility score
    only — AUROC/AUPRC are threshold-independent by construction. Different
    thresholds will give different utility scores; reporting utility at
    several thresholds is a reasonable thing to do in the dissertation.
    """
    y_true = df[label_col].to_numpy()
    y_prob = np.asarray(y_prob)
    predictions = (y_prob >= threshold).astype(int)

    utility = compute_challenge_utility(
        df, predictions, patient_col=patient_col, hour_col=hour_col, label_col=label_col
    )

    return {
        "auroc": compute_auroc(y_true, y_prob),
        "auprc": compute_auprc(y_true, y_prob),
        "utility_normalized": utility.normalized_score,
        "utility_raw": utility.raw_score,
        "utility_best": utility.best_score,
        "utility_inaction": utility.inaction_score,
        "threshold": threshold,
    }


def find_best_threshold(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    patient_col: str = "patient_id",
    hour_col: str = "ICULOS",
    label_col: str = "SepsisLabel",
) -> tuple:
    """Search for the threshold that maximises normalised Challenge
    utility, evaluated on the SAME data `df`/`y_prob` this function is
    called with.

    IMPORTANT: call this on the VALIDATION set only, never on test — the
    chosen threshold is a decision derived from data, exactly like a model
    hyperparameter, so selecting it on test would leak test information
    into the final reported numbers. Apply the returned threshold to test
    afterwards with evaluate_predictions(test_df, test_prob, threshold=best_threshold).

    This directly addresses a real finding from this project: after proper
    calibration, well-calibrated probabilities for a rare event (sepsis)
    rarely exceed 0.5, so the conventional default threshold can silently
    make a well-calibrated model look useless. The right threshold is a
    choice to make deliberately, not an assumption to leave unexamined.

    Returns:
        (best_threshold, best_utility_normalized)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = None
    best_utility = -np.inf

    for threshold in thresholds:
        result = evaluate_predictions(
            df, y_prob, threshold=threshold,
            patient_col=patient_col, hour_col=hour_col, label_col=label_col,
        )
        if result["utility_normalized"] > best_utility:
            best_utility = result["utility_normalized"]
            best_threshold = float(threshold)

    return best_threshold, best_utility
