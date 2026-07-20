"""
Metrics for evaluating probability calibration: Brier score, Expected
Calibration Error (ECE), and the binned data a reliability diagram plots.

These are separate from src/evaluation/metrics.py (AUROC/AUPRC/utility)
because they measure a fundamentally different property: AUROC asks "can
the model rank septic patients above non-septic ones?" while these metrics
ask "when the model says 70%, does sepsis actually happen 70% of the
time?" A model can score perfectly on the first and poorly on the second —
exactly what we observed with the uncalibrated LightGBM baseline.
"""

from dataclasses import dataclass

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and the binary
    outcome. Bounded in [0, 1]; lower is better, 0 is perfect.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


@dataclass
class ReliabilityBin:
    """One bin of a reliability diagram: patients whose predicted
    probability fell in [bin_lower, bin_upper), and how their mean
    predicted probability compares to their actual observed event rate.
    """

    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_observed: float
    count: int


def reliability_diagram_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list:
    """Bin predictions into `n_bins` equal-width bins by predicted
    probability. In each bin, compute the mean predicted probability and
    the mean observed (actual) event rate. A perfectly calibrated model
    has mean_predicted == mean_observed in every bin — i.e. every point
    lies on the diagonal when plotted.

    Empty bins are returned with NaN means and count=0, rather than being
    dropped, so that plotting code can show gaps honestly instead of
    silently skipping them.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)  # include 1.0 in the last bin
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(lower, upper, float("nan"), float("nan"), 0))
            continue

        bins.append(ReliabilityBin(
            bin_lower=float(lower),
            bin_upper=float(upper),
            mean_predicted=float(y_prob[mask].mean()),
            mean_observed=float(y_true[mask].mean()),
            count=count,
        ))

    return bins


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: the average absolute gap between predicted probability and
    observed event rate across equal-width bins, weighted by bin size.
    Bounded in [0, 1]; lower is better, 0 is perfect calibration.
    """
    bins = reliability_diagram_data(y_true, y_prob, n_bins=n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")

    ece = 0.0
    for b in bins:
        if b.count == 0:
            continue
        weight = b.count / total
        ece += weight * abs(b.mean_predicted - b.mean_observed)
    return ece
