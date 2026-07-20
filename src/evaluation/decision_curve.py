"""
Decision curve analysis (Vickers & Elkin, 2006): translates classifier
performance into NET BENEFIT, a clinically interpretable quantity that
answers the question a clinician actually asks — "at this alert
threshold, does using the model help more than it harms, compared to
treating everyone or treating no one?" — see project specification O7.

Net benefit at threshold probability pt:
    NB(pt) = (TP / n) - (FP / n) * (pt / (1 - pt))

The term pt / (1 - pt) is the implied exchange rate between a false
positive and a true positive at that threshold: a clinician who would
intervene once predicted risk exceeds pt is implicitly saying they accept
pt/(1-pt) false alarms for every true catch at that threshold.

Two reference strategies are always computed alongside the model:
  - "Treat all": intervene on every instance regardless of the model.
      NB_all(pt) = prevalence - (1 - prevalence) * (pt / (1 - pt))
    This crosses zero exactly at pt = prevalence — a useful sanity check
    (see tests).
  - "Treat none": never intervene. NB_none(pt) = 0 for every pt, by
    definition.

A model is only clinically useful at a given threshold if its net benefit
EXCEEDS BOTH reference strategies there — beating "treat none" alone is
not sufficient, since "treat all" is a strong baseline exactly when pt is
very low, which it typically is for a rare condition like sepsis (see the
Phase 3 finding that honestly calibrated sepsis probabilities rarely
exceed 0.5).

CONSISTENCY NOTE: like every other metric in this project (AUROC, AUPRC,
utility score, ECE), net benefit here is computed at the PATIENT-HOUR
level, not per patient — every hourly prediction is treated as an
independent decision instance, matching the granularity of the underlying
hourly sepsis-alert task and keeping this metric directly comparable to
every other reported result in the dissertation.
"""

import numpy as np
import pandas as pd


def _validate_threshold(threshold: float) -> None:
    if not (0 < threshold < 1):
        raise ValueError(
            f"threshold must be strictly between 0 and 1 (got {threshold}); "
            "pt/(1-pt) is undefined at the boundaries."
        )


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """Net benefit of using the model at a single threshold probability."""
    _validate_threshold(threshold)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    if n == 0:
        return float("nan")

    predicted_positive = y_prob >= threshold
    tp = int(np.sum(predicted_positive & (y_true == 1)))
    fp = int(np.sum(predicted_positive & (y_true == 0)))

    exchange_rate = threshold / (1 - threshold)
    return (tp / n) - (fp / n) * exchange_rate


def net_benefit_treat_all(y_true: np.ndarray, threshold: float) -> float:
    """Net benefit of intervening on every instance, regardless of the model."""
    _validate_threshold(threshold)
    y_true = np.asarray(y_true)
    n = len(y_true)
    if n == 0:
        return float("nan")

    prevalence = float(np.mean(y_true))
    exchange_rate = threshold / (1 - threshold)
    return prevalence - (1 - prevalence) * exchange_rate


def net_benefit_treat_none() -> float:
    """Net benefit of never intervening — always exactly 0, at every
    threshold, by definition. Implemented as a function (not a bare
    constant) so calling code can treat all three strategies uniformly.
    """
    return 0.0


def decision_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = None,
) -> pd.DataFrame:
    """Compute net benefit for the model, "treat all", and "treat none"
    across a range of threshold probabilities.

    Default threshold range (0.001 to 0.30) is deliberately narrow and
    concentrated near the low end: sepsis is rare, so clinically
    plausible alert thresholds sit far below 0.5.
    """
    if thresholds is None:
        thresholds = np.linspace(0.001, 0.30, 100)

    records = []
    for pt in thresholds:
        records.append({
            "threshold": float(pt),
            "net_benefit_model": net_benefit(y_true, y_prob, pt),
            "net_benefit_treat_all": net_benefit_treat_all(y_true, pt),
            "net_benefit_treat_none": net_benefit_treat_none(),
        })

    return pd.DataFrame(records)


def summarize_decision_curve(curve_df: pd.DataFrame) -> dict:
    """Summarise a decision curve into a few reportable numbers:
      - the threshold range where the model beats BOTH reference
        strategies (the range where using the model is genuinely useful)
      - the threshold at which the model achieves its own peak net benefit
    """
    model_wins = (
        (curve_df["net_benefit_model"] > curve_df["net_benefit_treat_all"]) &
        (curve_df["net_benefit_model"] > curve_df["net_benefit_treat_none"])
    )

    if model_wins.any():
        useful_range = (
            float(curve_df.loc[model_wins, "threshold"].min()),
            float(curve_df.loc[model_wins, "threshold"].max()),
        )
        fraction_useful = float(model_wins.mean())
    else:
        useful_range = None
        fraction_useful = 0.0

    best_idx = curve_df["net_benefit_model"].idxmax()

    return {
        "useful_threshold_range": useful_range,
        "fraction_of_range_useful": fraction_useful,
        "best_threshold": float(curve_df.loc[best_idx, "threshold"]),
        "best_net_benefit": float(curve_df.loc[best_idx, "net_benefit_model"]),
    }
