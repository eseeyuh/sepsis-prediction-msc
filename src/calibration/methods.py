"""
Post-hoc probability calibration for a trained sepsis model.

Every calibrator here is fit on a held-out set — by convention in this
project, the VALIDATION split (already held out from training, used
earlier only for early stopping). Calibration is NEVER fit on the test
split: test must stay untouched until final evaluation, exactly as
required by QR1/QR4 in the project specification. This choice (calibrate
on val, evaluate calibration quality on test) should be stated explicitly
in the dissertation's Methodology chapter.

Three methods, in increasing order of flexibility:
  - Platt scaling: fits a 1-D logistic regression on the logit of the raw
    probability. Simple and robust with limited calibration data, but
    assumes a sigmoid-shaped miscalibration curve.
  - Temperature scaling: a restricted, single-parameter case of Platt
    scaling — only rescales the logit (no additive shift). Introduced for
    neural networks by Guo et al. (2017); included here for comparison
    even though the baseline model is a tree ensemble.
  - Isotonic regression: a non-parametric, monotonic step-function mapping.
    More flexible (fixes non-sigmoid miscalibration shapes) but needs more
    calibration data to avoid overfitting the mapping itself.
"""

from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Probability -> log-odds, clipped to avoid -inf/inf at exactly 0 or 1."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class PlattScaler:
    """calibrated_logit = a * raw_logit + b, fit by logistic regression.
    The classic Platt (1999) scaling method.
    """

    def __init__(self):
        self.model: Optional[LogisticRegression] = None

    def fit(self, y_prob_raw: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        logits = _logit(np.asarray(y_prob_raw)).reshape(-1, 1)
        self.model = LogisticRegression()
        self.model.fit(logits, np.asarray(y_true))
        return self

    def transform(self, y_prob_raw: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PlattScaler must be fit before calling transform().")
        logits = _logit(np.asarray(y_prob_raw)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


class TemperatureScaler:
    """calibrated_logit = raw_logit / T, with T > 0 found by minimising
    negative log-likelihood on the calibration set. T > 1 makes the model
    less confident (probabilities pulled toward 0.5); T < 1 makes it more
    confident. Optimised in log-space so T is automatically kept positive.
    """

    def __init__(self):
        self.temperature: Optional[float] = None

    def fit(self, y_prob_raw: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        logits = _logit(np.asarray(y_prob_raw))
        y_true = np.asarray(y_true)

        def negative_log_likelihood(log_t: float) -> float:
            t = np.exp(log_t)
            calibrated = _sigmoid(logits / t)
            calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
            return -np.mean(
                y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated)
            )

        result = minimize_scalar(negative_log_likelihood, bounds=(-3, 3), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, y_prob_raw: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("TemperatureScaler must be fit before calling transform().")
        logits = _logit(np.asarray(y_prob_raw))
        return _sigmoid(logits / self.temperature)


class IsotonicCalibrator:
    """Non-parametric monotonic mapping: raw probability -> calibrated
    probability, fit to minimise squared error subject to being
    non-decreasing. More flexible than the sigmoid-based methods above,
    at the cost of needing more calibration data to avoid overfitting the
    mapping itself.
    """

    def __init__(self):
        self.model: Optional[IsotonicRegression] = None

    def fit(self, y_prob_raw: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(np.asarray(y_prob_raw), np.asarray(y_true))
        return self

    def transform(self, y_prob_raw: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator must be fit before calling transform().")
        return self.model.predict(np.asarray(y_prob_raw))


CALIBRATORS = {
    "platt": PlattScaler,
    "temperature": TemperatureScaler,
    "isotonic": IsotonicCalibrator,
}


def fit_calibrator(method: str, y_prob_raw: np.ndarray, y_true: np.ndarray):
    """Convenience factory, e.g. fit_calibrator("temperature", val_probs, val_labels)."""
    if method not in CALIBRATORS:
        raise ValueError(f"Unknown calibration method '{method}'. Choose from {list(CALIBRATORS)}.")
    return CALIBRATORS[method]().fit(y_prob_raw, y_true)
