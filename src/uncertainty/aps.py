"""
Adaptive Prediction Sets (APS) — a second conformal nonconformity score,
compared directly against the LAC score in conformal.py.

Motivation (see project diary): applied to this project's real, heavily
imbalanced data, the LAC score was found to almost never produce genuine
"ambiguous" {no sepsis, sepsis} sets — instead producing a growing rate of
EMPTY sets as alpha increased, because calibrated sepsis probabilities
rarely land near 0.5. APS is included specifically to test whether a
different, rank-based nonconformity score changes this behaviour. This is
a genuine methodological comparison, not a search for a "better" number —
whichever way it comes out is a reportable finding.

Reference: Romano, Sesia & Candès (2020), "Classification with Valid and
Adaptive Coverage", NeurIPS.

RANDOMIZATION: APS scores mix in a uniform random draw per point. This is
the standard formulation from the original paper, not an ad-hoc addition
— the non-randomized version collapses to only two possible score values
per class in binary classification (see module-level discussion in the
project diary), which makes calibration needlessly coarse/conservative.
Randomization requires a seeded RNG for reproducibility (QR4).

Prediction-set-level metrics (coverage, average size, size distribution,
singleton accuracy) are IDENTICAL in meaning regardless of which score
produced the underlying qhat, so this module deliberately reuses
empirical_coverage / average_set_size / set_size_distribution /
singleton_accuracy from conformal.py rather than duplicating them.
"""

import numpy as np


def _true_label_aps_scores(
    y_prob_positive: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomized APS nonconformity score for a given set of `labels`
    (used for the TRUE label at calibration time).

    For p = P(class 1):
      - if p >= 0.5 (class 1 ranks first): score(1) = u*p, score(0) = p + u*(1-p)
      - if p < 0.5  (class 0 ranks first): score(0) = u*(1-p), score(1) = (1-p) + u*p
    where u ~ Uniform(0, 1), drawn once per point.
    """
    y_prob_positive = np.asarray(y_prob_positive, dtype=float)
    labels = np.asarray(labels)
    u = rng.uniform(0.0, 1.0, size=len(y_prob_positive))

    p1 = y_prob_positive
    p0 = 1 - y_prob_positive
    class1_first = p1 >= 0.5

    cum_before = np.where(
        labels == 1,
        np.where(class1_first, 0.0, p0),
        np.where(class1_first, p1, 0.0),
    )
    own_prob = np.where(labels == 1, p1, p0)

    return cum_before + u * own_prob


def calibrate_conformal_aps(
    y_prob_calib: np.ndarray,
    y_true_calib: np.ndarray,
    alpha: float = 0.1,
    seed: int = 42,
) -> float:
    """APS analogue of calibrate_conformal() in conformal.py: same
    finite-sample-corrected quantile level, different (randomized,
    rank-based) nonconformity score.
    """
    y_prob_calib = np.asarray(y_prob_calib)
    y_true_calib = np.asarray(y_true_calib)
    n = len(y_true_calib)
    if n == 0:
        raise ValueError("Calibration set is empty.")

    rng = np.random.default_rng(seed)
    scores = _true_label_aps_scores(y_prob_calib, y_true_calib, rng)

    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def build_prediction_sets_aps(
    y_prob_positive: np.ndarray, qhat: float, seed: int = 123
) -> np.ndarray:
    """Build (n, 2) boolean prediction sets using the APS score: include
    candidate class y' if its randomized score <= qhat. Both candidate
    labels for a given point share the SAME random draw u — this is what
    the original formulation requires to keep the resulting sets properly
    nested as qhat increases (a larger qhat should never remove a class
    that a smaller qhat had included).

    Uses a different default seed to calibrate_conformal_aps deliberately
    — calibration-set and test-set randomization are conceptually
    different random draws, and keeping their default seeds visibly
    distinct avoids any appearance of shared randomness between the two.
    """
    y_prob_positive = np.asarray(y_prob_positive, dtype=float)
    n = len(y_prob_positive)
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n)

    p1 = y_prob_positive
    p0 = 1 - y_prob_positive
    class1_first = p1 >= 0.5

    cum_before_0 = np.where(class1_first, p1, 0.0)
    cum_before_1 = np.where(class1_first, 0.0, p0)

    score_0 = cum_before_0 + u * p0
    score_1 = cum_before_1 + u * p1

    include_0 = score_0 <= qhat
    include_1 = score_1 <= qhat

    return np.stack([include_0, include_1], axis=1)
