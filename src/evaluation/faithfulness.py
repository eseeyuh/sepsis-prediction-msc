"""
Faithfulness evaluation for SHAP explanations: deletion and insertion AUC.

SHAP magnitude alone cannot answer whether an explanation is actually
CORRECT — it can only say "the model's math attributes this much credit to
this feature". Faithfulness tests ask a sharper question: if the features
SHAP ranks as most important are actually removed (deletion), or are the
ONLY ones kept (insertion), does the model's prediction change the way the
explanation implies it should? An explanation that "looks reasonable" but
fails this test is not trustworthy enough for a clinical audience — see
project specification O5.

BASELINE CHOICE: a "removed" feature is set to NaN, not 0 or the feature
mean. This is deliberate: LightGBM natively routes missing values through
a learned default split direction (see src/models/lightgbm_model.py), so
NaN genuinely represents "this information is unavailable to the model" —
consistent with how missingness is already treated throughout this
project's preprocessing pipeline (src/data/preprocess.py). A numeric
placeholder like 0 would not have this property, since 0 is a real,
meaningful value for many engineered features (e.g. *_trend6h).

PERFORMANCE NOTE: computing a full deletion or insertion curve for one
instance requires n_features + 1 model predictions. Rather than calling
predict_fn once per step (n_features + 1 separate calls), this module
builds the full sequence of intermediate feature states as a single
DataFrame and calls predict_fn ONCE per instance — cutting per-instance
cost by roughly a factor of n_features. Even so, running this over
thousands of test rows is expensive; see src/explain.py, which
deliberately evaluates faithfulness on a random SAMPLE of instances
rather than the full test set.
"""

from typing import Callable

import numpy as np
import pandas as pd


def _order_features_by_importance(shap_row: pd.Series) -> list:
    """Feature names ordered from most to least important, by |SHAP value|."""
    return shap_row.abs().sort_values(ascending=False).index.tolist()


def deletion_curve(
    x_row: pd.Series,
    shap_row: pd.Series,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    """Predicted probability as features are progressively DELETED (set to
    NaN), cumulatively, in order of decreasing SHAP importance.

    Returns an array of length n_features + 1: index 0 = prediction with
    all features present, index i = prediction with the top i features
    (cumulatively) removed.
    """
    ordered_features = _order_features_by_importance(shap_row)
    n = len(ordered_features)

    # Build every intermediate state as one batch (row 0 = original,
    # row i = top-i features deleted), then predict ONCE on the whole batch.
    states = pd.DataFrame([x_row.to_numpy()] * (n + 1), columns=x_row.index)
    for i in range(1, n + 1):
        feature = ordered_features[i - 1]
        states.loc[i:, feature] = np.nan  # cumulative: stays deleted in all later rows too

    return np.asarray(predict_fn(states), dtype=float)


def insertion_curve(
    x_row: pd.Series,
    shap_row: pd.Series,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    """Predicted probability as features are progressively INSERTED
    (restored from an all-NaN baseline), cumulatively, in order of
    decreasing SHAP importance.

    Returns an array of length n_features + 1: index 0 = all features NaN,
    index i = top i features (cumulatively) restored.
    """
    ordered_features = _order_features_by_importance(shap_row)
    n = len(ordered_features)

    states = pd.DataFrame(
        np.nan, index=range(n + 1), columns=x_row.index
    )
    for i in range(1, n + 1):
        feature = ordered_features[i - 1]
        states.loc[i:, feature] = x_row[feature]

    return np.asarray(predict_fn(states), dtype=float)


def _normalised_auc(curve: np.ndarray) -> float:
    """Area under a curve, with the x-axis normalised to [0, 1] (fraction
    of features processed) via the trapezoidal rule — this makes AUC
    values comparable across instances/feature sets of different sizes.

    Uses np.trapezoid if available (NumPy >= 2.0) and falls back to the
    older np.trapz name otherwise, since both this project's development
    environment and a student's local environment may run different
    NumPy versions — np.trapz was renamed/deprecated in NumPy 2.0.
    """
    x = np.linspace(0, 1, len(curve))
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid_fn(curve, x))


def deletion_auc(x_row: pd.Series, shap_row: pd.Series, predict_fn: Callable) -> float:
    """Lower is better: a steep, early drop in prediction as important
    features are deleted means the explanation correctly identified
    features the model actually depends on.
    """
    return _normalised_auc(deletion_curve(x_row, shap_row, predict_fn))


def insertion_auc(x_row: pd.Series, shap_row: pd.Series, predict_fn: Callable) -> float:
    """Higher is better: a steep, early rise in prediction as important
    features are inserted means those features alone can drive the
    prediction toward the model's original output.
    """
    return _normalised_auc(insertion_curve(x_row, shap_row, predict_fn))


def faithfulness_summary(
    X: pd.DataFrame,
    shap_df: pd.DataFrame,
    predict_fn: Callable,
    n_samples: int = 200,
    random_state: int = 42,
    include_random_baseline: bool = True,
) -> pd.DataFrame:
    """Compute deletion and insertion AUC for a random sample of
    instances (not the full dataset — see module docstring on cost).

    If include_random_baseline is True (default), also compute the same
    metrics using a SHUFFLED feature ranking (same SHAP magnitudes,
    randomly reassigned to different features) for each sampled instance.
    A genuinely faithful explanation should beat this baseline — reporting
    deletion/insertion AUC in isolation, without this comparison, does not
    establish that SHAP is doing anything better than an arbitrary
    ordering would.
    """
    rng = np.random.default_rng(random_state)
    n_available = len(X)
    sample_size = min(n_samples, n_available)
    sample_idx = rng.choice(X.index, size=sample_size, replace=False)

    records = []
    for idx in sample_idx:
        x_row = X.loc[idx]
        shap_row = shap_df.loc[idx]

        record = {
            "index": idx,
            "deletion_auc": deletion_auc(x_row, shap_row, predict_fn),
            "insertion_auc": insertion_auc(x_row, shap_row, predict_fn),
        }

        if include_random_baseline:
            shuffled_row = pd.Series(
                rng.permutation(shap_row.to_numpy()), index=shap_row.index
            )
            record["deletion_auc_random"] = deletion_auc(x_row, shuffled_row, predict_fn)
            record["insertion_auc_random"] = insertion_auc(x_row, shuffled_row, predict_fn)

        records.append(record)

    return pd.DataFrame(records).set_index("index")


def stratified_faithfulness_summary(
    X: pd.DataFrame,
    shap_df: pd.DataFrame,
    predict_fn: Callable,
    y_prob: np.ndarray,
    high_risk_quantile: float = 0.9,
    n_samples_per_stratum: int = 100,
    random_state: int = 42,
    include_random_baseline: bool = True,
) -> pd.DataFrame:
    """Compute faithfulness SEPARATELY for "high risk" instances (predicted
    probability in the top (1 - high_risk_quantile) of this dataset, e.g.
    the top decile if high_risk_quantile=0.9) versus the rest.

    Motivation: sepsis is rare, so the vast majority of test instances
    have a very low predicted probability regardless of model quality.
    Averaging deletion/insertion AUC over a uniform random sample is
    dominated by these confidently-negative instances, where curves are
    naturally compressed near zero — this can obscure how faithful
    explanations are for the instances that actually matter clinically:
    the ones where the model is close to raising an alert, and a
    clinician would actually read and rely on the explanation.

    Returns a single DataFrame with a `stratum` column ("high_risk" or
    "rest") so both groups can be compared directly.
    """
    y_prob = np.asarray(y_prob)
    threshold = np.quantile(y_prob, high_risk_quantile)
    is_high_risk = y_prob >= threshold

    high_risk_idx = X.index[is_high_risk]
    rest_idx = X.index[~is_high_risk]

    def _summary_for_subset(idx_subset, stratum_name):
        if len(idx_subset) == 0:
            return pd.DataFrame()
        summary = faithfulness_summary(
            X.loc[idx_subset], shap_df.loc[idx_subset], predict_fn,
            n_samples=n_samples_per_stratum, random_state=random_state,
            include_random_baseline=include_random_baseline,
        )
        summary["stratum"] = stratum_name
        return summary

    high_summary = _summary_for_subset(high_risk_idx, "high_risk")
    rest_summary = _summary_for_subset(rest_idx, "rest")

    return pd.concat([high_summary, rest_summary])
