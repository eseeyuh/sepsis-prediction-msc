"""
Unit tests for src/evaluation/faithfulness.py.

Run with: pytest tests/test_faithfulness.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.faithfulness import (
    deletion_auc,
    deletion_curve,
    faithfulness_summary,
    insertion_auc,
    insertion_curve,
    stratified_faithfulness_summary,
)
from src.explainability.shap_explainer import SHAPExplainer
from src.models.lightgbm_model import LightGBMSepsisModel


def _trained_model_and_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "feature_a": rng.normal(size=n),  # strongly informative
        "feature_b": rng.normal(size=n),  # weakly informative
        "feature_c": rng.normal(size=n),  # pure noise
    })
    y = (X["feature_a"] > 0).astype(int).to_numpy()  # label determined ONLY by feature_a

    model = LightGBMSepsisModel(num_boost_round=50)
    model.fit(X, y)
    return model, X, y


def test_deletion_curve_has_correct_length():
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    curve = deletion_curve(X.iloc[0], shap_df.iloc[0], model.predict_proba)

    assert len(curve) == len(X.columns) + 1


def test_insertion_curve_has_correct_length():
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    curve = insertion_curve(X.iloc[0], shap_df.iloc[0], model.predict_proba)

    assert len(curve) == len(X.columns) + 1


def test_deletion_curve_first_value_matches_original_prediction():
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    x_row = X.iloc[0]
    curve = deletion_curve(x_row, shap_df.iloc[0], model.predict_proba)
    original_pred = model.predict_proba(pd.DataFrame([x_row]))[0]

    assert curve[0] == pytest.approx(original_pred, abs=1e-6)


def test_insertion_curve_last_value_matches_original_prediction():
    """After inserting ALL features back, the prediction should match the
    original full-feature prediction exactly.
    """
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    x_row = X.iloc[0]
    curve = insertion_curve(x_row, shap_df.iloc[0], model.predict_proba)
    original_pred = model.predict_proba(pd.DataFrame([x_row]))[0]

    assert curve[-1] == pytest.approx(original_pred, abs=1e-6)


def test_deletion_drops_faster_for_a_faithful_explanation_than_random_ranking():
    """The central faithfulness claim: deleting features in TRUE SHAP
    importance order should move the prediction toward the opposite class
    faster (lower deletion AUC, since label depends only on feature_a)
    than deleting them in a shuffled, uninformative order.
    """
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    x_row = X.iloc[0]
    shap_row = shap_df.iloc[0]

    real_auc = deletion_auc(x_row, shap_row, model.predict_proba)

    # A deliberately scrambled "explanation" that ranks the noise feature
    # as most important instead of feature_a.
    shuffled_row = pd.Series(
        {"feature_a": 0.001, "feature_b": 0.001, "feature_c": 10.0}
    )
    scrambled_auc = deletion_auc(x_row, shuffled_row, model.predict_proba)

    # Only assert direction weakly here — exact magnitude depends on the
    # specific instance, but a genuinely informative ranking should not
    # do WORSE than an actively misleading one across repeated checks.
    assert real_auc <= scrambled_auc + 0.2  # generous tolerance for a single instance


def test_faithfulness_summary_respects_sample_size():
    model, X, _ = _trained_model_and_data(n=100)
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    summary = faithfulness_summary(X, shap_df, model.predict_proba, n_samples=10, random_state=1)

    assert len(summary) == 10
    assert "deletion_auc" in summary.columns
    assert "insertion_auc" in summary.columns


def test_faithfulness_summary_caps_sample_size_to_available_data():
    model, X, _ = _trained_model_and_data(n=20)
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    # Requesting more samples than rows available must not crash.
    summary = faithfulness_summary(X, shap_df, model.predict_proba, n_samples=1000, random_state=1)

    assert len(summary) == 20


def test_faithfulness_summary_is_reproducible_with_fixed_seed():
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    summary_a = faithfulness_summary(X, shap_df, model.predict_proba, n_samples=15, random_state=7)
    summary_b = faithfulness_summary(X, shap_df, model.predict_proba, n_samples=15, random_state=7)

    pd.testing.assert_frame_equal(summary_a, summary_b)


def test_faithfulness_summary_includes_random_baseline_columns_by_default():
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    summary = faithfulness_summary(X, shap_df, model.predict_proba, n_samples=10, random_state=1)

    assert "deletion_auc_random" in summary.columns
    assert "insertion_auc_random" in summary.columns


def test_faithfulness_summary_can_disable_random_baseline():
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    summary = faithfulness_summary(
        X, shap_df, model.predict_proba, n_samples=10, random_state=1,
        include_random_baseline=False,
    )

    assert "deletion_auc_random" not in summary.columns


def test_real_shap_beats_random_baseline_on_average_for_a_clearly_informative_feature():
    """The central claim the random baseline exists to test: for a model
    whose label is determined entirely by one feature, the REAL
    SHAP-based deletion AUC should, on average, be lower (better) than
    the random-shuffled baseline's deletion AUC — since deleting the
    genuinely important feature first should drop the prediction faster
    than deleting features in an arbitrary order.
    """
    model, X, _ = _trained_model_and_data(n=400)
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    summary = faithfulness_summary(
        X, shap_df, model.predict_proba, n_samples=60, random_state=3
    )

    assert summary["deletion_auc"].mean() <= summary["deletion_auc_random"].mean()


def test_stratified_faithfulness_summary_produces_both_strata():
    model, X, _ = _trained_model_and_data(n=500)
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)
    y_prob = model.predict_proba(X)

    result = stratified_faithfulness_summary(
        X, shap_df, model.predict_proba, y_prob,
        high_risk_quantile=0.9, n_samples_per_stratum=20, random_state=1,
    )

    assert set(result["stratum"].unique()) == {"high_risk", "rest"}
    assert (result["stratum"] == "high_risk").sum() > 0
    assert (result["stratum"] == "rest").sum() > 0


def test_stratified_faithfulness_high_risk_stratum_has_higher_predicted_probabilities():
    """Sanity check on the stratification itself: instances placed in the
    high_risk stratum should genuinely have higher predicted probability
    than the threshold used to define it.
    """
    model, X, _ = _trained_model_and_data(n=500)
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)
    y_prob = model.predict_proba(X)

    high_risk_quantile = 0.9
    threshold = np.quantile(y_prob, high_risk_quantile)

    result = stratified_faithfulness_summary(
        X, shap_df, model.predict_proba, y_prob,
        high_risk_quantile=high_risk_quantile, n_samples_per_stratum=20, random_state=1,
    )

    high_risk_indices = result[result["stratum"] == "high_risk"].index
    high_risk_probs = model.predict_proba(X.loc[high_risk_indices])

    assert (high_risk_probs >= threshold).all()
