"""
Unit tests for src/explainability/shap_explainer.py.

Run with: pytest tests/test_shap_explainer.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import SHAPExplainer
from src.models.lightgbm_model import LightGBMSepsisModel


def _trained_model_and_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "feature_a": rng.normal(size=n),  # informative, strong
        "feature_b": rng.normal(size=n),  # informative, weaker
        "feature_c": rng.normal(size=n),  # pure noise
    })
    y = (X["feature_a"] + 0.5 * X["feature_b"] > 0).astype(int).to_numpy()

    model = LightGBMSepsisModel(num_boost_round=50)
    model.fit(X, y)
    return model, X, y


def test_explainer_requires_trained_model():
    untrained = LightGBMSepsisModel()
    with pytest.raises(RuntimeError):
        SHAPExplainer(untrained)


def test_explain_returns_dataframe_with_matching_shape():
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)

    shap_df = explainer.explain(X)

    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == model.feature_columns
    assert list(shap_df.index) == list(X.index)


def test_shap_values_satisfy_additivity_property():
    """The defining property of Shapley values: base value + sum of SHAP
    values for a row must equal the model's raw margin output for that
    row. This is what makes SHAP values 'exact' for tree models rather
    than a heuristic approximation — worth testing explicitly rather than
    trusting the library blindly.
    """
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    raw_margin = model.booster.predict(X[model.feature_columns], raw_score=True)
    reconstructed = shap_df.sum(axis=1).to_numpy() + explainer.explainer.expected_value

    np.testing.assert_allclose(reconstructed, raw_margin, atol=1e-6)


def test_informative_feature_has_higher_mean_absolute_shap_than_noise():
    """Sanity check that SHAP actually identifies the informative feature
    (feature_a, which the label was constructed from) as more important
    than pure noise (feature_c).
    """
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    mean_abs_shap = shap_df.abs().mean()
    assert mean_abs_shap["feature_a"] > mean_abs_shap["feature_c"]


def test_top_features_for_instance_returns_requested_count():
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X)

    top = explainer.top_features_for_instance(X.iloc[0], shap_df.iloc[0], n=2)

    assert len(top) == 2
    assert "feature_value" in top.columns
    assert "shap_value" in top.columns


def test_explain_reorders_columns_to_match_training_order():
    """Regression guard mirroring the same concern already tested for
    LightGBMSepsisModel.predict_proba: passing columns in a different
    order must not silently misalign SHAP values with the wrong features.
    """
    model, X, _ = _trained_model_and_data()
    explainer = SHAPExplainer(model)

    X_reordered = X[["feature_c", "feature_a", "feature_b"]]
    shap_original = explainer.explain(X)
    shap_from_reordered = explainer.explain(X_reordered)

    pd.testing.assert_frame_equal(shap_original, shap_from_reordered, check_like=False)
