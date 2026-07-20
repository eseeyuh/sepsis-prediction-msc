"""
Unit tests for src/models/lightgbm_model.py.

Uses small synthetic data — not the real Challenge dataset — so these tests
run in well under a second and don't depend on the dataset being
downloaded. The goal here is to test the WRAPPER logic (fit/predict/
save/load, column-order safety), not LightGBM itself.

Run with: pytest tests/test_lightgbm_model.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.models.lightgbm_model import LightGBMSepsisModel


def _synthetic_data(n=400, seed=0):
    """A trivially separable synthetic dataset: label is (mostly)
    determined by feature_a, so a correctly-wired model should achieve
    high AUROC — if this test fails, the bug is almost certainly in our
    wrapper code, not in LightGBM itself.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "feature_a": rng.normal(size=n),
        "feature_b": rng.normal(size=n),  # noise, uninformative
    })
    y = (X["feature_a"] > 0).astype(int).to_numpy()
    return X, y


def test_fit_predict_achieves_high_auroc_on_easy_synthetic_task():
    X, y = _synthetic_data(n=400)
    X_train, y_train = X.iloc[:300], y[:300]
    X_val, y_val = X.iloc[300:], y[300:]

    model = LightGBMSepsisModel(num_boost_round=50, early_stopping_rounds=10)
    model.fit(X_train, y_train, X_val, y_val)

    probs = model.predict_proba(X_val)
    auroc = roc_auc_score(y_val, probs)

    assert auroc > 0.9


def test_predict_proba_returns_values_in_valid_probability_range():
    X, y = _synthetic_data(n=200)
    model = LightGBMSepsisModel(num_boost_round=20)
    model.fit(X, y)

    probs = model.predict_proba(X)

    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_before_fit_raises_clear_error():
    model = LightGBMSepsisModel()
    X, _ = _synthetic_data(n=10)

    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_save_before_fit_raises_clear_error(tmp_path):
    model = LightGBMSepsisModel()
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "model.txt")


def test_save_and_load_roundtrip_gives_identical_predictions(tmp_path):
    X, y = _synthetic_data(n=200)
    model = LightGBMSepsisModel(num_boost_round=20)
    model.fit(X, y)

    original_preds = model.predict_proba(X)

    save_path = tmp_path / "model.txt"
    model.save(save_path)

    loaded_model = LightGBMSepsisModel.load(save_path)
    loaded_preds = loaded_model.predict_proba(X)

    np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)


def test_feature_column_order_is_enforced_at_predict_time():
    """Regression guard for a realistic bug: if predict_proba is given
    columns in a different order than training, results must be identical
    (correctly realigned), not silently wrong.
    """
    X, y = _synthetic_data(n=200)
    model = LightGBMSepsisModel(num_boost_round=20)
    model.fit(X, y)

    X_reordered = X[["feature_b", "feature_a"]]  # deliberately reversed

    preds_original_order = model.predict_proba(X)
    preds_reordered = model.predict_proba(X_reordered)

    np.testing.assert_allclose(preds_original_order, preds_reordered, rtol=1e-5)


def test_predict_missing_required_column_raises_error():
    X, y = _synthetic_data(n=100)
    model = LightGBMSepsisModel(num_boost_round=10)
    model.fit(X, y)

    X_missing_column = X[["feature_a"]]  # dropped feature_b

    with pytest.raises(KeyError):
        model.predict_proba(X_missing_column)


def test_feature_importance_returns_series_indexed_by_feature_name():
    X, y = _synthetic_data(n=200)
    model = LightGBMSepsisModel(num_boost_round=20)
    model.fit(X, y)

    importances = model.feature_importance()

    assert set(importances.index) == {"feature_a", "feature_b"}
    # feature_a is the informative one — it should dominate feature_b.
    assert importances["feature_a"] > importances["feature_b"]


def test_custom_params_override_defaults_without_dropping_others():
    model = LightGBMSepsisModel(params={"learning_rate": 0.123})

    assert model.params["learning_rate"] == 0.123
    # Other defaults (e.g. objective) must still be present.
    assert model.params["objective"] == "binary"
