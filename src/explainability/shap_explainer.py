"""
SHAP-based explainability for the LightGBM sepsis model.

Uses TreeSHAP (via the `shap` library's TreeExplainer), which computes
EXACT Shapley values for tree ensembles in polynomial time. Unlike
KernelSHAP (a general-purpose, sampling-based approximation), TreeSHAP is
both exact for this model class and fast enough to run on the full test
set rather than a small sample — see project specification O5.

VERSION NOTE: the installed `shap` version emits a UserWarning claiming
its LightGBM output format "has changed to a list of ndarray". Empirically
(verified during development against this project's LightGBM wrapper),
shap_values() returns a single ndarray directly for our model, not a
list. This wrapper handles both cases defensively so it keeps working
regardless of which format a given shap version actually returns.
"""

import numpy as np
import pandas as pd
import shap

from src.models.lightgbm_model import LightGBMSepsisModel


class SHAPExplainer:
    """Wraps a trained LightGBMSepsisModel to produce per-instance,
    per-feature SHAP values. Positive value = pushed the prediction toward
    "sepsis"; negative = pushed toward "no sepsis"; magnitude = strength.
    """

    def __init__(self, model: LightGBMSepsisModel):
        if model.booster is None:
            raise RuntimeError(
                "Model must be trained or loaded before creating an explainer."
            )
        self.model = model
        self.explainer = shap.TreeExplainer(model.booster)

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of SHAP values, same row index and (feature)
        column set as X, reordered to the model's training-time feature
        order first (same safety net as LightGBMSepsisModel.predict_proba).
        """
        X = X[self.model.feature_columns]
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            # Some shap versions return one array per class; take the
            # positive ("sepsis") class, conventionally index 1.
            shap_values = shap_values[1]

        return pd.DataFrame(shap_values, columns=self.model.feature_columns, index=X.index)

    def top_features_for_instance(
        self, x_row: pd.Series, shap_row: pd.Series, n: int = 5
    ) -> pd.DataFrame:
        """Convenience for a single patient-hour: the n features with the
        largest |SHAP value|, alongside their actual value and signed
        contribution. Useful for a bedside-style "why did it alert"
        printout, and for the demonstration notebook deliverable.
        """
        ranked = shap_row.abs().sort_values(ascending=False).head(n)
        return pd.DataFrame({
            "feature_value": x_row[ranked.index],
            "shap_value": shap_row[ranked.index],
        })
