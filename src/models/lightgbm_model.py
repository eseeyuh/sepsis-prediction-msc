"""
LightGBM baseline model for sepsis prediction.

Chosen as the first (and, within the 4-week scoped plan, primary) model —
see project specification O2 and the scoping discussion in the project
diary. LightGBM trains in minutes rather than hours, handles missing values
and mixed feature scales natively, and gives a strong, fast-to-iterate
baseline on top of which calibration, uncertainty, and explainability are
built in later phases.
"""

import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseSepsisModel

# Sensible defaults for an imbalanced binary classification task on tabular
# clinical time-series features. `is_unbalance=True` lets LightGBM
# automatically reweight the loss for the rare positive (sepsis) class,
# since positive hours are a small minority of all patient-hours.
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "is_unbalance": True,
    "verbosity": -1,
    "seed": 42,
}


class LightGBMSepsisModel(BaseSepsisModel):
    def __init__(
        self,
        params: Optional[dict] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 30,
    ):
        # Merge user-supplied params over the defaults, rather than
        # replacing them wholesale — lets a config file override just
        # learning_rate, say, without having to restate every other param.
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.booster: Optional[lgb.Booster] = None
        self.feature_columns: Optional[list] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
    ) -> None:
        # Freeze the column order seen at training time — predict_proba
        # will re-index to this exact order, so a caller can never
        # accidentally feed columns in a different order and silently get
        # garbage predictions.
        self.feature_columns = list(X_train.columns)

        train_set = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_columns)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=50)]

        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val[self.feature_columns], label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError(
                "Model has not been trained or loaded yet — call fit() or load() first."
            )
        # Re-index to the exact training-time column order. This is a
        # deliberate safety net: if a caller passes columns in a different
        # order (or an extra/missing column), this either fixes it
        # silently (reorder) or raises a clear KeyError (missing column),
        # rather than LightGBM quietly matching the wrong values to the
        # wrong feature names.
        X = X[self.feature_columns]
        best_iteration = self.booster.best_iteration or self.booster.current_iteration()
        return self.booster.predict(X, num_iteration=best_iteration)

    def save(self, path: Path) -> None:
        if self.booster is None:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

        # LightGBM's own save_model does not preserve our enforced column
        # order, so we save it alongside — load() reads this back to
        # restore the same safety net predict_proba relies on.
        feature_path = path.with_suffix(path.suffix + ".features.json")
        with open(feature_path, "w") as f:
            json.dump(self.feature_columns, f)

    @classmethod
    def load(cls, path: Path) -> "LightGBMSepsisModel":
        path = Path(path)
        instance = cls()
        instance.booster = lgb.Booster(model_file=str(path))

        feature_path = path.with_suffix(path.suffix + ".features.json")
        with open(feature_path) as f:
            instance.feature_columns = json.load(f)

        return instance

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Convenience method for a quick look at which features the model
        relies on most — NOT a substitute for the proper SHAP-based
        explainability work in Phase 5, just a fast sanity check during
        development (e.g. "is Lactate actually near the top, as expected
        clinically?").
        """
        if self.booster is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        importances = self.booster.feature_importance(importance_type=importance_type)
        return pd.Series(importances, index=self.feature_columns).sort_values(ascending=False)
