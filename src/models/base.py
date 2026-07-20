"""
Common interface for every predictive model in this project (LightGBM now;
GRU and/or Transformer later if time allows). The training script and
evaluation harness are written against this interface, not against any
specific model — swapping LightGBM for a different model later should not
require touching src/train.py or src/evaluation/metrics.py.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that must never be used as model input:
#   - patient_id: an identifier, not a clinical feature
#   - SepsisLabel: the prediction target itself
#   - hospital: deliberately excluded from the FEATURE set for the baseline
#     model. Keeping it out means the model cannot learn "hospital" as a
#     shortcut feature, which would make the Phase 6 cross-site evaluation
#     (train on A+B, test on a held-out partition) meaningless — the model
#     would just be memorising a hospital ID rather than learning
#     transferable clinical patterns.
NON_FEATURE_COLUMNS = {"patient_id", "SepsisLabel", "hospital"}


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the model input columns in a fixed, reproducible (sorted)
    order. Using a sorted order everywhere means train/val/test and any
    saved/reloaded model always see features in the same column order,
    even if the DataFrame they came from was built differently.
    """
    return sorted(c for c in df.columns if c not in NON_FEATURE_COLUMNS)


class BaseSepsisModel(ABC):
    """Abstract interface every model in src/models must implement."""

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
    ) -> None:
        """Train the model. X_train/X_val must already be sliced down to
        feature columns only (see get_feature_columns) — this keeps the
        model class itself agnostic to how features were engineered.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted probability of sepsis for each row."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseSepsisModel":
        raise NotImplementedError
