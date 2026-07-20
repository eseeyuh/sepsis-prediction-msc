"""
Preprocessing for the PhysioNet/CinC Challenge 2019 dataset.

Pipeline, in order:
  1. Add a `{col}_missing` binary flag for every clinical variable, BEFORE
     any imputation. Missingness is informative in ICU data (a lab is often
     ordered because a clinician already suspects a problem), so this signal
     must be preserved, not thrown away by filling.
  2. Forward-fill within a clinically reasonable window, separately for
     vitals (measured often) and labs (measured less often). Beyond the
     window, a value is treated as genuinely unknown again.
  3. Fill any values still missing after step 2 (e.g. the first hours of a
     stay, before any real measurement exists) using per-column medians
     computed ONLY on the training split — this avoids leaking information
     from validation/test patients into imputation.

Usage:
    train_processed, stats = preprocess(train_df, fit_stats=True)
    val_processed, _ = preprocess(val_df, stats=stats)
    test_processed, _ = preprocess(test_df, stats=stats)
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

VITAL_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]

LAB_COLUMNS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

DEMOGRAPHIC_COLUMNS = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

CLINICAL_COLUMNS = VITAL_COLUMNS + LAB_COLUMNS

# How many consecutive hours we carry a value forward before treating it as
# unknown again. Vitals are recorded far more often than labs in practice,
# so labs get a longer allowance. These are a documented design choice —
# worth an ablation later (see dissertation "sensitivity analysis").
VITAL_FFILL_LIMIT_HOURS = 8
LAB_FFILL_LIMIT_HOURS = 24


#: Used only when a column is entirely missing in the training split, so its
#: median is undefined (NaN). This should be rare on the full dataset — if it
#: happens often, it's worth investigating rather than silently trusting 0.0.
FALLBACK_FILL_VALUE = 0.0


@dataclass
class ImputationStats:
    """Per-column fill values for remaining gaps, fit on the TRAINING split only."""

    medians: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, columns: list) -> "ImputationStats":
        self.medians = {}
        for col in columns:
            median = df[col].median()
            if pd.isna(median):
                warnings.warn(
                    f"Column '{col}' has no non-missing values in the training "
                    f"split — falling back to {FALLBACK_FILL_VALUE}. Check whether "
                    f"this column should be dropped instead."
                )
                median = FALLBACK_FILL_VALUE
            self.medians[col] = median
        return self


def add_missingness_flags(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Add a `{col}_missing` binary column for each variable.

    Must run BEFORE any forward-fill or imputation — it records the true,
    original missingness pattern, which is itself a predictive signal.
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def _sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure rows are ordered by patient, then by hour, before any
    forward-fill — ffill is meaningless on unordered rows.
    """
    sort_cols = ["patient_id"]
    if "ICULOS" in df.columns:
        sort_cols.append("ICULOS")
    return df.sort_values(sort_cols).reset_index(drop=True)


def forward_fill_within_limit(
    df: pd.DataFrame, columns: list, limit: int
) -> pd.DataFrame:
    """Forward-fill each column within each patient, capped at `limit`
    consecutive missing hours. Beyond that, the value stays missing.
    """
    df = df.copy()
    df[columns] = df.groupby("patient_id")[columns].ffill(limit=limit)
    return df


def fill_remaining_gaps(df: pd.DataFrame, stats: ImputationStats) -> pd.DataFrame:
    """Fill values still missing after forward-fill using training-set
    medians. Must run AFTER add_missingness_flags — the flags already
    captured the true missingness pattern, so this step is purely about
    giving the model a valid number to work with.
    """
    df = df.copy()
    for col, median in stats.medians.items():
        df[col] = df[col].fillna(median)
    return df


def preprocess(
    df: pd.DataFrame,
    stats: Optional[ImputationStats] = None,
    fit_stats: bool = False,
) -> tuple:
    """Run the full preprocessing pipeline on a long-format DataFrame
    (one row per patient-hour, as produced by src.data.loader).

    Args:
        df: raw data for ONE split (train, val, or test — never mix splits).
        stats: ImputationStats already fit on the training split. Required
            unless fit_stats=True.
        fit_stats: set True ONLY when `df` is the training split. Fits new
            ImputationStats on `df` itself. Passing the val/test split here
            would leak test-set statistics into imputation — this is exactly
            the kind of leakage QR1 in the project spec exists to prevent.

    Returns:
        (processed_df, stats) — reuse the returned `stats` (do not refit)
        when processing validation/test splits.
    """
    df = _sort_chronologically(df)
    df = add_missingness_flags(df, CLINICAL_COLUMNS)

    df = forward_fill_within_limit(df, VITAL_COLUMNS, VITAL_FFILL_LIMIT_HOURS)
    df = forward_fill_within_limit(df, LAB_COLUMNS, LAB_FFILL_LIMIT_HOURS)

    if fit_stats:
        stats = ImputationStats().fit(df, CLINICAL_COLUMNS)
    elif stats is None:
        raise ValueError(
            "Must pass `stats` fit on the training split, or set fit_stats=True "
            "if this IS the training split."
        )

    df = fill_remaining_gaps(df, stats)

    return df, stats


if __name__ == "__main__":
    # Quick manual smoke test — run with: python -m src.data.preprocess
    import numpy as np

    toy = pd.DataFrame({
        "patient_id": ["p1", "p1", "p1", "p2", "p2"],
        "ICULOS": [1, 2, 3, 1, 2],
        "HR": [80, np.nan, 84, 90, np.nan],
        "Lactate": [1.2, np.nan, np.nan, np.nan, 2.1],
    })
    for col in LAB_COLUMNS + VITAL_COLUMNS:
        if col not in toy.columns:
            toy[col] = np.nan

    processed, stats = preprocess(toy, fit_stats=True)
    print(processed[["patient_id", "ICULOS", "HR", "HR_missing", "Lactate", "Lactate_missing"]])
