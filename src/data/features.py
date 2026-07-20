"""
Feature engineering for the PhysioNet/CinC Challenge 2019 dataset.

Adds features that capture TRENDS over time, not just the current-hour
value — this matters clinically, since a deteriorating patient is defined
by a trajectory (e.g. heart rate climbing over several hours), not by any
single measurement in isolation.

All rolling/trend features are computed per patient, using only past and
current hours (never future hours) — this is what makes them usable at
prediction time in a real deployment, where future values are unknown.

Adds three families of features:
  1. Rolling statistics (mean, min, max, std) over short (6h) and longer
     (24h) windows, for the core vital signs.
  2. A simple trend estimate (slope) over a short window, for the same
     vitals — captures direction of change, not just spread.
  3. Clinically-motivated composite scores (e.g. shock index) that are
     cheap to compute and have known predictive value in the literature.
"""

import numpy as np
import pandas as pd

# Vitals we compute rolling features for. Kept to a focused subset rather
# than all 8 vital columns — these are the ones most consistently linked to
# early deterioration in the sepsis literature (heart rate, blood pressure,
# respiratory rate, temperature).
ROLLING_FEATURE_COLUMNS = ["HR", "SBP", "MAP", "Resp", "Temp"]

SHORT_WINDOW_HOURS = 6
LONG_WINDOW_HOURS = 24


def _rolling_stats_for_patient(
    series: pd.Series, window: int, min_periods: int = 1
) -> pd.DataFrame:
    """Rolling mean/min/max/std for a single patient's time series.

    min_periods=1 means we compute a statistic even with just 1 hour of
    history (e.g. hour 1 has a 'rolling mean' equal to itself) rather than
    returning NaN until the window fully fills — early hours still get
    usable features this way.
    """
    roll = series.rolling(window=window, min_periods=min_periods)
    return pd.DataFrame({
        "mean": roll.mean(),
        "min": roll.min(),
        "max": roll.max(),
        "std": roll.std(),
    })


def add_rolling_features(
    df: pd.DataFrame, columns: list = None, short_window: int = SHORT_WINDOW_HOURS,
    long_window: int = LONG_WINDOW_HOURS,
) -> pd.DataFrame:
    """Add rolling mean/min/max/std over a short and a long window, per patient.

    Must run on data that is already sorted by patient then by hour, and
    AFTER preprocessing (missing values already filled) — rolling stats
    over raw NaNs would be misleading.
    """
    columns = columns or ROLLING_FEATURE_COLUMNS
    df = df.copy()

    for col in columns:
        for window, label in [(short_window, f"{short_window}h"), (long_window, f"{long_window}h")]:
            stats = (
                df.groupby("patient_id")[col]
                .apply(lambda s: _rolling_stats_for_patient(s, window))
                .reset_index(level=0, drop=True)
            )
            for stat_name in ["mean", "min", "max", "std"]:
                df[f"{col}_roll{label}_{stat_name}"] = stats[stat_name]

    return df


def _trend_slope(series: pd.Series, window: int) -> pd.Series:
    """A simple, fast trend estimate: (current value - value `window` hours
    ago) / window. Positive = rising, negative = falling. Cheap alternative
    to fitting a full linear regression per window, and easy to explain in
    the dissertation.
    """
    return (series - series.shift(window)) / window


def add_trend_features(
    df: pd.DataFrame, columns: list = None, window: int = SHORT_WINDOW_HOURS
) -> pd.DataFrame:
    """Add a short-window trend (slope) feature per patient, per column."""
    columns = columns or ROLLING_FEATURE_COLUMNS
    df = df.copy()

    for col in columns:
        df[f"{col}_trend{window}h"] = (
            df.groupby("patient_id")[col]
            .apply(lambda s: _trend_slope(s, window))
            .reset_index(level=0, drop=True)
        )

    return df


def add_time_since_admission(df: pd.DataFrame) -> pd.DataFrame:
    """ICULOS already tracks hours since ICU admission, but we keep it as
    an explicit named feature for clarity and to make it easy to swap in a
    different admission-time definition later without touching other code.
    """
    df = df.copy()
    df["hours_since_admission"] = df["ICULOS"]
    return df


def add_shock_index(df: pd.DataFrame) -> pd.DataFrame:
    """Shock index = heart rate / systolic blood pressure.

    A simple, well-established clinical composite: rises when heart rate
    climbs and/or blood pressure drops — exactly the pattern seen in early
    sepsis-related deterioration. Guarded against division by zero / missing
    SBP, which would otherwise produce inf or NaN silently.
    """
    df = df.copy()
    safe_sbp = df["SBP"].replace(0, np.nan)
    df["shock_index"] = df["HR"] / safe_sbp
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline, in order.

    Expects `df` to already be preprocessed (see src.data.preprocess) —
    sorted by patient and hour, with missing values already handled.
    """
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_time_since_admission(df)
    df = add_shock_index(df)
    return df


if __name__ == "__main__":
    # Quick manual smoke test — run with: python -m src.data.features
    toy = pd.DataFrame({
        "patient_id": ["p1"] * 8,
        "ICULOS": list(range(1, 9)),
        "HR": [70, 72, 75, 80, 85, 92, 100, 110],
        "SBP": [120, 118, 115, 110, 105, 100, 95, 88],
        "MAP": [90, 89, 87, 85, 83, 80, 77, 72],
        "Resp": [16, 16, 17, 18, 19, 20, 22, 24],
        "Temp": [36.8, 36.9, 37.0, 37.3, 37.6, 38.0, 38.4, 38.8],
    })

    result = engineer_features(toy)
    print(result[["ICULOS", "HR", "HR_roll6h_mean", "HR_trend6h", "shock_index"]])
