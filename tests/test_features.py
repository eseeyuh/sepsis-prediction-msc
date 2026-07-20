"""
Unit tests for src/data/features.py.

The key invariant tested here: every rolling/trend feature must be computed
using ONLY past and current hours for a given patient, and must never leak
information from another patient. Both are direct extensions of the
no-leakage principle we already enforce in splits.py — leakage can happen
at the feature level just as easily as at the split level.

Run with: pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_rolling_features,
    add_shock_index,
    add_time_since_admission,
    add_trend_features,
    engineer_features,
)


def _toy_frame():
    return pd.DataFrame({
        "patient_id": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
        "ICULOS": [1, 2, 3, 4, 1, 2, 3, 4],
        "HR": [70, 72, 75, 80, 200, 190, 180, 170],  # p2 deliberately very different
        "SBP": [120, 118, 115, 110, 80, 82, 85, 88],
        "MAP": [90, 89, 87, 85, 60, 61, 63, 65],
        "Resp": [16, 16, 17, 18, 30, 29, 28, 27],
        "Temp": [36.8, 36.9, 37.0, 37.3, 39.0, 38.8, 38.5, 38.2],
    })


def test_rolling_mean_uses_only_past_and_current_hours():
    df = _toy_frame()
    result = add_rolling_features(df, short_window=2, long_window=3)

    p1 = result[result["patient_id"] == "p1"].reset_index(drop=True)
    # At hour 2 (index 1), the 2-hour rolling mean of HR should be
    # mean(70, 72) = 71 — only hours 1 and 2, nothing from hour 3 or later.
    assert p1.loc[1, "HR_roll2h_mean"] == pytest.approx(71.0)


def test_rolling_features_do_not_leak_across_patients():
    df = _toy_frame()
    result = add_rolling_features(df, short_window=6, long_window=24)

    # p2's HR values are wildly different from p1's (200 vs 70-ish). If
    # rolling stats leaked across patients, p1's early rows would show
    # inflated means. They must not.
    p1 = result[result["patient_id"] == "p1"]
    assert p1["HR_roll6h_mean"].max() < 100  # nowhere near p2's ~200 range


def test_rolling_min_periods_1_gives_a_value_on_first_hour():
    df = _toy_frame()
    result = add_rolling_features(df, short_window=6, long_window=24)

    first_hour = result[(result["patient_id"] == "p1") & (result["ICULOS"] == 1)]
    # Even with only 1 hour of history, mean/min/max should equal that value,
    # not be NaN (min_periods=1 is what makes this work).
    assert first_hour["HR_roll6h_mean"].iloc[0] == 70
    assert first_hour["HR_roll6h_min"].iloc[0] == 70
    assert first_hour["HR_roll6h_max"].iloc[0] == 70


def test_trend_is_nan_before_window_fills_then_becomes_defined():
    df = _toy_frame()
    result = add_trend_features(df, window=2)

    p1 = result[result["patient_id"] == "p1"].reset_index(drop=True)
    # Hour 1: no prior value 2 hours back -> NaN.
    assert pd.isna(p1.loc[0, "HR_trend2h"])
    # Hour 3 (index 2): trend = (HR[3] - HR[1]) / 2 = (75 - 70) / 2 = 2.5
    assert p1.loc[2, "HR_trend2h"] == pytest.approx(2.5)


def test_trend_does_not_leak_across_patients():
    df = _toy_frame()
    result = add_trend_features(df, window=2)

    # p2's first two rows must not "borrow" a prior value from p1, even
    # though p1's rows come first in the concatenated DataFrame.
    p2_first = result[(result["patient_id"] == "p2") & (result["ICULOS"] == 1)]
    assert pd.isna(p2_first["HR_trend2h"].iloc[0])


def test_shock_index_computed_correctly():
    df = pd.DataFrame({"HR": [100], "SBP": [50]})
    result = add_shock_index(df)
    assert result["shock_index"].iloc[0] == 2.0


def test_shock_index_handles_zero_sbp_without_crashing():
    df = pd.DataFrame({"HR": [100], "SBP": [0]})
    result = add_shock_index(df)
    # Division by zero must become a clean NaN, not inf or a crash.
    assert pd.isna(result["shock_index"].iloc[0])


def test_time_since_admission_matches_iculos():
    df = pd.DataFrame({"ICULOS": [1, 2, 3]})
    result = add_time_since_admission(df)
    assert list(result["hours_since_admission"]) == [1, 2, 3]


def test_engineer_features_end_to_end_runs_without_error():
    df = _toy_frame()
    result = engineer_features(df)

    # Sanity check: original columns preserved, new columns added, same
    # number of rows (feature engineering must not drop or duplicate rows).
    assert len(result) == len(df)
    assert "shock_index" in result.columns
    assert "HR_roll6h_mean" in result.columns
    assert "HR_trend6h" in result.columns
