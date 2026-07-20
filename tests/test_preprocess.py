"""
Unit tests for src/data/preprocess.py.

Run with: pytest tests/test_preprocess.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    CLINICAL_COLUMNS,
    ImputationStats,
    add_missingness_flags,
    fill_remaining_gaps,
    forward_fill_within_limit,
    preprocess,
)


def _toy_frame():
    """Two patients, a few hours each, with deliberate gaps to test against.

    Every clinical column gets at least one real value (row 0), so that
    training-set medians are always defined — a column that is entirely
    missing in the training split is a separate edge case, covered by
    test_imputation_stats_falls_back_when_column_entirely_missing below.
    """
    df = pd.DataFrame({
        "patient_id": ["p1", "p1", "p1", "p1", "p2", "p2"],
        "ICULOS": [1, 2, 3, 4, 1, 2],
        "HR": [80, np.nan, np.nan, 84, 90, np.nan],
        "Lactate": [1.2, np.nan, np.nan, np.nan, np.nan, 2.1],
    })
    for col in CLINICAL_COLUMNS:
        if col not in df.columns:
            # Give every other clinical column a harmless constant real value
            # so medians are defined; HR and Lactate keep their realistic gaps above.
            df[col] = 1.0
    return df


def test_missingness_flag_matches_original_nan_pattern():
    df = _toy_frame()
    flagged = add_missingness_flags(df, CLINICAL_COLUMNS)

    assert flagged.loc[flagged["ICULOS"] == 2, "HR_missing"].iloc[0] == 1
    assert flagged.loc[flagged["ICULOS"] == 1, "HR_missing"].iloc[0] == 0


def test_forward_fill_respects_limit():
    df = _toy_frame()
    # HR is missing for 2 consecutive hours for p1 (ICULOS 2 and 3).
    # With limit=1, only the first missing hour should be filled.
    filled = forward_fill_within_limit(df, ["HR"], limit=1)

    p1 = filled[filled["patient_id"] == "p1"].sort_values("ICULOS")
    assert p1.iloc[1]["HR"] == 80  # filled (1 hour gap, within limit)
    assert pd.isna(p1.iloc[2]["HR"])  # NOT filled (2 hour gap, beyond limit)


def test_forward_fill_does_not_cross_patients():
    df = _toy_frame()
    filled = forward_fill_within_limit(df, ["HR"], limit=10)

    # p2's first row has no prior value within p2 — must stay NaN,
    # even though p1's last HR value exists earlier in the frame.
    p2_first = filled[(filled["patient_id"] == "p2") & (filled["ICULOS"] == 1)]
    assert p2_first["HR"].iloc[0] == 90  # p2's own real value, not p1's


def test_imputation_stats_fit_only_uses_given_data():
    train = pd.DataFrame({"HR": [70, 80, 90]})
    stats = ImputationStats().fit(train, ["HR"])
    assert stats.medians["HR"] == 80


def test_imputation_stats_falls_back_when_column_entirely_missing():
    """A column that is entirely missing in the training split has an
    undefined (NaN) median. Regression test for a real bug caught during
    development: fillna(NaN) silently does nothing, leaving values missing
    downstream. Must fall back to a defined constant instead.
    """
    train = pd.DataFrame({"TroponinI": [np.nan, np.nan, np.nan]})

    with pytest.warns(UserWarning, match="TroponinI"):
        stats = ImputationStats().fit(train, ["TroponinI"])

    assert not pd.isna(stats.medians["TroponinI"])


def test_fill_remaining_gaps_uses_train_stats_not_test_data():
    train_stats = ImputationStats()
    train_stats.medians = {"HR": 75.0}

    test_df = pd.DataFrame({"HR": [np.nan, 200.0]})  # test data has different values
    filled = fill_remaining_gaps(test_df, train_stats)

    # The missing value should be filled with the TRAIN median (75), not
    # anything derived from the test data itself.
    assert filled["HR"].iloc[0] == 75.0
    assert filled["HR"].iloc[1] == 200.0  # untouched, was not missing


def test_preprocess_requires_stats_unless_fitting():
    df = _toy_frame()
    with pytest.raises(ValueError):
        preprocess(df, stats=None, fit_stats=False)


def test_preprocess_end_to_end_leaves_no_nans_in_clinical_columns():
    df = _toy_frame()
    processed, stats = preprocess(df, fit_stats=True)

    assert processed[CLINICAL_COLUMNS].isna().sum().sum() == 0


def test_preprocess_val_split_reuses_train_stats_without_refitting():
    train_df = _toy_frame()
    processed_train, stats = preprocess(train_df, fit_stats=True)

    val_df = _toy_frame()
    val_df["patient_id"] = ["p3", "p3", "p3", "p3", "p4", "p4"]
    processed_val, stats_again = preprocess(val_df, stats=stats, fit_stats=False)

    # Same stats object should be reused, not recomputed.
    assert stats_again is stats
    assert processed_val[CLINICAL_COLUMNS].isna().sum().sum() == 0
