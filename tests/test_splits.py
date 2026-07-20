"""
Unit tests for src/data/splits.py.

test_no_patient_overlap_between_splits is the single most important test in
this entire project — it directly protects QR1 in the project specification
(no patient may appear in more than one split). Every other test here exists
to build confidence in the split logic that feeds that guarantee.

Run with: pytest tests/test_splits.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    PatientSplits,
    apply_split,
    assert_no_patient_overlap,
    make_patient_splits,
)


def _toy_dataset(n_patients=100, hours_per_patient=5, sepsis_rate=0.2, seed=0):
    """A synthetic long-format dataset: n_patients patients, each with
    hours_per_patient rows. `sepsis_rate` fraction of patients get a
    SepsisLabel=1 on their last hour (the rest stay 0 throughout).
    """
    rng = np.random.default_rng(seed)
    n_septic = int(n_patients * sepsis_rate)
    septic_patient_idx = set(rng.choice(n_patients, size=n_septic, replace=False))

    rows = []
    for i in range(n_patients):
        patient_id = f"p{i:05d}"
        is_septic = i in septic_patient_idx
        for hour in range(hours_per_patient):
            label = 1 if (is_septic and hour == hours_per_patient - 1) else 0
            rows.append({"patient_id": patient_id, "ICULOS": hour + 1, "SepsisLabel": label})

    return pd.DataFrame(rows)


def test_no_patient_overlap_between_splits():
    df = _toy_dataset()
    splits = make_patient_splits(df, random_state=42)

    # Must not raise.
    assert_no_patient_overlap(splits)


def test_overlap_detection_actually_catches_a_leak():
    """Regression test for the checker itself: if we deliberately introduce
    an overlap, assert_no_patient_overlap must raise, not silently pass.
    """
    leaky_splits = PatientSplits(
        train_ids=["p1", "p2", "p3"],
        val_ids=["p3", "p4"],  # p3 leaked into both train and val
        test_ids=["p5"],
    )
    with pytest.raises(AssertionError):
        assert_no_patient_overlap(leaky_splits)


def test_split_covers_every_patient_exactly_once():
    df = _toy_dataset(n_patients=100)
    splits = make_patient_splits(df, random_state=42)

    all_ids = set(splits.train_ids) | set(splits.val_ids) | set(splits.test_ids)
    expected_ids = set(df["patient_id"].unique())

    assert all_ids == expected_ids
    total_assigned = len(splits.train_ids) + len(splits.val_ids) + len(splits.test_ids)
    assert total_assigned == df["patient_id"].nunique()


def test_split_proportions_are_approximately_correct():
    df = _toy_dataset(n_patients=1000)  # large enough for stable proportions
    splits = make_patient_splits(df, train_frac=0.70, val_frac=0.15, random_state=42)

    n_total = df["patient_id"].nunique()
    assert abs(len(splits.train_ids) / n_total - 0.70) < 0.02
    assert abs(len(splits.val_ids) / n_total - 0.15) < 0.02
    assert abs(len(splits.test_ids) / n_total - 0.15) < 0.02


def test_split_is_reproducible_with_same_seed():
    df = _toy_dataset()
    splits_a = make_patient_splits(df, random_state=42)
    splits_b = make_patient_splits(df, random_state=42)

    assert splits_a.train_ids == splits_b.train_ids
    assert splits_a.val_ids == splits_b.val_ids
    assert splits_a.test_ids == splits_b.test_ids


def test_different_seeds_give_different_splits():
    df = _toy_dataset(n_patients=200)
    splits_a = make_patient_splits(df, random_state=42)
    splits_b = make_patient_splits(df, random_state=7)

    assert splits_a.train_ids != splits_b.train_ids


def test_stratification_preserves_sepsis_rate_across_splits():
    """The proportion of septic patients should be roughly similar in each
    split — this is what stratify= in the underlying train_test_split call
    is protecting against (an unlucky split with almost no positive cases
    in test).
    """
    df = _toy_dataset(n_patients=1000, sepsis_rate=0.2)
    splits = make_patient_splits(df, random_state=42)

    patient_labels = df.groupby("patient_id")["SepsisLabel"].max()

    train_rate = patient_labels.loc[splits.train_ids].mean()
    val_rate = patient_labels.loc[splits.val_ids].mean()
    test_rate = patient_labels.loc[splits.test_ids].mean()

    for rate in (train_rate, val_rate, test_rate):
        assert abs(rate - 0.2) < 0.05


def test_apply_split_returns_only_requested_patients():
    df = _toy_dataset(n_patients=10, hours_per_patient=3)
    subset_ids = ["p00000", "p00001"]

    result = apply_split(df, subset_ids)

    assert set(result["patient_id"].unique()) == set(subset_ids)
    assert len(result) == 2 * 3  # 2 patients, 3 hours each


def test_save_and_load_roundtrip(tmp_path):
    splits = PatientSplits(train_ids=["p1", "p2"], val_ids=["p3"], test_ids=["p4"])
    path = tmp_path / "splits.json"

    splits.save(path)
    loaded = PatientSplits.load(path)

    assert loaded.train_ids == splits.train_ids
    assert loaded.val_ids == splits.val_ids
    assert loaded.test_ids == splits.test_ids


def test_invalid_fractions_raise_error():
    df = _toy_dataset(n_patients=20)
    with pytest.raises(ValueError):
        make_patient_splits(df, train_frac=0.7, val_frac=0.4)  # sums to > 1.0
