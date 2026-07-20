"""
Patient-level train/val/test splitting for the PhysioNet/CinC Challenge 2019 dataset.

The single most important rule in this module: split by PATIENT, never by row.
A patient's hours must never be spread across more than one split — doing so
leaks information from that patient's future (or past) into whichever split
tests on them, and silently inflates every downstream metric.

Splits are stratified on patient-level sepsis status (did this patient have
at least one septic hour, yes/no) so that the rare positive class is
represented proportionally in train/val/test, rather than landing unevenly
by chance.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PatientSplits:
    """Frozen patient-ID assignment for one train/val/test split."""

    train_ids: list
    val_ids: list
    test_ids: list

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "PatientSplits":
        with open(path) as f:
            data = json.load(f)
        return cls(train_ids=data["train"], val_ids=data["val"], test_ids=data["test"])


def _patient_level_sepsis_label(df: pd.DataFrame) -> pd.Series:
    """Reduce hourly SepsisLabel to one label per patient: 1 if the patient
    had sepsis at any hour, else 0. Used only for stratifying the split —
    the hourly labels themselves are untouched everywhere else.
    """
    return df.groupby("patient_id")["SepsisLabel"].max()


def make_patient_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> PatientSplits:
    """Create a stratified, patient-level train/val/test split.

    Args:
        df: long-format DataFrame (one row per patient-hour), must contain
            `patient_id` and `SepsisLabel` columns.
        train_frac: fraction of PATIENTS (not rows) assigned to train.
        val_frac: fraction assigned to validation. Remaining goes to test.
        random_state: fixed seed for reproducibility.

    Returns:
        PatientSplits with disjoint patient_id lists for train/val/test.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave a nonzero test fraction")

    patient_labels = _patient_level_sepsis_label(df)
    patient_ids = patient_labels.index.to_numpy()
    strata = patient_labels.to_numpy()

    # First split off train, then split the remainder into val/test.
    train_ids, remainder_ids, strata_train, strata_remainder = train_test_split(
        patient_ids,
        strata,
        train_size=train_frac,
        stratify=strata,
        random_state=random_state,
    )

    # val_frac was defined relative to the WHOLE dataset, so convert it to a
    # fraction of the remainder before the second split.
    remainder_frac = 1.0 - train_frac
    val_frac_of_remainder = val_frac / remainder_frac

    val_ids, test_ids = train_test_split(
        remainder_ids,
        train_size=val_frac_of_remainder,
        stratify=strata_remainder,
        random_state=random_state,
    )

    return PatientSplits(
        train_ids=sorted(train_ids.tolist()),
        val_ids=sorted(val_ids.tolist()),
        test_ids=sorted(test_ids.tolist()),
    )


def apply_split(df: pd.DataFrame, patient_ids: list) -> pd.DataFrame:
    """Return only the rows belonging to the given patient IDs."""
    return df[df["patient_id"].isin(patient_ids)].copy()


def assert_no_patient_overlap(splits: PatientSplits) -> None:
    """Hard safety check: raises if any patient appears in more than one split.

    This is the single most important invariant in the whole pipeline (QR1
    in the project specification) — call this in a test, and ideally also
    at the start of any training script, right after loading the splits.
    """
    train_set = set(splits.train_ids)
    val_set = set(splits.val_ids)
    test_set = set(splits.test_ids)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise AssertionError(
            "Patient overlap detected between splits — this is a data leakage "
            f"bug. train/val: {overlap_train_val}, train/test: {overlap_train_test}, "
            f"val/test: {overlap_val_test}"
        )


if __name__ == "__main__":
    # Quick manual check — run with: python -m src.data.splits
    from src.data.loader import load_challenge_dataset

    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    df = load_challenge_dataset(root)

    splits = make_patient_splits(df)
    assert_no_patient_overlap(splits)

    print(f"Train: {len(splits.train_ids)} patients")
    print(f"Val:   {len(splits.val_ids)} patients")
    print(f"Test:  {len(splits.test_ids)} patients")

    out_path = Path(__file__).resolve().parents[2] / "configs" / "splits.json"
    splits.save(out_path)
    print(f"Saved splits to {out_path}")
