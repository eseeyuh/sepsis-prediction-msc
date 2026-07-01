"""
Loader for the PhysioNet/CinC Challenge 2019 dataset.

Each patient is stored as a single .psv (pipe-separated values) file with one
row per ICU hour. This module reads individual patient files and assembles
them into a single long-format DataFrame for downstream preprocessing.
"""

from pathlib import Path

import pandas as pd


def load_patient_file(filepath: Path) -> pd.DataFrame:
    """Load a single patient .psv file.

    Adds a `patient_id` column derived from the filename (e.g. 'p00001.psv' -> 'p00001')
    so that patient-level grouping and splitting is easy downstream.
    """
    df = pd.read_csv(filepath, sep="|")
    df["patient_id"] = filepath.stem
    return df


def load_hospital_system(raw_dir: Path, pattern: str = "*.psv") -> pd.DataFrame:
    """Load every patient file in a hospital-system directory (e.g. training_setA).

    Returns a single long-format DataFrame: one row per patient-hour.
    """
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No .psv files found in {raw_dir}. "
            "Check data/README.md for download instructions."
        )

    frames = [load_patient_file(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def load_challenge_dataset(data_root: Path) -> pd.DataFrame:
    """Load both hospital systems (A and B) and tag each row with its source hospital.

    `data_root` should point to the `data/raw/` directory containing
    `training_setA/` and `training_setB/`.
    """
    set_a = load_hospital_system(data_root / "training_setA")
    set_a["hospital"] = "A"

    set_b = load_hospital_system(data_root / "training_setB")
    set_b["hospital"] = "B"

    return pd.concat([set_a, set_b], ignore_index=True)


if __name__ == "__main__":
    # Quick manual check — run with: python -m src.data.loader
    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    df = load_challenge_dataset(root)
    print(f"Loaded {df['patient_id'].nunique()} patients, {len(df)} patient-hours")
    print(df.head())
