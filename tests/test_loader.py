"""
Example unit test for the data pipeline.

Run with: pytest tests/

This is intentionally minimal — it exists as a template. Expand it once
src/data/loader.py and src/data/preprocess.py are filled in. In particular,
add a test that asserts no patient_id appears in more than one split
(this directly protects Quality Requirement QR1 in the project specification).
"""

import pandas as pd

from src.data.loader import load_patient_file


def test_load_patient_file_adds_patient_id(tmp_path):
    # Create a tiny fake .psv file to test against, so this test does not
    # depend on the real dataset being downloaded.
    fake_file = tmp_path / "p00001.psv"
    fake_file.write_text("HR|O2Sat|SepsisLabel\n80|98|0\n85|97|0\n")

    df = load_patient_file(fake_file)

    assert "patient_id" in df.columns
    assert (df["patient_id"] == "p00001").all()
    assert len(df) == 2


def test_no_patient_overlap_between_splits():
    """
    Placeholder for the most important pipeline test in this project (QR1):
    no patient_id should ever appear in both train and test splits.
    Fill this in once train/test splitting is implemented.
    """
    pass
