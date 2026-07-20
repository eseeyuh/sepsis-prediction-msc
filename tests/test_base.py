import pandas as pd

from src.models.base import get_feature_columns


def test_get_feature_columns_excludes_reserved_columns():
    df = pd.DataFrame({
        "patient_id": ["p1"],
        "SepsisLabel": [0],
        "hospital": ["A"],
        "HR": [80],
        "Lactate": [1.2],
    })
    cols = get_feature_columns(df)

    assert "patient_id" not in cols
    assert "SepsisLabel" not in cols
    assert "hospital" not in cols
    assert "HR" in cols
    assert "Lactate" in cols


def test_get_feature_columns_is_sorted_and_reproducible():
    df = pd.DataFrame({
        "z_col": [1], "a_col": [2],
        "patient_id": ["p1"], "SepsisLabel": [0], "hospital": ["A"],
    })
    cols = get_feature_columns(df)
    assert cols == sorted(cols)
    assert cols == ["a_col", "z_col"]
