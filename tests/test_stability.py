"""
Unit tests for src/evaluation/stability.py.

Run with: pytest tests/test_stability.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stability import (
    consecutive_hour_stability,
    cosine_similarity,
    top_k_overlap,
)


def test_cosine_similarity_is_1_for_identical_vectors():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_is_minus_1_for_opposite_vectors():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_cosine_similarity_is_0_for_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_handles_zero_vector_without_crashing():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert np.isnan(cosine_similarity(a, b))


def test_top_k_overlap_is_1_for_identical_rankings():
    row = pd.Series({"a": 5.0, "b": 3.0, "c": 1.0, "d": 0.1})
    assert top_k_overlap(row, row, k=2) == pytest.approx(1.0)


def test_top_k_overlap_is_0_for_completely_disjoint_top_k():
    row_a = pd.Series({"a": 5.0, "b": 4.0, "c": 0.1, "d": 0.1})
    row_b = pd.Series({"a": 0.1, "b": 0.1, "c": 5.0, "d": 4.0})
    assert top_k_overlap(row_a, row_b, k=2) == pytest.approx(0.0)


def test_top_k_overlap_partial_match():
    row_a = pd.Series({"a": 5.0, "b": 4.0, "c": 0.1})
    row_b = pd.Series({"a": 5.0, "b": 0.1, "c": 4.0})
    # top-2 of a: {a, b}; top-2 of b: {a, c} -> overlap = {a} -> 1/2
    assert top_k_overlap(row_a, row_b, k=2) == pytest.approx(0.5)


def test_consecutive_hour_stability_computes_expected_number_of_pairs():
    # Patient p1: 4 hours -> 3 consecutive pairs. Patient p2: 3 hours -> 2 pairs.
    shap_df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(7, 3)),
        columns=["f1", "f2", "f3"],
    )
    patient_ids = pd.Series(["p1", "p1", "p1", "p1", "p2", "p2", "p2"])
    hours = pd.Series([1, 2, 3, 4, 1, 2, 3])

    result = consecutive_hour_stability(shap_df, patient_ids, hours, k=2)

    assert len(result) == 3 + 2


def test_consecutive_hour_stability_skips_non_consecutive_gaps():
    """If a patient's hours have a gap (e.g. hour 3 missing), that pair
    must be skipped, not silently compared as if it were consecutive.
    """
    shap_df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(3, 2)),
        columns=["f1", "f2"],
    )
    patient_ids = pd.Series(["p1", "p1", "p1"])
    hours = pd.Series([1, 2, 5])  # gap between hour 2 and hour 5

    result = consecutive_hour_stability(shap_df, patient_ids, hours, k=1)

    # Only the (1,2) pair is truly consecutive; (2,5) must be skipped.
    assert len(result) == 1
    assert result.iloc[0]["hour"] == 1


def test_identical_consecutive_explanations_give_perfect_stability():
    shap_df = pd.DataFrame({"f1": [1.0, 1.0], "f2": [2.0, 2.0]})
    patient_ids = pd.Series(["p1", "p1"])
    hours = pd.Series([1, 2])

    result = consecutive_hour_stability(shap_df, patient_ids, hours, k=2)

    assert result.iloc[0]["cosine_similarity"] == pytest.approx(1.0)
    assert result.iloc[0]["top_k_overlap"] == pytest.approx(1.0)


def test_single_hour_patient_produces_no_pairs():
    shap_df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    patient_ids = pd.Series(["p1"])
    hours = pd.Series([1])

    result = consecutive_hour_stability(shap_df, patient_ids, hours, k=1)

    assert len(result) == 0
