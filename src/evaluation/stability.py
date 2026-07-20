"""
Stability of SHAP explanations across consecutive hours for the same
patient.

Faithfulness (src/evaluation/faithfulness.py) asks "is this explanation
correct?". Stability asks a different, complementary question: "does the
explanation change wildly from one hour to the next, even when the
patient's underlying clinical picture hasn't changed much?" An
explanation that flickers between contradictory "top reasons" hour to
hour is hard for a clinician to trust and act on, independent of whether
each individual hour's explanation is technically faithful — see project
specification O5.
"""

import numpy as np
import pandas as pd


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.
    Applied here to two SHAP vectors to measure how similarly two hours'
    explanations are shaped, independent of overall magnitude.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def top_k_overlap(shap_row_a: pd.Series, shap_row_b: pd.Series, k: int = 5) -> float:
    """Fraction of the top-k most important features (by |SHAP value|) at
    hour A that are still in the top-k at hour B. 1.0 = identical top-k
    sets, 0.0 = completely different.
    """
    top_a = set(shap_row_a.abs().sort_values(ascending=False).head(k).index)
    top_b = set(shap_row_b.abs().sort_values(ascending=False).head(k).index)
    return len(top_a & top_b) / k


def consecutive_hour_stability(
    shap_df: pd.DataFrame,
    patient_ids: pd.Series,
    hours: pd.Series,
    k: int = 5,
) -> pd.DataFrame:
    """For every pair of CONSECUTIVE hours (hour gap of exactly 1) within
    each patient, compute cosine similarity and top-k overlap between
    their SHAP explanations.

    Non-consecutive hour pairs (e.g. if a row was dropped somewhere) are
    deliberately skipped rather than compared, since a stability claim
    only makes sense between genuinely adjacent hours.

    Returns one row per consecutive-hour pair, indexed by patient and the
    earlier hour of the pair.
    """
    feature_cols = list(shap_df.columns)

    working = shap_df.copy()
    working["_patient_id"] = patient_ids.to_numpy()
    working["_hour"] = hours.to_numpy()
    working = working.sort_values(["_patient_id", "_hour"]).reset_index(drop=True)

    records = []
    for patient_id, group in working.groupby("_patient_id"):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            hour_a = group.loc[i, "_hour"]
            hour_b = group.loc[i + 1, "_hour"]
            if hour_b - hour_a != 1:
                continue  # not truly consecutive — skip rather than misrepresent

            row_a = group.loc[i, feature_cols]
            row_b = group.loc[i + 1, feature_cols]

            records.append({
                "patient_id": patient_id,
                "hour": hour_a,
                "cosine_similarity": cosine_similarity(row_a.to_numpy(dtype=float), row_b.to_numpy(dtype=float)),
                "top_k_overlap": top_k_overlap(row_a, row_b, k=k),
            })

    return pd.DataFrame(records)
