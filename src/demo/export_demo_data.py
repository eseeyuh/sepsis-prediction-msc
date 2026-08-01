"""
Step 1 of the interactive ICU demonstration: select illustrative patients
from the held-out test split and export their hour-by-hour model outputs
to a single JSON file for browser playback.

WHY PRECOMPUTE OFFLINE: the trained LightGBM booster, the isotonic
calibrator, and the conformal qhat all live in Python. Rather than
standing up a server for a demonstration (a fragile dependency during a
viva), everything is computed once here using the SAME pipeline modules
that produced the dissertation's reported results, then replayed by a
static browser page. This also guarantees the demo shows honest, already-
validated numbers rather than a parallel simplified reimplementation.

HONESTY OF THE DEMO DATA: patients are drawn exclusively from the TEST
split — never seen during training, probability calibration, or conformal
calibration. The model, calibrator, and qhat are all fit on train/val
only, exactly as in every earlier phase.

WHAT THIS IS NOT: a live clinical system, or additional validation
evidence. It replays historical, de-identified data at accelerated speed
purely to communicate how the system behaves. The dissertation should
present it as a communication artefact, kept separate from the Results
chapter's evidence.

Usage:
    python -m src.demo.export_demo_data
    python -m src.demo.export_demo_data --alpha 0.05 --out outputs/demo_data.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.methods import fit_calibrator
from src.data.features import engineer_features
from src.data.loader import load_challenge_dataset
from src.data.preprocess import preprocess
from src.data.splits import PatientSplits, apply_split, assert_no_patient_overlap
from src.explainability.shap_explainer import SHAPExplainer
from src.models.base import get_feature_columns
from src.models.lightgbm_model import LightGBMSepsisModel
from src.uncertainty.aps import build_prediction_sets_aps, calibrate_conformal_aps

# Vitals shown on the demo dashboard. Deliberately a small, familiar set —
# a bedside monitor shows a handful of numbers, not 40 lab values.
DISPLAY_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]

# Human-readable names for the features most likely to appear in the top-3
# SHAP explanations. Anything not listed falls back to a generated label,
# so this map never needs to be exhaustive.
FEATURE_DISPLAY_NAMES = {
    "HR": "Heart rate",
    "SBP": "Systolic blood pressure",
    "MAP": "Mean arterial pressure",
    "DBP": "Diastolic blood pressure",
    "Resp": "Respiratory rate",
    "Temp": "Temperature",
    "O2Sat": "Oxygen saturation",
    "Lactate": "Lactate",
    "WBC": "White blood cell count",
    "Creatinine": "Creatinine",
    "BUN": "Blood urea nitrogen",
    "FiO2": "Inspired oxygen fraction",
    "Platelets": "Platelet count",
    "shock_index": "Shock index (HR/SBP)",
    "ICULOS": "Hours in ICU",
    "hours_since_admission": "Hours since ICU admission",
    "HospAdmTime": "Time from hospital to ICU admission",
    "Unit1": "ICU unit type (MICU)",
    "Unit2": "ICU unit type (SICU)",
    "Age": "Age",
    "Gender": "Sex",
}


def humanise_feature_name(feature: str) -> str:
    """Turn an engineered feature name into something a clinician could
    read on a dashboard. Handles the three engineered patterns from
    src/data/features.py (rolling stats, trends, missingness flags) and
    falls back gracefully for anything unmapped.
    """
    if feature in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature]

    if feature.endswith("_missing"):
        base = feature[: -len("_missing")]
        return f"{FEATURE_DISPLAY_NAMES.get(base, base)} not measured"

    if "_trend" in feature:
        base, _, window = feature.partition("_trend")
        return f"{FEATURE_DISPLAY_NAMES.get(base, base)} trend ({window})"

    if "_roll" in feature:
        base, _, rest = feature.partition("_roll")
        window, _, stat = rest.partition("_")
        stat_names = {"mean": "average", "min": "minimum", "max": "maximum", "std": "variability"}
        return f"{FEATURE_DISPLAY_NAMES.get(base, base)} {stat_names.get(stat, stat)} ({window})"

    return feature.replace("_", " ")


def derive_onset_hour(patient_df: pd.DataFrame) -> float:
    """Recover t_sepsis from the hourly SepsisLabel, using the same
    convention as src/evaluation/metrics.py: the Challenge shifts labels
    6 hours ahead of clinical onset, so t_sepsis = first labelled hour + 6.
    Returns None for non-septic patients.
    """
    septic = patient_df[patient_df["SepsisLabel"] == 1]
    if len(septic) == 0:
        return None
    return float(septic["ICULOS"].min()) + 6.0


def _first_crossing_hour(hours: np.ndarray, probs: np.ndarray, threshold: float):
    """First hour at which calibrated risk crosses the operating threshold."""
    crossed = np.where(probs >= threshold)[0]
    return float(hours[crossed[0]]) if len(crossed) else None


def score_candidates(
    test_df: pd.DataFrame,
    probs_calibrated: np.ndarray,
    prediction_sets: np.ndarray,
    threshold: float,
    min_hours: int = 12,
    max_hours: int = 72,
) -> pd.DataFrame:
    """Build a per-patient summary table used to pick demo cases
    algorithmically rather than by eyeballing thousands of patients.

    Length bounds exist purely for demo watchability: at 15 seconds per
    hour, a 200-hour stay would take 50 minutes to play back.
    """
    working = test_df[["patient_id", "ICULOS", "SepsisLabel"]].copy()
    working["prob"] = probs_calibrated
    working["set_size"] = prediction_sets.sum(axis=1)

    records = []
    for patient_id, group in working.groupby("patient_id"):
        group = group.sort_values("ICULOS")
        hours = group["ICULOS"].to_numpy()
        probs = group["prob"].to_numpy()

        n_hours = len(group)
        if not (min_hours <= n_hours <= max_hours):
            continue

        onset_hour = derive_onset_hour(group)
        alert_hour = _first_crossing_hour(hours, probs, threshold)

        lead_time = None
        if onset_hour is not None and alert_hour is not None:
            lead_time = onset_hour - alert_hour  # positive = alerted before onset

        records.append({
            "patient_id": patient_id,
            "n_hours": n_hours,
            "is_septic": onset_hour is not None,
            "onset_hour": onset_hour,
            "first_alert_hour": alert_hour,
            "lead_time_hours": lead_time,
            "max_prob": float(probs.max()),
            "final_prob": float(probs[-1]),
            "n_ambiguous_hours": int((group["set_size"] == 2).sum()),
            "n_alert_hours": int((probs >= threshold).sum()),
        })

    return pd.DataFrame(records)


def select_demo_patients(candidates: pd.DataFrame) -> dict:
    """Pick one patient per demonstration scenario.

    Case D (a visible failure) is included deliberately: a demo showing
    only successes invites the reasonable objection that cases were
    cherry-picked. Showing where the system fails, clearly labelled, is
    consistent with how limitations are reported throughout this project.
    """
    selected = {}

    # Case A — early correct warning: alerted between 3 and 12 hours before
    # onset. Prefer the clearest lead time within that clinically useful band.
    case_a = candidates[
        candidates["is_septic"]
        & candidates["lead_time_hours"].notna()
        & (candidates["lead_time_hours"] >= 3)
        & (candidates["lead_time_hours"] <= 12)
    ].sort_values("max_prob", ascending=False)
    if len(case_a):
        selected["early_warning"] = case_a.iloc[0]["patient_id"]

    # Case B — genuine uncertainty: the most ambiguous-hour-heavy patient,
    # showing the "defer to clinician" behaviour APS makes possible.
    case_b = candidates[candidates["n_ambiguous_hours"] > 0].sort_values(
        "n_ambiguous_hours", ascending=False
    )
    if len(case_b):
        selected["uncertain"] = case_b.iloc[0]["patient_id"]

    # Case C — quiet, stable, non-septic patient with no false alarms.
    case_c = candidates[
        (~candidates["is_septic"]) & (candidates["n_alert_hours"] == 0)
    ].sort_values("n_hours", ascending=False)
    if len(case_c):
        selected["stable"] = case_c.iloc[0]["patient_id"]

    # Case D — an honest failure. Prefer a missed septic patient (never
    # alerted); fall back to the worst false-alarm case if none qualifies.
    missed = candidates[
        candidates["is_septic"] & candidates["first_alert_hour"].isna()
    ].sort_values("n_hours", ascending=False)
    false_alarm = candidates[
        (~candidates["is_septic"]) & (candidates["n_alert_hours"] > 0)
    ].sort_values("n_alert_hours", ascending=False)

    if len(missed):
        selected["failure_missed"] = missed.iloc[0]["patient_id"]
    elif len(false_alarm):
        selected["failure_false_alarm"] = false_alarm.iloc[0]["patient_id"]

    # Deliberately no exception if a scenario finds no candidate — the
    # caller reports which scenarios were filled, so a missing case is
    # visible rather than silently substituted with an unsuitable patient.
    return selected


def build_patient_record(
    patient_id: str,
    test_df: pd.DataFrame,
    probs_raw: np.ndarray,
    probs_calibrated: np.ndarray,
    prediction_sets: np.ndarray,
    shap_df: pd.DataFrame,
    threshold: float,
    top_k_explanations: int = 3,
    min_shap_magnitude: float = 1e-4,
) -> dict:
    """Assemble one patient's full hour-by-hour playback record.

    `min_shap_magnitude` drops features whose contribution is effectively
    zero. Without it, a confidently-negative hour (most hours, since
    sepsis is rare) pads its top-3 with features whose SHAP value is
    ~1e-16 — the dashboard would then display "Heart rate increases risk"
    for a feature that did nothing, which is worse than showing fewer
    reasons. An hour with no meaningful drivers correctly yields an empty
    explanation list, and the page can say so plainly.
    """
    mask = (test_df["patient_id"] == patient_id).to_numpy()
    patient_df = test_df[mask].sort_values("ICULOS")
    order = patient_df.index

    patient_probs_raw = probs_raw[mask]
    patient_probs_cal = probs_calibrated[mask]
    patient_sets = prediction_sets[mask]
    patient_shap = shap_df.loc[order]

    hours = []
    for i, (idx, row) in enumerate(patient_df.iterrows()):
        shap_row = patient_shap.loc[idx]
        ranked = shap_row.abs().sort_values(ascending=False).head(top_k_explanations)
        ranked = ranked[ranked >= min_shap_magnitude]

        explanations = [
            {
                "feature": feat,
                "label": humanise_feature_name(feat),
                "shap_value": float(shap_row[feat]),
                # Direction is what a clinician reads first: is this
                # pushing the alert up or down?
                "direction": "increases risk" if shap_row[feat] > 0 else "decreases risk",
                "feature_value": (
                    float(row[feat]) if feat in row.index and pd.notna(row[feat]) else None
                ),
            }
            for feat in ranked.index
        ]

        includes_negative, includes_positive = bool(patient_sets[i][0]), bool(patient_sets[i][1])
        set_size = int(includes_negative) + int(includes_positive)
        if set_size == 2:
            uncertainty_state = "uncertain"
        elif set_size == 0:
            uncertainty_state = "no_confident_class"
        elif includes_positive:
            uncertainty_state = "confident_high_risk"
        else:
            uncertainty_state = "confident_low_risk"

        hours.append({
            "iculos": int(row["ICULOS"]),
            "vitals": {
                v: (float(row[v]) if v in row.index and pd.notna(row[v]) else None)
                for v in DISPLAY_VITALS
            },
            "prob_raw": float(patient_probs_raw[i]),
            "prob_calibrated": float(patient_probs_cal[i]),
            "above_threshold": bool(patient_probs_cal[i] >= threshold),
            "uncertainty_state": uncertainty_state,
            "prediction_set_size": set_size,
            "explanations": explanations,
            # Ground truth is exported so the page can REVEAL it after
            # playback finishes — it must not be displayed live, or the
            # demo stops representing what the system actually knew at
            # each hour.
            "sepsis_label": int(row["SepsisLabel"]),
        })

    onset_hour = derive_onset_hour(patient_df)
    return {
        "patient_id": patient_id,
        "n_hours": len(hours),
        "is_septic": onset_hour is not None,
        "onset_hour": onset_hour,
        "hours": hours,
    }


def export_demo_data(
    data_root: Path,
    splits_path: Path,
    model_path: Path,
    out_path: Path,
    calibration_method: str,
    alpha: float,
    threshold: float,
):
    print(f"Loading raw data from {data_root} ...")
    df = load_challenge_dataset(data_root)

    splits = PatientSplits.load(splits_path)
    assert_no_patient_overlap(splits)

    train_df = apply_split(df, splits.train_ids)
    val_df = apply_split(df, splits.val_ids)
    test_df = apply_split(df, splits.test_ids)

    print("Preprocessing (imputation stats fit on train only) ...")
    train_df, stats = preprocess(train_df, fit_stats=True)
    val_df, _ = preprocess(val_df, stats=stats)
    test_df, _ = preprocess(test_df, stats=stats)

    print("Engineering features ...")
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    print(f"Loading trained model from {model_path} ...")
    model = LightGBMSepsisModel.load(model_path)
    feature_cols = get_feature_columns(test_df)

    val_prob_raw = model.predict_proba(val_df[feature_cols])
    test_prob_raw = model.predict_proba(test_df[feature_cols])
    val_labels = val_df["SepsisLabel"].to_numpy()

    print(f"Calibrating ('{calibration_method}', fit on val) ...")
    calibrator = fit_calibrator(calibration_method, val_prob_raw, val_labels)
    val_prob_cal = calibrator.transform(val_prob_raw)
    test_prob_cal = calibrator.transform(test_prob_raw)

    # APS rather than LAC: Phase 4 found LAC never produces genuine
    # two-class ("uncertain") sets on this data, so it cannot demonstrate
    # the abstention behaviour the demo is meant to show.
    print(f"Computing conformal prediction sets (APS, alpha={alpha}) ...")
    qhat = calibrate_conformal_aps(val_prob_cal, val_labels, alpha=alpha, seed=42)
    test_sets = build_prediction_sets_aps(test_prob_cal, qhat, seed=123)

    print("Scoring candidate patients ...")
    candidates = score_candidates(test_df, test_prob_cal, test_sets, threshold)
    print(f"  {len(candidates)} patients met the demo length bounds "
          f"(out of {test_df['patient_id'].nunique()} in the test split).")

    selected = select_demo_patients(candidates)
    print("\nSelected demo cases:")
    for scenario, patient_id in selected.items():
        row = candidates[candidates["patient_id"] == patient_id].iloc[0]
        print(f"  {scenario:22s} -> {patient_id} "
              f"({row['n_hours']} hours, septic={row['is_septic']}, "
              f"lead_time={row['lead_time_hours']}, "
              f"ambiguous_hours={row['n_ambiguous_hours']})")

    expected_scenarios = {"early_warning", "uncertain", "stable"}
    missing = expected_scenarios - set(selected)
    if missing:
        print(f"\n  NOTE: no candidate found for: {', '.join(sorted(missing))}. "
              f"Try relaxing --threshold, --alpha, or the length bounds.")

    print("\nComputing SHAP values for the selected patients only ...")
    selected_ids = list(selected.values())
    selected_mask = test_df["patient_id"].isin(selected_ids)
    explainer = SHAPExplainer(model)
    shap_selected = explainer.explain(test_df.loc[selected_mask, feature_cols])

    # Re-index the per-row arrays down to the selected patients so
    # build_patient_record can use a single consistent mask.
    test_subset = test_df[selected_mask]
    subset_positions = np.where(selected_mask.to_numpy())[0]
    probs_raw_subset = test_prob_raw[subset_positions]
    probs_cal_subset = test_prob_cal[subset_positions]
    sets_subset = test_sets[subset_positions]

    print("Building playback records ...")
    patients = {}
    for scenario, patient_id in selected.items():
        patients[scenario] = build_patient_record(
            patient_id, test_subset, probs_raw_subset, probs_cal_subset,
            sets_subset, shap_selected, threshold,
        )

    payload = {
        "metadata": {
            "description": (
                "Precomputed playback data for the ICU demonstration dashboard. "
                "Replays historical, de-identified PhysioNet/CinC Challenge 2019 "
                "test-split records at accelerated speed. NOT a live clinical system."
            ),
            "source": "PhysioNet/CinC Challenge 2019, held-out test split",
            "calibration_method": calibration_method,
            "conformal_method": "APS",
            "conformal_alpha": alpha,
            "conformal_qhat": qhat,
            "operating_threshold": threshold,
            "display_vitals": DISPLAY_VITALS,
        },
        "patients": patients,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)

    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {len(patients)} demo patients to {out_path} ({size_kb:.0f} KB)")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Export demo playback data.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/lightgbm_baseline.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/demo_data.json"))
    parser.add_argument("--calibration-method", choices=["platt", "temperature", "isotonic"], default="isotonic")
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Conformal miscoverage rate. Phase 4 found ambiguous "
                              "sets are most frequent at the strictest alpha (0.05).")
    parser.add_argument("--threshold", type=float, default=0.03,
                         help="Operating threshold. Default matches the value selected "
                              "on the validation set in Phase 3.")
    args = parser.parse_args()

    export_demo_data(
        args.data_root, args.splits, args.model, args.out,
        args.calibration_method, args.alpha, args.threshold,
    )


if __name__ == "__main__":
    main()
