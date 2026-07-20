"""
Decision curve analysis on the trained, calibrated LightGBM model —
translates technical metrics into clinical net benefit (project
specification O7).

Calibration follows the established project convention: fit on val,
evaluated on test. Uses isotonic calibration by default, consistent with
Phases 3-6.

Usage:
    python -m src.decision_curve_analysis
"""

import argparse
import json
from pathlib import Path

from src.calibration.methods import fit_calibrator
from src.data.features import engineer_features
from src.data.loader import load_challenge_dataset
from src.data.preprocess import preprocess
from src.data.splits import PatientSplits, apply_split, assert_no_patient_overlap
from src.evaluation.decision_curve import decision_curve_data, summarize_decision_curve
from src.models.base import get_feature_columns
from src.models.lightgbm_model import LightGBMSepsisModel


def run_decision_curve_analysis(
    data_root: Path, splits_path: Path, model_path: Path, calibration_method: str
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
    feature_cols = get_feature_columns(val_df)

    val_prob_raw = model.predict_proba(val_df[feature_cols])
    test_prob_raw = model.predict_proba(test_df[feature_cols])
    val_labels = val_df["SepsisLabel"].to_numpy()
    test_labels = test_df["SepsisLabel"].to_numpy()

    print(f"Fitting '{calibration_method}' calibrator on validation set ...")
    calibrator = fit_calibrator(calibration_method, val_prob_raw, val_labels)
    test_prob_calibrated = calibrator.transform(test_prob_raw)

    print("\nComputing decision curve on TEST set (calibrated probabilities) ...")
    curve = decision_curve_data(test_labels, test_prob_calibrated)
    summary = summarize_decision_curve(curve)

    prevalence = float(test_labels.mean())
    print(f"\nTest set sepsis prevalence: {prevalence:.4f} "
          f"({int(test_labels.sum())} of {len(test_labels)} patient-hours)")

    print("\n--- Decision curve summary ---")
    print(json.dumps(summary, indent=2))

    print(
        "\nInterpretation: 'useful_threshold_range' is the range of alert "
        "thresholds where the model provides more net benefit than both "
        "treating every patient-hour as high-risk and never alerting at "
        "all. Thresholds outside this range mean the model is not adding "
        "clinical value there, regardless of what AUROC or utility_score "
        "say — this is what makes decision curve analysis a distinct, "
        "necessary complement to those metrics rather than a repeat of them."
    )

    if summary["useful_threshold_range"] is not None:
        lo, hi = summary["useful_threshold_range"]
        print(f"\nModel is clinically useful for alert thresholds roughly "
              f"in [{lo:.3f}, {hi:.3f}]. For context, this project's earlier "
              f"threshold selection on val (Phase 3) chose a threshold of "
              f"around 0.03 — check whether that choice falls inside this range.")

    # A few illustrative rows from the curve, spread across the range —
    # useful for a compact table in the dissertation without dumping all 100 rows.
    sample_rows = curve.iloc[:: max(1, len(curve) // 10)]
    print("\n--- Sample of the decision curve (every ~10th threshold) ---")
    print(sample_rows.to_string(index=False))

    return {"curve": curve, "summary": summary, "prevalence": prevalence}


def main():
    parser = argparse.ArgumentParser(description="Decision curve analysis for the sepsis model.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/lightgbm_baseline.txt"))
    parser.add_argument("--calibration-method", choices=["platt", "temperature", "isotonic"], default="isotonic")
    args = parser.parse_args()

    run_decision_curve_analysis(args.data_root, args.splits, args.model, args.calibration_method)


if __name__ == "__main__":
    main()
