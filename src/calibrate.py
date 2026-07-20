"""
Apply post-hoc calibration to the trained LightGBM baseline and report the
before/after improvement in calibration quality.

Calibration is fit on the VALIDATION split (already held out from
training) and evaluated on the TEST split (never touched until now) — see
src/calibration/methods.py for why this ordering matters.

Usage:
    python -m src.calibrate
    python -m src.calibrate --method isotonic
"""

import argparse
import json
from pathlib import Path

from src.calibration.methods import fit_calibrator
from src.data.features import engineer_features
from src.data.loader import load_challenge_dataset
from src.data.preprocess import preprocess
from src.data.splits import PatientSplits, apply_split, assert_no_patient_overlap
from src.evaluation.calibration_metrics import (
    brier_score,
    expected_calibration_error,
)
from src.evaluation.metrics import evaluate_predictions, find_best_threshold
from src.models.base import get_feature_columns
from src.models.lightgbm_model import LightGBMSepsisModel


def run_calibration(
    data_root: Path,
    splits_path: Path,
    model_path: Path,
    method: str,
    threshold: float = 0.5,
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

    # --- BEFORE calibration ---
    ece_before = expected_calibration_error(test_labels, test_prob_raw)
    brier_before = brier_score(test_labels, test_prob_raw)

    # --- Fit calibrator on val, apply to test ---
    print(f"Fitting '{method}' calibrator on validation set ...")
    calibrator = fit_calibrator(method, val_prob_raw, val_labels)
    test_prob_calibrated = calibrator.transform(test_prob_raw)

    # --- AFTER calibration ---
    ece_after = expected_calibration_error(test_labels, test_prob_calibrated)
    brier_after = brier_score(test_labels, test_prob_calibrated)

    print("\n--- Calibration quality on TEST set (never used for fitting) ---")
    print(json.dumps({
        "method": method,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "brier_before": brier_before,
        "brier_after": brier_after,
    }, indent=2))

    print("\n--- Full evaluation (AUROC/AUPRC/utility) on TEST, before calibration ---")
    metrics_before = evaluate_predictions(test_df, test_prob_raw, threshold=threshold)
    print(json.dumps(metrics_before, indent=2))

    print("\n--- Full evaluation (AUROC/AUPRC/utility) on TEST, after calibration ---")
    metrics_after = evaluate_predictions(test_df, test_prob_calibrated, threshold=threshold)
    print(json.dumps(metrics_after, indent=2))

    # --- Threshold selection: chosen on VAL (calibrated), applied to TEST ---
    # This directly addresses a real finding from development: after honest
    # calibration, well-calibrated probabilities for a rare event rarely
    # cross 0.5, so the default threshold can make a well-calibrated model
    # look artificially useless. The threshold itself is a choice, made
    # here on val only, then applied once to test — never tuned on test.
    print(f"\nSearching for the best decision threshold on VAL (calibrated) ...")
    val_prob_calibrated = calibrator.transform(val_prob_raw)
    best_threshold, best_val_utility = find_best_threshold(val_df, val_prob_calibrated)
    print(f"Best threshold found on val: {best_threshold:.3f} "
          f"(val utility_normalized = {best_val_utility:.4f})")

    print(f"\n--- Full evaluation on TEST, calibrated probabilities, "
          f"threshold chosen on val ({best_threshold:.3f}) ---")
    metrics_best_threshold = evaluate_predictions(
        test_df, test_prob_calibrated, threshold=best_threshold
    )
    print(json.dumps(metrics_best_threshold, indent=2))

    # --- Fair comparison: does calibration add anything BEYOND what
    # optimal thresholding of the raw, uncalibrated scores already gives?
    # Without this, "calibrated + tuned threshold" vs "raw + threshold=0.5"
    # is not an apples-to-apples comparison — the raw scores deserve the
    # same threshold-tuning opportunity before we credit calibration with
    # any utility gain.
    print(f"\nSearching for the best decision threshold on VAL (RAW, uncalibrated) ...")
    raw_best_threshold, raw_best_val_utility = find_best_threshold(val_df, val_prob_raw)
    print(f"Best threshold found on val (raw): {raw_best_threshold:.3f} "
          f"(val utility_normalized = {raw_best_val_utility:.4f})")

    print(f"\n--- Full evaluation on TEST, RAW probabilities, "
          f"threshold chosen on val ({raw_best_threshold:.3f}) ---")
    metrics_raw_best_threshold = evaluate_predictions(
        test_df, test_prob_raw, threshold=raw_best_threshold
    )
    print(json.dumps(metrics_raw_best_threshold, indent=2))

    print(
        "\n--- Summary: utility_normalized under each condition ---\n"
        f"  Raw, threshold=0.5:                  {metrics_before['utility_normalized']:.4f}\n"
        f"  Raw, best threshold (tuned on val):   {metrics_raw_best_threshold['utility_normalized']:.4f}\n"
        f"  Calibrated, threshold=0.5:            {metrics_after['utility_normalized']:.4f}\n"
        f"  Calibrated, best threshold (val):     {metrics_best_threshold['utility_normalized']:.4f}\n"
        "\nIf the last two rows are close to each other but both clearly "
        "beat the first row, the main lesson is: THRESHOLD SELECTION matters "
        "more than calibration for this metric. Calibration's value lies "
        "elsewhere — in the honesty of the probability itself (ECE/Brier), "
        "which matters for decision curve analysis and conformal prediction "
        "in later phases, not for utility-at-optimal-threshold specifically."
    )

    print(
        "\nNote: AUROC/AUPRC are threshold-independent and should be "
        "IDENTICAL before/after calibration (calibration only rescales "
        "probabilities, it never changes their rank order for Platt/"
        "temperature scaling; isotonic regression can occasionally tie "
        "values and change AUROC by a negligible amount). If AUROC "
        "changes substantially, something is wrong."
    )

    return {
        "ece_before": ece_before, "ece_after": ece_after,
        "brier_before": brier_before, "brier_after": brier_after,
        "metrics_before": metrics_before, "metrics_after": metrics_after,
        "best_threshold": best_threshold, "metrics_best_threshold": metrics_best_threshold,
        "raw_best_threshold": raw_best_threshold,
        "metrics_raw_best_threshold": metrics_raw_best_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate the trained sepsis model.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/lightgbm_baseline.txt"))
    parser.add_argument("--method", choices=["platt", "temperature", "isotonic"], default="temperature")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_calibration(args.data_root, args.splits, args.model, args.method, args.threshold)


if __name__ == "__main__":
    main()
