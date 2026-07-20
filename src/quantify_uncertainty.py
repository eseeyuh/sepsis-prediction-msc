"""
Apply split conformal prediction to the trained, calibrated LightGBM model.

Calibration (in the conformal sense — computing qhat) is fit on the
VALIDATION split, same convention as src/calibrate.py, and evaluated on
the TEST split. Conformal prediction runs on top of ISOTONIC-calibrated
probabilities by default, since Phase 3 found isotonic calibration gives
near-perfect ECE — well-behaved underlying probabilities tend to produce
tighter (more useful) conformal sets for the same coverage level, even
though conformal prediction's coverage guarantee holds regardless of the
input probabilities' calibration quality.

Usage:
    python -m src.quantify_uncertainty
    python -m src.quantify_uncertainty --alpha 0.05
"""

import argparse
import json
from pathlib import Path

from src.calibration.methods import fit_calibrator
from src.data.features import engineer_features
from src.data.loader import load_challenge_dataset
from src.data.preprocess import preprocess
from src.data.splits import PatientSplits, apply_split, assert_no_patient_overlap
from src.models.base import get_feature_columns
from src.models.lightgbm_model import LightGBMSepsisModel
from src.uncertainty.aps import build_prediction_sets_aps, calibrate_conformal_aps
from src.uncertainty.conformal import (
    average_set_size,
    build_prediction_sets,
    calibrate_conformal,
    empirical_coverage,
    set_size_distribution,
    singleton_accuracy,
)


def run_uncertainty_quantification(
    data_root: Path,
    splits_path: Path,
    model_path: Path,
    calibration_method: str,
    alphas: list,
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

    print(f"Applying '{calibration_method}' probability calibration "
          f"(fit on val, per Phase 3 convention) ...")
    calibrator = fit_calibrator(calibration_method, val_prob_raw, val_labels)
    val_prob_calibrated = calibrator.transform(val_prob_raw)
    test_prob_calibrated = calibrator.transform(test_prob_raw)

    results = {}
    for alpha in alphas:
        print(f"\n=== alpha = {alpha} (target coverage >= {1 - alpha:.2f}) ===")

        # --- LAC (primary method) ---
        qhat_lac = calibrate_conformal(val_prob_calibrated, val_labels, alpha=alpha)
        test_sets_lac = build_prediction_sets(test_prob_calibrated, qhat_lac)

        result_lac = {
            "method": "LAC",
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "qhat": qhat_lac,
            "empirical_coverage": empirical_coverage(test_sets_lac, test_labels),
            "average_set_size": average_set_size(test_sets_lac),
            "set_size_distribution": set_size_distribution(test_sets_lac),
            "singleton_accuracy": singleton_accuracy(test_sets_lac, test_labels),
        }
        print("--- LAC ---")
        print(json.dumps(result_lac, indent=2))

        # --- APS (comparison method — see project diary for motivation) ---
        qhat_aps = calibrate_conformal_aps(val_prob_calibrated, val_labels, alpha=alpha, seed=42)
        test_sets_aps = build_prediction_sets_aps(test_prob_calibrated, qhat_aps, seed=123)

        result_aps = {
            "method": "APS",
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "qhat": qhat_aps,
            "empirical_coverage": empirical_coverage(test_sets_aps, test_labels),
            "average_set_size": average_set_size(test_sets_aps),
            "set_size_distribution": set_size_distribution(test_sets_aps),
            "singleton_accuracy": singleton_accuracy(test_sets_aps, test_labels),
        }
        print("--- APS ---")
        print(json.dumps(result_aps, indent=2))

        results[alpha] = {"lac": result_lac, "aps": result_aps}

        for name, result in [("LAC", result_lac), ("APS", result_aps)]:
            coverage = result["empirical_coverage"]
            if coverage < (1 - alpha) - 0.02:
                print(
                    f"  NOTE ({name}): empirical coverage ({coverage:.3f}) fell meaningfully "
                    f"below the target ({1 - alpha:.2f}). On small calibration sets this can "
                    f"be ordinary finite-sample variance (more likely for APS, since it adds "
                    f"its own random draws); for LAC specifically, also check "
                    f"set_size_distribution[0] (empty-set rate) as a possible cause."
                )

    print(
        "\nComparison guide: LAC and APS should achieve SIMILAR empirical "
        "coverage (both honour the same nominal guarantee) but can differ "
        "substantially in HOW they express uncertainty — LAC tends toward "
        "empty sets under severe class imbalance (calibrated probabilities "
        "far from 0.5), while APS is more likely to produce genuine {0,1} "
        "double-inclusion ('truly ambiguous') sets, since it ranks classes "
        "rather than thresholding raw probability magnitude. Report "
        "set_size_distribution for both methods side by side in the Results "
        "chapter — this is the central comparison this phase was extended to make."
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Conformal prediction for the sepsis model.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/lightgbm_baseline.txt"))
    parser.add_argument("--calibration-method", choices=["platt", "temperature", "isotonic"], default="isotonic")
    parser.add_argument("--alpha", type=float, action="append", dest="alphas",
                         help="Target miscoverage rate. Can be passed multiple times. Default: 0.05, 0.1, 0.2")
    args = parser.parse_args()

    alphas = args.alphas if args.alphas else [0.05, 0.1, 0.2]

    run_uncertainty_quantification(
        args.data_root, args.splits, args.model, args.calibration_method, alphas
    )


if __name__ == "__main__":
    main()
