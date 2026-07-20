"""
Phase 6 — cross-site (hospital) stress test.

DESIGN NOTE — improvement over the original project specification: the
specification anticipated using a held-out partition of pooled hospital
A+B data as a PROXY for site shift, since the Challenge's true hidden
test hospital (system C) is not publicly released. However, the dataset
does contain two real, distinct hospital systems (A and B, tagged via the
`hospital` column, deliberately excluded from the feature set in
src/models/base.py). This script uses that real structure directly：
train / validate / in-distribution-test entirely on hospital A, then
evaluate out-of-distribution on ALL of hospital B. This is a strictly
stronger generalisation test than a random re-split of pooled data, and
is used here instead of the originally-planned proxy.

Everything hospital-B-related is touched ONLY at evaluation time:
imputation statistics, the trained model, the probability calibrator, and
the conformal qhat are all fit exclusively on hospital A data. Applying
them unchanged to hospital B is what makes this a genuine shift test —
refitting anything on hospital B would leak site-specific information
back in and defeat the purpose.

Usage:
    python -m src.cross_site
    python -m src.cross_site --alpha 0.1 --calibration-method isotonic
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.calibration.methods import fit_calibrator
from src.data.features import engineer_features
from src.data.loader import load_challenge_dataset
from src.data.preprocess import preprocess
from src.data.splits import (
    PatientSplits,
    apply_split,
    assert_no_patient_overlap,
    make_patient_splits,
)
from src.evaluation.calibration_metrics import brier_score, expected_calibration_error
from src.evaluation.metrics import evaluate_predictions
from src.explainability.shap_explainer import SHAPExplainer
from src.models.base import get_feature_columns
from src.models.lightgbm_model import LightGBMSepsisModel
from src.uncertainty.conformal import (
    build_prediction_sets,
    calibrate_conformal,
    empirical_coverage,
    set_size_distribution,
)


def _load_or_create_hospital_a_splits(hospital_a: pd.DataFrame, splits_path: Path) -> PatientSplits:
    """Same convention as get_or_create_splits in src/train.py, but with an
    explicit safety check: this MUST be a hospital-A-only splits file, not
    the pooled splits.json from earlier phases. Loading the wrong file
    would silently drop the missing patient IDs in apply_split() rather
    than raising an obvious error, which is exactly the kind of bug that
    is easy to miss and would quietly invalidate the whole experiment.
    """
    if splits_path.exists():
        print(f"Loading existing hospital-A splits from {splits_path}")
        splits = PatientSplits.load(splits_path)
    else:
        print("No existing hospital-A splits found — creating new ones.")
        splits = make_patient_splits(hospital_a)
        splits.save(splits_path)
        print(f"Saved new hospital-A splits to {splits_path}")

    hospital_a_ids = set(hospital_a["patient_id"].unique())
    all_split_ids = set(splits.train_ids) | set(splits.val_ids) | set(splits.test_ids)
    if not all_split_ids.issubset(hospital_a_ids):
        raise ValueError(
            f"Splits loaded from {splits_path} contain patient IDs not present "
            "in hospital A. This usually means the pooled (A+B) splits.json was "
            "loaded by mistake instead of a hospital-A-only splits file. Use a "
            "distinct --splits path for this script (default: "
            "configs/splits_hospitalA.json)."
        )
    return splits


def run_cross_site_stress_test(
    data_root: Path,
    splits_path: Path,
    config_path: Path,
    alpha: float,
    calibration_method: str,
):
    print(f"Loading raw data from {data_root} ...")
    df = load_challenge_dataset(data_root)

    hospital_a = df[df["hospital"] == "A"].copy()
    hospital_b = df[df["hospital"] == "B"].copy()
    print(f"Hospital A: {hospital_a['patient_id'].nunique()} patients "
          f"({len(hospital_a)} patient-hours). "
          f"Hospital B: {hospital_b['patient_id'].nunique()} patients "
          f"({len(hospital_b)} patient-hours).")

    splits = _load_or_create_hospital_a_splits(hospital_a, splits_path)
    assert_no_patient_overlap(splits)

    train_df = apply_split(hospital_a, splits.train_ids)
    val_df = apply_split(hospital_a, splits.val_ids)
    id_test_df = apply_split(hospital_a, splits.test_ids)  # in-distribution test
    ood_test_df = hospital_b  # out-of-distribution test: ALL of hospital B, untouched until now

    print("\nPreprocessing (imputation stats fit on HOSPITAL A TRAIN ONLY — "
          "applied unchanged to hospital B, deliberately never re-fit on it) ...")
    train_df, stats = preprocess(train_df, fit_stats=True)
    val_df, _ = preprocess(val_df, stats=stats)
    id_test_df, _ = preprocess(id_test_df, stats=stats)
    ood_test_df, _ = preprocess(ood_test_df, stats=stats)

    print("Engineering features ...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    id_test_df = engineer_features(id_test_df)
    ood_test_df = engineer_features(ood_test_df)

    feature_cols = get_feature_columns(train_df)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = LightGBMSepsisModel(
        params=config.get("params", {}),
        num_boost_round=config.get("num_boost_round", 500),
        early_stopping_rounds=config.get("early_stopping_rounds", 30),
    )

    print("\nTraining LightGBM on HOSPITAL A ONLY ...")
    model.fit(
        train_df[feature_cols], train_df["SepsisLabel"].to_numpy(),
        val_df[feature_cols], val_df["SepsisLabel"].to_numpy(),
    )

    val_labels = val_df["SepsisLabel"].to_numpy()
    val_prob = model.predict_proba(val_df[feature_cols])
    id_test_prob = model.predict_proba(id_test_df[feature_cols])
    ood_test_prob = model.predict_proba(ood_test_df[feature_cols])

    # === 1. Discrimination under shift ===
    print("\n=== 1. DISCRIMINATION under shift ===")
    id_metrics = evaluate_predictions(id_test_df, id_test_prob)
    ood_metrics = evaluate_predictions(ood_test_df, ood_test_prob)
    print("In-distribution (held-out hospital A):")
    print(json.dumps(id_metrics, indent=2))
    print("Out-of-distribution (all of hospital B):")
    print(json.dumps(ood_metrics, indent=2))
    print(f"AUROC gap (in-distribution minus out-of-distribution): "
          f"{id_metrics['auroc'] - ood_metrics['auroc']:.4f}")

    # === 2. Calibration under shift ===
    print("\n=== 2. CALIBRATION under shift ===")
    id_labels = id_test_df["SepsisLabel"].to_numpy()
    ood_labels = ood_test_df["SepsisLabel"].to_numpy()

    ece_id_before = expected_calibration_error(id_labels, id_test_prob)
    ece_ood_before = expected_calibration_error(ood_labels, ood_test_prob)
    brier_id_before = brier_score(id_labels, id_test_prob)
    brier_ood_before = brier_score(ood_labels, ood_test_prob)

    print(f"Fitting '{calibration_method}' calibrator on hospital-A validation set ...")
    calibrator = fit_calibrator(calibration_method, val_prob, val_labels)
    id_test_prob_cal = calibrator.transform(id_test_prob)
    ood_test_prob_cal = calibrator.transform(ood_test_prob)

    ece_id_after = expected_calibration_error(id_labels, id_test_prob_cal)
    ece_ood_after = expected_calibration_error(ood_labels, ood_test_prob_cal)
    brier_id_after = brier_score(id_labels, id_test_prob_cal)
    brier_ood_after = brier_score(ood_labels, ood_test_prob_cal)

    print(json.dumps({
        "ece_in_distribution": {"before": ece_id_before, "after": ece_id_after},
        "ece_out_of_distribution": {"before": ece_ood_before, "after": ece_ood_after},
        "brier_in_distribution": {"before": brier_id_before, "after": brier_id_after},
        "brier_out_of_distribution": {"before": brier_ood_before, "after": brier_ood_after},
    }, indent=2))
    print(
        "Interpretation: the calibrator is fit ONLY on hospital-A validation "
        "data. If ece_out_of_distribution stays close to ece_in_distribution, "
        "calibration transfers across sites. A sharp increase demonstrates "
        "that calibration itself is site-specific — a realistic and "
        "important limitation for real-world deployment across hospitals."
    )

    # === 3. Conformal coverage under shift ===
    print(f"\n=== 3. CONFORMAL COVERAGE under shift (alpha={alpha}) ===")
    val_prob_cal = calibrator.transform(val_prob)
    qhat = calibrate_conformal(val_prob_cal, val_labels, alpha=alpha)

    id_sets = build_prediction_sets(id_test_prob_cal, qhat)
    ood_sets = build_prediction_sets(ood_test_prob_cal, qhat)

    id_coverage = empirical_coverage(id_sets, id_labels)
    ood_coverage = empirical_coverage(ood_sets, ood_labels)

    print(json.dumps({
        "target_coverage": 1 - alpha,
        "in_distribution_coverage": id_coverage,
        "out_of_distribution_coverage": ood_coverage,
        "in_distribution_set_size_distribution": set_size_distribution(id_sets),
        "out_of_distribution_set_size_distribution": set_size_distribution(ood_sets),
    }, indent=2))
    print(
        "Interpretation: the conformal coverage GUARANTEE relies on "
        "exchangeability between the calibration set (hospital A val) and "
        "the set it's applied to. It is NOT expected to hold under genuine "
        "distribution shift — if out_of_distribution_coverage falls "
        "meaningfully below target, that is the documented breakdown this "
        "experiment exists to measure, not a bug."
    )

    # === 4. Feature importance under shift (tests the Phase 5 HospAdmTime/Unit1 hypothesis) ===
    print("\n=== 4. FEATURE IMPORTANCE under shift ===")
    explainer = SHAPExplainer(model)
    id_shap = explainer.explain(id_test_df[feature_cols])
    ood_shap = explainer.explain(ood_test_df[feature_cols])

    id_importance = id_shap.abs().mean().rename("in_distribution")
    ood_importance = ood_shap.abs().mean().rename("out_of_distribution")
    comparison = pd.concat([id_importance, ood_importance], axis=1)
    comparison["ratio_ood_over_id"] = comparison["out_of_distribution"] / comparison["in_distribution"]
    comparison = comparison.sort_values("in_distribution", ascending=False)

    print("\nTop 10 features by in-distribution importance, compared to "
          "out-of-distribution importance:")
    print(comparison.head(10).to_string())
    print(
        "\nInterpretation: a ratio near 1.0 means a feature's importance held "
        "up under shift; a ratio near 0 means it collapsed. This directly "
        "tests the Phase 5 hypothesis flagged for this experiment: were "
        "HospAdmTime and Unit1 (hospital-administrative features that ranked "
        "above Lactate in Phase 5) relying on hospital-A-specific patterns "
        "that don't transfer to hospital B?"
    )
    for feature in ["HospAdmTime", "Unit1", "Lactate"]:
        if feature in comparison.index:
            row = comparison.loc[feature]
            print(f"  {feature}: in-distribution={row['in_distribution']:.4f}, "
                  f"out-of-distribution={row['out_of_distribution']:.4f}, "
                  f"ratio={row['ratio_ood_over_id']:.2f}")

    return {
        "id_metrics": id_metrics, "ood_metrics": ood_metrics,
        "calibration": {
            "ece_id": (ece_id_before, ece_id_after), "ece_ood": (ece_ood_before, ece_ood_after),
            "brier_id": (brier_id_before, brier_id_after), "brier_ood": (brier_ood_before, brier_ood_after),
        },
        "coverage": {"id": id_coverage, "ood": ood_coverage},
        "feature_importance_comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-site (hospital) stress test.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_hospitalA.json"),
                         help="MUST be distinct from the pooled configs/splits.json used in earlier phases.")
    parser.add_argument("--config", type=Path, default=Path("configs/lightgbm.yaml"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--calibration-method", choices=["platt", "temperature", "isotonic"], default="isotonic")
    args = parser.parse_args()

    run_cross_site_stress_test(
        args.data_root, args.splits, args.config, args.alpha, args.calibration_method
    )


if __name__ == "__main__":
    main()
