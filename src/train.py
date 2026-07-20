"""
End-to-end training script: raw data -> splits -> preprocessing ->
feature engineering -> LightGBM training -> evaluation.

This is the first place in the project where every earlier module is used
together, so it doubles as an integration check on the whole pipeline, not
just a training script.

Usage:
    python -m src.train
    python -m src.train --data-root data/raw --config configs/lightgbm.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from src.data.features import engineer_features
from src.data.loader import load_challenge_dataset
from src.data.preprocess import preprocess
from src.data.splits import (
    PatientSplits,
    apply_split,
    assert_no_patient_overlap,
    make_patient_splits,
)
from src.evaluation.metrics import evaluate_predictions
from src.models.base import get_feature_columns
from src.models.lightgbm_model import LightGBMSepsisModel


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_or_create_splits(df, splits_path: Path) -> PatientSplits:
    """Reuse existing splits if they're already saved, rather than
    re-randomising every run — this is what makes every subsequent
    experiment comparable on exactly the same train/val/test patients.
    """
    if splits_path.exists():
        print(f"Loading existing splits from {splits_path}")
        return PatientSplits.load(splits_path)

    print("No existing splits found — creating new patient-level splits.")
    splits = make_patient_splits(df)
    splits.save(splits_path)
    print(f"Saved new splits to {splits_path}")
    return splits


def run_pipeline(data_root: Path, config_path: Path, splits_path: Path, model_out_path: Path):
    print(f"Loading raw data from {data_root} ...")
    df = load_challenge_dataset(data_root)
    print(f"Loaded {df['patient_id'].nunique()} patients, {len(df)} patient-hours.")

    splits = get_or_create_splits(df, splits_path)
    assert_no_patient_overlap(splits)  # QR1 — never skip this check

    train_df = apply_split(df, splits.train_ids)
    val_df = apply_split(df, splits.val_ids)
    test_df = apply_split(df, splits.test_ids)
    print(f"Train: {train_df['patient_id'].nunique()} patients, "
          f"Val: {val_df['patient_id'].nunique()} patients, "
          f"Test: {test_df['patient_id'].nunique()} patients.")

    print("Preprocessing (imputation stats fit on train only) ...")
    train_df, stats = preprocess(train_df, fit_stats=True)
    val_df, _ = preprocess(val_df, stats=stats)
    test_df, _ = preprocess(test_df, stats=stats)

    print("Engineering features ...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    feature_cols = get_feature_columns(train_df)
    print(f"Using {len(feature_cols)} feature columns.")

    config = load_config(config_path)
    model = LightGBMSepsisModel(
        params=config.get("params", {}),
        num_boost_round=config.get("num_boost_round", 500),
        early_stopping_rounds=config.get("early_stopping_rounds", 30),
    )

    print("Training LightGBM ...")
    model.fit(
        train_df[feature_cols],
        train_df["SepsisLabel"].to_numpy(),
        val_df[feature_cols],
        val_df["SepsisLabel"].to_numpy(),
    )

    print("\n--- Validation metrics ---")
    val_prob = model.predict_proba(val_df[feature_cols])
    val_metrics = evaluate_predictions(val_df, val_prob)
    print(json.dumps(val_metrics, indent=2))

    print("\n--- Test metrics ---")
    test_prob = model.predict_proba(test_df[feature_cols])
    test_metrics = evaluate_predictions(test_df, test_prob)
    print(json.dumps(test_metrics, indent=2))

    model.save(model_out_path)
    print(f"\nModel saved to {model_out_path}")

    top_features = model.feature_importance().head(10)
    print("\nTop 10 features by gain (quick sanity check, not a substitute for SHAP):")
    print(top_features.to_string())

    return {"val": val_metrics, "test": test_metrics}


def main():
    parser = argparse.ArgumentParser(description="Train the LightGBM sepsis baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/lightgbm.yaml"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--model-out", type=Path, default=Path("outputs/lightgbm_baseline.txt"))
    args = parser.parse_args()

    run_pipeline(args.data_root, args.config, args.splits, args.model_out)


if __name__ == "__main__":
    main()
