"""
Run full explainability analysis on the trained LightGBM model: SHAP
values on the test set, faithfulness (deletion/insertion AUC) on a
sample, and explanation stability across consecutive hours.

Usage:
    python -m src.explain
    python -m src.explain --faithfulness-samples 300
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.features import engineer_features
from src.data.loader import load_challenge_dataset
from src.data.preprocess import preprocess
from src.data.splits import PatientSplits, apply_split, assert_no_patient_overlap
from src.evaluation.faithfulness import stratified_faithfulness_summary
from src.evaluation.stability import consecutive_hour_stability
from src.explainability.shap_explainer import SHAPExplainer
from src.models.base import get_feature_columns
from src.models.lightgbm_model import LightGBMSepsisModel


def run_explainability(
    data_root: Path,
    splits_path: Path,
    model_path: Path,
    faithfulness_samples: int,
    stability_k: int,
):
    print(f"Loading raw data from {data_root} ...")
    df = load_challenge_dataset(data_root)

    splits = PatientSplits.load(splits_path)
    assert_no_patient_overlap(splits)

    train_df = apply_split(df, splits.train_ids)
    test_df = apply_split(df, splits.test_ids)

    print("Preprocessing (imputation stats fit on train only) ...")
    train_df, stats = preprocess(train_df, fit_stats=True)
    test_df, _ = preprocess(test_df, stats=stats)

    print("Engineering features ...")
    test_df = engineer_features(test_df)

    print(f"Loading trained model from {model_path} ...")
    model = LightGBMSepsisModel.load(model_path)
    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols]

    print("Computing SHAP values on the test set (TreeSHAP — exact, no sampling needed) ...")
    explainer = SHAPExplainer(model)
    shap_df = explainer.explain(X_test)

    print("\n--- Global feature importance (mean |SHAP value|) — top 15 ---")
    global_importance = shap_df.abs().mean().sort_values(ascending=False)
    print(global_importance.head(15).to_string())

    print(f"\nComputing faithfulness (deletion/insertion AUC, vs. random-ranking "
          f"baseline) separately for high-risk (top decile predicted "
          f"probability) and remaining test instances "
          f"({faithfulness_samples} samples per stratum) ...")
    y_prob_test = model.predict_proba(X_test)
    faith_df = stratified_faithfulness_summary(
        X_test, shap_df, model.predict_proba, y_prob_test,
        high_risk_quantile=0.9, n_samples_per_stratum=faithfulness_samples,
    )

    print("\n--- Faithfulness summary, by stratum (mean over sample) ---")
    summary_table = faith_df.groupby("stratum")[
        ["deletion_auc", "deletion_auc_random", "insertion_auc", "insertion_auc_random"]
    ].mean()
    print(summary_table.to_string())

    print(
        "\nInterpretation: within each stratum, compare 'deletion_auc' to "
        "'deletion_auc_random' (real should be LOWER — deleting genuinely "
        "important features drops the prediction faster than deleting them "
        "in an arbitrary order) and 'insertion_auc' to 'insertion_auc_random' "
        "(real should be HIGHER). The high_risk stratum is the clinically "
        "relevant one — it isolates instances where a clinician would "
        "actually see an alert and rely on the explanation, rather than "
        "being dominated by the many confidently-negative hours where "
        "sepsis probability is near zero regardless of explanation quality."
    )

    for stratum in summary_table.index:
        row = summary_table.loc[stratum]
        deletion_wins = row["deletion_auc"] < row["deletion_auc_random"]
        insertion_wins = row["insertion_auc"] > row["insertion_auc_random"]
        verdict = "beats" if (deletion_wins and insertion_wins) else "does NOT clearly beat"
        print(f"  [{stratum}] SHAP {verdict} the random-ranking baseline on both metrics.")

    print(f"\nComputing explanation stability across consecutive hours "
          f"(top-{stability_k} overlap, cosine similarity) ...")
    stability_df = consecutive_hour_stability(
        shap_df, test_df["patient_id"], test_df["ICULOS"], k=stability_k
    )
    print("\n--- Stability summary (mean over all consecutive-hour pairs) ---")
    print(json.dumps({
        "mean_cosine_similarity": float(stability_df["cosine_similarity"].mean()),
        "mean_top_k_overlap": float(stability_df["top_k_overlap"].mean()),
        "n_pairs": len(stability_df),
    }, indent=2))

    return {
        "global_importance": global_importance,
        "faithfulness": faith_df,
        "stability": stability_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Explainability analysis for the sepsis model.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/lightgbm_baseline.txt"))
    parser.add_argument("--faithfulness-samples", type=int, default=200,
                         help="Number of test instances to sample for the expensive "
                              "deletion/insertion AUC computation.")
    parser.add_argument("--stability-k", type=int, default=5,
                         help="k for top-k feature overlap in the stability metric.")
    args = parser.parse_args()

    run_explainability(
        args.data_root, args.splits, args.model,
        args.faithfulness_samples, args.stability_k,
    )


if __name__ == "__main__":
    main()
