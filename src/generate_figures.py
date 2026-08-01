"""
Generate the core figures for the dissertation Results chapter:
  1. Reliability diagram (before / after calibration)
  2. Conformal prediction coverage-efficiency trade-off (LAC vs. APS)
  3. Decision curve (net benefit vs. threshold)
  4. SHAP summary plot (global feature importance, on a sample for readability)

All figures are saved as 300 DPI PNGs to outputs/figures/ — suitable for
direct inclusion in a Word document or LaTeX build without re-exporting.

Every number behind these plots reuses the SAME functions already used
(and tested) in earlier phases (src/evaluation/, src/calibration/,
src/uncertainty/, src/explainability/) — this script only adds
visualisation on top, no new metric logic.

Usage:
    python -m src.generate_figures
    python -m src.generate_figures --shap-sample-size 1000
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe to run without a display
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.calibration.methods import fit_calibrator
from src.data.features import engineer_features
from src.data.loader import load_challenge_dataset
from src.data.preprocess import preprocess
from src.data.splits import PatientSplits, apply_split, assert_no_patient_overlap
from src.evaluation.calibration_metrics import reliability_diagram_data
from src.evaluation.decision_curve import decision_curve_data
from src.explainability.shap_explainer import SHAPExplainer
from src.models.base import get_feature_columns
from src.models.lightgbm_model import LightGBMSepsisModel
from src.uncertainty.aps import build_prediction_sets_aps, calibrate_conformal_aps
from src.uncertainty.conformal import (
    average_set_size,
    build_prediction_sets,
    calibrate_conformal,
    empirical_coverage,
)

FIGURE_DPI = 300


def plot_reliability_diagram(y_true, y_prob_before, y_prob_after, out_path, n_bins=10):
    bins_before = reliability_diagram_data(y_true, y_prob_before, n_bins=n_bins)
    bins_after = reliability_diagram_data(y_true, y_prob_after, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, bins, title in [
        (axes[0], bins_before, "Before calibration"),
        (axes[1], bins_after, "After calibration"),
    ]:
        x = [b.mean_predicted for b in bins if b.count > 0]
        y = [b.mean_observed for b in bins if b.count > 0]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.plot(x, y, marker="o", color="tab:blue", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Observed sepsis rate")
    fig.suptitle("Reliability Diagram: Predicted Probability vs. Observed Sepsis Rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_conformal_tradeoff(
    val_prob_cal, val_labels, test_prob_cal, test_labels, out_path, alphas=None
):
    alphas = alphas if alphas is not None else np.linspace(0.02, 0.4, 15)

    lac_coverage, lac_size = [], []
    aps_coverage, aps_size = [], []

    for alpha in alphas:
        qhat_lac = calibrate_conformal(val_prob_cal, val_labels, alpha=alpha)
        sets_lac = build_prediction_sets(test_prob_cal, qhat_lac)
        lac_coverage.append(empirical_coverage(sets_lac, test_labels))
        lac_size.append(average_set_size(sets_lac))

        qhat_aps = calibrate_conformal_aps(val_prob_cal, val_labels, alpha=alpha, seed=42)
        sets_aps = build_prediction_sets_aps(test_prob_cal, qhat_aps, seed=123)
        aps_coverage.append(empirical_coverage(sets_aps, test_labels))
        aps_size.append(average_set_size(sets_aps))

    target_coverage = 1 - alphas

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(target_coverage, lac_coverage, marker="o", label="LAC")
    axes[0].plot(target_coverage, aps_coverage, marker="s", label="APS")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Target = achieved")
    axes[0].set_xlabel("Target coverage (1 \u2212 \u03b1)")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Coverage: target vs. achieved")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(target_coverage, lac_size, marker="o", label="LAC")
    axes[1].plot(target_coverage, aps_size, marker="s", label="APS")
    axes[1].set_xlabel("Target coverage (1 \u2212 \u03b1)")
    axes[1].set_ylabel("Average prediction set size")
    axes[1].set_title("Efficiency: average set size vs. coverage")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Conformal Prediction: Coverage\u2013Efficiency Trade-off (LAC vs. APS)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_decision_curve(y_true, y_prob, out_path):
    curve = decision_curve_data(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(curve["threshold"], curve["net_benefit_model"], label="Model", color="tab:blue", linewidth=2)
    ax.plot(curve["threshold"], curve["net_benefit_treat_all"], label="Treat all", color="tab:red", linestyle="--")
    ax.plot(curve["threshold"], curve["net_benefit_treat_none"], label="Treat none", color="gray", linestyle=":")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.grid(True, alpha=0.3)
    ax.legend()
    lower_bound = min(curve["net_benefit_treat_all"].min(), curve["net_benefit_model"].min(), 0.0)
    ax.set_ylim(bottom=lower_bound - 0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_shap_summary(shap_values_df, X_sample, out_path, max_display=15):
    """SHAP's own summary_plot handles the "beeswarm" rendering — this
    wrapper just points it at this project's DataFrame outputs and saves
    the result at dissertation-ready resolution.
    """
    plt.figure(figsize=(9, 7))
    shap.summary_plot(
        shap_values_df.to_numpy(),
        X_sample[shap_values_df.columns],
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    data_root: Path,
    splits_path: Path,
    model_path: Path,
    calibration_method: str,
    out_dir: Path,
    shap_sample_size: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

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
    val_prob_cal = calibrator.transform(val_prob_raw)
    test_prob_cal = calibrator.transform(test_prob_raw)

    print("Plot 1/4: reliability diagram ...")
    plot_reliability_diagram(
        test_labels, test_prob_raw, test_prob_cal, out_dir / "reliability_diagram.png"
    )

    print("Plot 2/4: conformal coverage-efficiency trade-off (LAC vs APS) ...")
    plot_conformal_tradeoff(
        val_prob_cal, val_labels, test_prob_cal, test_labels,
        out_dir / "conformal_tradeoff.png",
    )

    print("Plot 3/4: decision curve ...")
    plot_decision_curve(test_labels, test_prob_cal, out_dir / "decision_curve.png")

    print(f"Plot 4/4: SHAP summary plot (sampling {shap_sample_size} test "
          f"instances for a readable beeswarm plot) ...")
    rng = np.random.default_rng(42)
    sample_size = min(shap_sample_size, len(test_df))
    sample_idx = rng.choice(test_df.index, size=sample_size, replace=False)
    X_sample = test_df.loc[sample_idx, feature_cols]

    explainer = SHAPExplainer(model)
    shap_sample = explainer.explain(X_sample)
    plot_shap_summary(shap_sample, X_sample, out_dir / "shap_summary.png")

    print(f"\nAll 4 figures saved to {out_dir}/")
    print("  - reliability_diagram.png")
    print("  - conformal_tradeoff.png")
    print("  - decision_curve.png")
    print("  - shap_summary.png")


def main():
    parser = argparse.ArgumentParser(description="Generate dissertation Results-chapter figures.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    parser.add_argument("--model", type=Path, default=Path("outputs/lightgbm_baseline.txt"))
    parser.add_argument("--calibration-method", choices=["platt", "temperature", "isotonic"], default="isotonic")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--shap-sample-size", type=int, default=2000,
                         help="Number of test instances sampled for the SHAP summary plot "
                              "(full TreeSHAP is exact and cheap, but a beeswarm plot with "
                              "hundreds of thousands of points is unreadable and slow to render).")
    args = parser.parse_args()

    generate_all_figures(
        args.data_root, args.splits, args.model, args.calibration_method,
        args.out_dir, args.shap_sample_size,
    )


if __name__ == "__main__":
    main()
