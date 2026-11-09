import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

DPI = 300

FAITHFULNESS = {
    "high_risk": {"deletion_shap": 0.3735, "deletion_random": 0.5665,
                   "insertion_shap": 0.7427, "insertion_random": 0.5741},
    "rest": {"deletion_shap": 0.3640, "deletion_random": 0.3647,
              "insertion_shap": 0.3472, "insertion_random": 0.4023},
}

# Table 4.6
CROSS_SITE = {
    "AUROC": {"in": 0.8260, "out": 0.7470},
    "AUPRC": {"in": 0.1213, "out": 0.0626},
    "ECE after calibration": {"in": 0.00048, "out": 0.00345},
    "Conformal coverage (α=0.10)": {"in": 0.8984, "out": 0.9349},
}

# Table 4.7 
IMPORTANCE_SHIFT = [
    ("hours_since_admission", 0.2029, 0.2244),
    ("ICULOS",                0.1998, 0.2173),
    ("HospAdmTime",           0.1642, 0.2433),
    ("Temp_roll6h_max",       0.1568, 0.1333),
    ("Creatinine",            0.1281, 0.1101),
    ("Resp_roll24h_min",      0.1258, 0.1226),
    ("Unit1",                 0.1199, 0.1325),
    ("Unit2",                 0.1194, 0.1529),
    ("WBC",                   0.1165, 0.1220),
    ("FiO2",                  0.1149, 0.1065),
    ("Lactate",               0.1065, 0.0918),
]
HIGHLIGHT = {"HospAdmTime": "#b85450", "Unit1": "#b85450", "Lactate": "#4c78a8"}

ALPHAS = [0.05, 0.10, 0.20]



def prepare(data_root: Path, splits_path: Path, model_path: Path):
    """Load data, rebuild the splits, and return everything the figures need."""
    from src.calibration.methods import fit_calibrator
    from src.data.features import engineer_features
    from src.data.loader import load_challenge_dataset
    from src.data.preprocess import preprocess
    from src.data.splits import PatientSplits, apply_split
    from src.models.base import get_feature_columns
    from src.models.lightgbm_model import LightGBMSepsisModel

    print(f"Loading data from {data_root} ...")
    df = load_challenge_dataset(data_root)
    splits = PatientSplits.load(splits_path)

    train_df = apply_split(df, splits.train_ids)
    val_df = apply_split(df, splits.val_ids)
    test_df = apply_split(df, splits.test_ids)

    print("Preprocessing and engineering features ...")
    train_df, stats = preprocess(train_df, fit_stats=True)
    val_df, _ = preprocess(val_df, stats=stats)
    test_df, _ = preprocess(test_df, stats=stats)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    model = LightGBMSepsisModel.load(model_path)
    cols = get_feature_columns(test_df)

    val_raw = model.predict_proba(val_df[cols])
    test_raw = model.predict_proba(test_df[cols])
    val_y = val_df["SepsisLabel"].to_numpy()
    test_y = test_df["SepsisLabel"].to_numpy()

    print("Fitting calibrators on the validation split ...")
    cal = {name: fit_calibrator(name, val_raw, val_y)
           for name in ("platt", "temperature", "isotonic")}

    return {
        "model": model, "cols": cols,
        "val_df": val_df, "test_df": test_df,
        "val_raw": val_raw, "test_raw": test_raw,
        "val_y": val_y, "test_y": test_y,
        "cal": cal,
        "val_iso": cal["isotonic"].transform(val_raw),
        "test_iso": cal["isotonic"].transform(test_raw),
    }


def fig_4_1(ctx, out: Path):
    from src.evaluation.calibration_metrics import expected_calibration_error, reliability_diagram_data

    panels = [
        ("Uncalibrated", ctx["test_raw"]),
        ("Platt scaling", ctx["cal"]["platt"].transform(ctx["test_raw"])),
        ("Temperature scaling", ctx["cal"]["temperature"].transform(ctx["test_raw"])),
        ("Isotonic regression", ctx["test_iso"]),
    ]
    y = ctx["test_y"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, (title, p) in zip(axes, panels):
        bins = reliability_diagram_data(y, p, n_bins=10)
        xs = [b.mean_predicted for b in bins if b.count > 0]
        ys = [b.mean_observed for b in bins if b.count > 0]
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
        ax.plot(xs, ys, "o-", color="#4c78a8", lw=2, ms=5, label="Model")
        ece = expected_calibration_error(y, p)
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=11)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.set_axisbelow(True)
    axes[0].set_ylabel("Observed sepsis rate")
    axes[0].legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_4_1_reliability.png", dpi=DPI)
    plt.close(fig)
    print("  Figure 4.1 written")


def fig_4_5(ctx, out: Path, n_thresholds: int = 60):
    from src.evaluation.metrics import evaluate_predictions

    thresholds = np.linspace(0.005, 0.60, n_thresholds)
    print(f"  sweeping {n_thresholds} thresholds (this is the slow figure) ...")

    series = {}
    for label, probs in (("Raw probabilities", ctx["test_raw"]),
                          ("Calibrated (isotonic)", ctx["test_iso"])):
        vals = []
        for i, t in enumerate(thresholds):
            vals.append(evaluate_predictions(ctx["test_df"], probs, threshold=float(t))["utility_normalized"])
            if (i + 1) % 15 == 0:
                print(f"    {label}: {i + 1}/{n_thresholds}")
        series[label] = vals

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresholds, series["Raw probabilities"], lw=2, color="#9ecae1", label="Raw probabilities")
    ax.plot(thresholds, series["Calibrated (isotonic)"], lw=2.5, color="#08519c", label="Calibrated (isotonic)")

    ax.axvline(0.5, color="#b85450", ls="--", lw=1.5)
    ax.annotate("conventional default (0.5)\nno alerts fire after calibration",
                xy=(0.5, 0.05), xytext=(0.36, 0.30), fontsize=9, color="#b85450",
                arrowprops=dict(arrowstyle="->", color="#b85450", lw=1.2))
    ax.axvline(0.03, color="#2f7d54", ls="--", lw=1.5)
    ax.annotate("selected on validation (0.030)",
                xy=(0.03, 0.50), xytext=(0.10, 0.58), fontsize=9, color="#2f7d54",
                arrowprops=dict(arrowstyle="->", color="#2f7d54", lw=1.2))

    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Normalised utility score")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_4_5_utility_threshold.png", dpi=DPI)
    plt.close(fig)
    print("  Figure 4.5 written")

def fig_4_6(out: Path):
    strata = ["High risk\n(top decile)", "Remaining hours"]
    keys = ["high_risk", "rest"]
    width = 0.2
    x = np.arange(len(strata))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = [
        ("Deletion — SHAP", "deletion_shap", -1.5, "#08519c"),
        ("Deletion — random", "deletion_random", -0.5, "#9ecae1"),
        ("Insertion — SHAP", "insertion_shap", 0.5, "#2f7d54"),
        ("Insertion — random", "insertion_random", 1.5, "#a1d99b"),
    ]
    for label, key, offset, colour in bars:
        vals = [FAITHFULNESS[k][key] for k in keys]
        ax.bar(x + offset * width, vals, width, label=label, color=colour)

    ax.set_xticks(x); ax.set_xticklabels(strata, fontsize=11)
    ax.set_ylabel("Area under curve")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.text(0.02, 0.96, "Deletion: lower is more faithful   ·   Insertion: higher is more faithful",
            transform=ax.transAxes, fontsize=9, style="italic", va="top", color="#444444")
    fig.tight_layout()
    fig.savefig(out / "fig_4_6_faithfulness.png", dpi=DPI)
    plt.close(fig)
    print("  Figure 4.6 written")

def fig_4_7(out: Path):
    metrics = list(CROSS_SITE)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, name in zip(axes, metrics):
        vals = [CROSS_SITE[name]["in"], CROSS_SITE[name]["out"]]
        ax.bar(["Hospital A\n(in-dist.)", "Hospital B\n(out-of-dist.)"], vals,
               color=["#4c78a8", "#b85450"], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}".rstrip("0").rstrip("."), ha="center", va="bottom", fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.28)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_4_7_cross_site.png", dpi=DPI)
    plt.close(fig)
    print("  Figure 4.7 written")

def fig_4_8(ctx, out: Path):
    y, p = ctx["test_y"], ctx["test_raw"]
    prevalence = float(y.mean())

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(fpr, tpr, lw=2, color="#08519c")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve"); axes[0].grid(alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].plot(rec, prec, lw=2, color="#2f7d54")
    axes[1].axhline(prevalence, ls="--", color="grey", lw=1.2)
    axes[1].annotate(f"prevalence = {prevalence*100:.2f}%\n(a random ranker)",
                     xy=(0.55, prevalence), xytext=(0.45, prevalence + 0.12), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="grey", lw=1))
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall curve")
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig_4_8_roc_pr.png", dpi=DPI)
    plt.close(fig)
    print("  Figure 4.8 written")

def fig_4_9(ctx, out: Path):
    """Two panels: overall composition, and the ambiguous share on its own
    scale. The second panel exists because the ambiguous share is only a few
    percent — in a single stacked bar it would be an invisible sliver, and
    that share is the finding."""
    from src.uncertainty.aps import build_prediction_sets_aps, calibrate_conformal_aps
    from src.uncertainty.conformal import build_prediction_sets, calibrate_conformal, set_size_distribution

    rows = []
    for a in ALPHAS:
        q = calibrate_conformal(ctx["val_iso"], ctx["val_y"], alpha=a)
        rows.append(("LAC", a, set_size_distribution(build_prediction_sets(ctx["test_iso"], q))))
        q2 = calibrate_conformal_aps(ctx["val_iso"], ctx["val_y"], alpha=a, seed=42)
        rows.append(("APS", a, set_size_distribution(build_prediction_sets_aps(ctx["test_iso"], q2, seed=123))))

    labels = [f"{m}\nα={a}" for m, a, _ in rows]
    empty = np.array([d[0] * 100 for _, _, d in rows])
    single = np.array([d[1] * 100 for _, _, d in rows])
    double = np.array([d[2] * 100 for _, _, d in rows])
    colours = ["#4c78a8" if m == "LAC" else "#b85450" for m, _, _ in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(rows))

    axes[0].bar(x, single, color="#4c78a8", label="One class (committed)")
    axes[0].bar(x, empty, bottom=single, color="#c7c7c7", label="Empty (no class confident)")
    axes[0].bar(x, double, bottom=single + empty, color="#b85450", label="Two classes (ambiguous)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylabel("Proportion of test patient-hours (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)
    axes[0].grid(axis="y", alpha=0.3); axes[0].set_axisbelow(True)
    axes[0].set_title("Overall composition of prediction sets", fontsize=11)

    axes[1].bar(x, double, color=colours)
    for i, v in enumerate(double):
        axes[1].text(i, v + max(double.max(), 0.5) * 0.03,
                     "0.00%" if v < 0.005 else f"{v:.2f}%",
                     ha="center", fontsize=10, fontweight="bold",
                     color="#4c78a8" if colours[i] == "#4c78a8" else "#b85450")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel("Ambiguous sets (%)")
    axes[1].set_ylim(0, max(double.max() * 1.35, 0.6))
    axes[1].grid(axis="y", alpha=0.3); axes[1].set_axisbelow(True)
    axes[1].set_title("Ambiguous sets only, on their own scale", fontsize=11)

    fig.tight_layout()
    fig.savefig(out / "fig_4_9_set_composition.png", dpi=DPI)
    plt.close(fig)
    print("  Figure 4.9 written")


def fig_4_10(ctx, out: Path, sample_patients: int = 800):
    from src.evaluation.stability import consecutive_hour_stability
    from src.explainability.shap_explainer import SHAPExplainer

    test_df = ctx["test_df"]
    ids = test_df["patient_id"].drop_duplicates()
    if sample_patients and len(ids) > sample_patients:
        ids = ids.sample(sample_patients, random_state=42)
        print(f"  sampling {sample_patients} patients for the stability distribution")
    subset = test_df[test_df["patient_id"].isin(ids)]

    print("  computing SHAP values ...")
    shap_df = SHAPExplainer(ctx["model"]).explain(subset[ctx["cols"]])
    stab = consecutive_hour_stability(shap_df, subset["patient_id"], subset["ICULOS"], k=5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(stab["top_k_overlap"], bins=np.arange(-0.1, 1.15, 0.1), color="#4c78a8",
                 edgecolor="white")
    m = stab["top_k_overlap"].mean()
    axes[0].axvline(m, color="#b85450", ls="--", lw=2)
    axes[0].annotate(f"mean = {m:.4f}", xy=(m, axes[0].get_ylim()[1] * 0.9),
                     xytext=(m - 0.45, axes[0].get_ylim()[1] * 0.9), fontsize=10, color="#b85450",
                     arrowprops=dict(arrowstyle="->", color="#b85450"))
    axes[0].set_xlabel("Top-five feature overlap between consecutive hours")
    axes[0].set_ylabel("Consecutive-hour pairs")
    axes[0].grid(axis="y", alpha=0.3); axes[0].set_axisbelow(True)

    axes[1].hist(stab["cosine_similarity"].dropna(), bins=40, color="#2f7d54", edgecolor="white")
    m2 = stab["cosine_similarity"].mean()
    axes[1].axvline(m2, color="#b85450", ls="--", lw=2)
    axes[1].set_xlabel("Cosine similarity between consecutive-hour SHAP vectors")
    axes[1].set_ylabel("Consecutive-hour pairs")
    axes[1].grid(axis="y", alpha=0.3); axes[1].set_axisbelow(True)

    fig.suptitle(f"n = {len(stab):,} consecutive-hour pairs", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig_4_10_stability.png", dpi=DPI)
    plt.close(fig)
    print(f"  Figure 4.10 written (mean overlap {m:.4f}, mean cosine {m2:.4f})")


def fig_5_1(out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for name, a, b in IMPORTANCE_SHIFT:
        colour = HIGHLIGHT.get(name, "#cccccc")
        width = 2.4 if name in HIGHLIGHT else 1.2
        z = 3 if name in HIGHLIGHT else 1
        ax.plot([0, 1], [a, b], "-o", color=colour, lw=width, ms=6, zorder=z)
        if name in HIGHLIGHT:
            ratio = b / a
            arrow = "\u2191" if ratio > 1 else "\u2193"
            ax.text(-0.03, a, f"{name}  {a:.3f}", ha="right", va="center",
                    fontsize=10, color=colour, fontweight="bold")
            ax.text(1.03, b, f"{b:.3f}  {arrow} {ratio:.2f}\u00d7", ha="left", va="center",
                    fontsize=10, color=colour, fontweight="bold")

    ax.set_xlim(-0.45, 1.45); ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hospital A\n(in-distribution)", "Hospital B\n(out-of-distribution)"], fontsize=11)
    ax.set_ylabel("Mean |SHAP value|")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.5, 1.02, "Grey lines: the remaining eight of the top ten features",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#666666")
    fig.tight_layout()
    fig.savefig(out / "fig_5_1_importance_shift.png", dpi=DPI)
    plt.close(fig)
    print("  Figure 5.1 written")


def fig_5_3(ctx, out: Path, threshold: float = 0.03, alpha: float = 0.05):
    from src.uncertainty.aps import build_prediction_sets_aps, calibrate_conformal_aps

    test_df = ctx["test_df"].copy()
    test_df["_prob"] = ctx["test_iso"]

    q = calibrate_conformal_aps(ctx["val_iso"], ctx["val_y"], alpha=alpha, seed=42)
    sets = build_prediction_sets_aps(ctx["test_iso"], q, seed=123)
    test_df["_size"] = sets.sum(axis=1)

    best, best_lead = None, -1
    for pid, g in test_df.groupby("patient_id"):
        g = g.sort_values("ICULOS")
        if g["SepsisLabel"].max() != 1 or not (12 <= len(g) <= 60):
            continue
        onset = g.loc[g["SepsisLabel"] == 1, "ICULOS"].min() + 6
        crossed = g.loc[g["_prob"] >= threshold, "ICULOS"]
        if crossed.empty:
            continue
        lead = onset - crossed.min()
        if 3 <= lead <= 14 and lead > best_lead:
            best, best_lead = pid, lead
    if best is None:
        print("  Figure 5.3 skipped: no suitable patient found")
        return

    g = test_df[test_df["patient_id"] == best].sort_values("ICULOS")
    hours = g["ICULOS"].to_numpy()
    onset = g.loc[g["SepsisLabel"] == 1, "ICULOS"].min() + 6
    alert = g.loc[g["_prob"] >= threshold, "ICULOS"].min()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 2.2, 0.5]})

    for col, colour, lbl in (("HR", "#b85450", "Heart rate (bpm)"),
                              ("SBP", "#4c78a8", "Systolic BP (mmHg)"),
                              ("Resp", "#2f7d54", "Respiratory rate (/min)")):
        if col in g.columns:
            axes[0].plot(hours, g[col], "-", color=colour, lw=1.6, label=lbl)
    axes[0].legend(fontsize=9, ncol=3, loc="upper left")
    axes[0].set_ylabel("Vital signs")
    axes[0].grid(alpha=0.3); axes[0].set_axisbelow(True)

    risk = np.clip(g["_prob"].to_numpy() * 100, 0.05, None)
    axes[1].plot(hours, risk, "-o", color="#08519c", lw=2, ms=4)
    axes[1].axhline(threshold * 100, color="#d79b00", ls="--", lw=1.5)
    axes[1].text(hours[0], threshold * 100 * 1.25, f"alert threshold {threshold*100:.0f}%",
                 fontsize=9, color="#d79b00")
    axes[1].set_ylabel("Calibrated risk (%)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.04, 130)
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)

    state_colours = {0: "#c7c7c7", 1: "#4c78a8", 2: "#b85450"}
    for h, sz in zip(hours, g["_size"]):
        axes[2].axvspan(h - 0.5, h + 0.5, color=state_colours.get(int(sz), "#c7c7c7"))
    axes[2].set_yticks([]); axes[2].set_ylabel("Conformal\nstate", fontsize=9)
    axes[2].set_xlabel("ICU hour")
    handles = [plt.Rectangle((0, 0), 1, 1, color=state_colours[k]) for k in (1, 2, 0)]
    axes[2].legend(handles, ["Committed to one class", "Uncertain (both classes)", "No confident class"],
                   loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, fontsize=9, frameon=False)

    for ax in axes:
        ax.axvline(alert, color="#d79b00", ls="-", lw=1.5, alpha=0.8)
        ax.axvline(onset, color="#b85450", ls="--", lw=1.8)

    span = hours.max() - hours.min()
    axes[0].text((alert - hours.min()) / span, 1.04, f"alert fires (hour {alert:.0f})",
                 transform=axes[0].transAxes, fontsize=9, color="#d79b00", ha="right", va="bottom")
    axes[0].text((onset - hours.min()) / span, 1.04, f"sepsis onset (hour {onset:.0f})",
                 transform=axes[0].transAxes, fontsize=9, color="#b85450", ha="left", va="bottom")

    fig.suptitle(f"Patient {best} — alert {best_lead:.0f} hours before documented onset", fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(out / "fig_5_3_case_study.png", dpi=DPI)
    plt.close(fig)
    print(f"  Figure 5.3 written (patient {best}, lead time {best_lead:.0f} h)")


ALL = ["4.1", "4.5", "4.6", "4.7", "4.8", "4.9", "4.10", "5.1", "5.3"]
NEEDS_DATA = {"4.1", "4.5", "4.8", "4.9", "4.10", "5.3"}


def main():
    ap = argparse.ArgumentParser(description="Generate Chapter 4 and 5 figures.")
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    ap.add_argument("--model", type=Path, default=Path("outputs/lightgbm_baseline.txt"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    ap.add_argument("--figures", type=str, default=None,
                     help="Comma-separated list, e.g. 4.5,4.9,5.1. Default: all.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--stability-patients", type=int, default=800,
                     help="Patients sampled for Figure 4.10; 0 uses the full test split.")
    args = ap.parse_args()

    wanted = ALL if (args.all or not args.figures) else [f.strip() for f in args.figures.split(",")]
    unknown = [f for f in wanted if f not in ALL]
    if unknown:
        raise SystemExit(f"Unknown figures: {unknown}. Choose from {ALL}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ctx = None
    if any(f in NEEDS_DATA for f in wanted):
        ctx = prepare(args.data_root, args.splits, args.model)

    print(f"\nGenerating: {', '.join(wanted)}")
    if "4.1" in wanted: fig_4_1(ctx, args.out_dir)
    if "4.5" in wanted: fig_4_5(ctx, args.out_dir)
    if "4.6" in wanted: fig_4_6(args.out_dir)
    if "4.7" in wanted: fig_4_7(args.out_dir)
    if "4.8" in wanted: fig_4_8(ctx, args.out_dir)
    if "4.9" in wanted: fig_4_9(ctx, args.out_dir)
    if "4.10" in wanted: fig_4_10(ctx, args.out_dir, args.stability_patients)
    if "5.1" in wanted: fig_5_1(args.out_dir)
    if "5.3" in wanted: fig_5_3(ctx, args.out_dir)

    print(f"\nDone. Figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
