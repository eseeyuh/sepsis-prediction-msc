# Evaluation notes

This file holds the methodological detail behind the README's Results and Limitations
sections. Each statement comes from reading the code in `src/`. None of it comes from new
experiments.

## Data use by split

| Split | Used for |
|---|---|
| Train (70% of patients) | Imputation medians, LightGBM fitting |
| Validation (15%) | LightGBM early stopping on AUROC, calibrator fitting, conformal quantile (LAC and APS), alert-threshold search on utility |
| Test (15%) | All reported metrics and figures. The top-decile risk cut-off used to stratify the faithfulness analysis is computed on test predictions. It is used only to group samples, never to fit anything. |

The pipeline avoids these sources of leakage:

- Patients do not overlap between splits (`assert_no_patient_overlap`).
- Imputation statistics are fitted on train only.
- Forward fill, rolling windows and trend features use only the current and earlier hours
  of the same patient.

The pipeline has these risks:

- **One validation split serves several purposes.** The conformal nonconformity scores are
  computed on the same validation points used to fit the calibrator. Those calibrated
  scores are therefore not exchangeable with calibrated test scores. This matters most for
  isotonic regression, which can overfit its fitting data. A separate conformal calibration
  split would be needed for the standard split-conformal argument.
- **Hour-level exchangeability.** Calibration and test sets are both patient-hours. Hours
  within a stay are strongly dependent, and long stays dominate. The marginal coverage
  guarantee assumes exchangeable points, so it does not formally hold here. Coverage
  figures are empirical estimates.
- **Marginal, not class-conditional coverage.** About 98% of hours are negative. Average
  coverage can therefore meet the target while coverage on positive hours is far lower. No
  per-class coverage is computed.
- **Validation-tuned threshold.** The utility-optimal threshold is chosen on validation and
  then applied to test, which is correct. The same validation split also selected the
  number of boosting rounds.

## Splitting is patient-level, not temporal

`make_patient_splits` draws a random sample of patients, stratified by whether each
patient ever has a positive label. Hospitals A and B are pooled. This guards against
memorising individual patients. It does not test performance on later admissions, because
the public data has no calendar time. The cross-site experiment is the only shift test.

## Cross-site experiment

- `src/cross_site.py` builds a separate split of hospital-A patients only
  (`configs/splits_hospitalA.json`: 14,235 / 3,050 / 3,051). It trains a new model on that
  split.
- Hospital-B rows are preprocessed with hospital-A training medians. Nothing is refitted
  on B.
- Only LAC at one α (default 0.10) is evaluated under shift. APS is not.
- Global SHAP importance (mean |SHAP|) is compared between the two sites.
- The public release contains only hospitals A and B. Hospital C was the Challenge's hidden
  test set.

## Utility score versus the official Challenge metric

`src/evaluation/metrics.py` reimplements the Challenge utility. It uses the same
parameters (dt_early −12, dt_optimal −6, dt_late +3, max reward 1, missed-sepsis penalty
−2, false-positive penalty −0.05). It also defines onset the same way: the first positive
label + 6 h. Compared with the official
[`evaluate_sepsis_score.py`](https://github.com/physionetchallenges/evaluation-2019/blob/master/evaluate_sepsis_score.py),
it differs in two ways:

1. **Early alarms.** Before onset − 12 h, a positive prediction for a septic patient scores
   0 here. The official code scores −0.05 (`max(..., u_fp)`).
2. **Late hours.** More than 3 h after onset, a missed hour scores −2 here. The official
   code excludes those hours and scores them 0.

The normalised values are therefore labelled "challenge-style utility". They should not be
compared with official Challenge scores.

## Calibration metrics

- ECE uses 10 equal-width bins on patient-hours. Most predictions fall in the lowest bin,
  so ECE mostly measures calibration near zero. Upper bins hold few samples and are noisy;
  the isotonic panel of Figure 4.1 shows this.
- `is_unbalance: true` reweights the positive class, which inflates raw probabilities.
  Temperature scaling, in this implementation, divides logits by T without an intercept and
  cannot correct that shift. Platt scaling and isotonic regression can.

## Explanation evaluation

- SHAP values come from `shap.TreeExplainer` on the raw booster, so they are in log-odds
  units of the uncalibrated model. Platt and isotonic calibration are monotone: they keep
  the risk ranking but change the probability scale.
- Deletion and insertion replace features with NaN, and LightGBM routes NaN through its
  learned missing-value branches. After imputation, most clinical columns are never NaN in
  training. The perturbed inputs are therefore out of distribution.
- Faithfulness is computed on 200 sampled test hours per stratum (default). Stability
  covers consecutive hours of the same patient only (ICULOS difference of 1).
- `ICULOS` and `hours_since_admission` are identical features, so importance is split
  between them. Both head the importance table recorded in `make_figures.py`
  (`IMPORTANCE_SHIFT`).

## Provenance of figures in `outputs/figures/`

| Figure(s) | Source | Status |
|---|---|---|
| `reliability_diagram.png`, `conformal_tradeoff.png`, `decision_curve.png`, `shap_summary.png` | `src/generate_figures.py` | Computed from the trained model on the pooled test split |
| `fig_4_1`, `fig_4_5`, `fig_4_8`, `fig_4_9`, `fig_4_10` | `src/demo/make_figures.py` | Computed from the trained model on the pooled test split. Fig 4.10 samples 800 patients. |
| `fig_5_3_case_study.png` | `src/demo/make_figures.py` | Single test patient, illustrative |
| `fig_4_6`, `fig_4_7`, `fig_5_1` | `src/demo/make_figures.py` | Plotted from **hard-coded constants** transcribed from dissertation tables. They are not regenerated from a saved run. |
| `fig_3_2`, `fig_3_4` | `src/demo/make_ch3_figures.py` | Descriptive plots of the raw data (example record, missingness) |
| `outputs/demo/index.html`, `outputs/demo_data.json` | `src/demo/export_demo_data.py` | Illustrative dashboard for four selected test patients. Not an evaluation. |

No metrics file (JSON or CSV) is saved by any script. To audit a number, rerun the
relevant entry point; see [`reproduction.md`](reproduction.md).
