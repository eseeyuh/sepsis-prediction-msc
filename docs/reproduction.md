# Reproduction guide

Run all commands from the repository root. Every entry point is a module (`python -m src.…`),
so the `src` package must be importable from the working directory.

## Environment

- The recorded environment is Python 3.14.6 on Windows. The frozen package list is in
  [`../appendix_d_versions.txt`](../appendix_d_versions.txt), a UTF-16 `pip freeze`. Key
  versions: lightgbm 4.6.0, shap 0.52.0, scikit-learn 1.9.0, numpy 2.4.6, pandas 2.3.3.
- [`../requirements.txt`](../requirements.txt) is unpinned. Results may change slightly with
  newer library versions.
- The optional demo alert server (`src/demo/alert_server.py`, `src/demo/test_twilio.py`)
  also needs `flask`, `python-dotenv` and `twilio`. These are **not** in
  `requirements.txt`. It reads Twilio credentials from environment variables or a local
  `.env` file, which is git-ignored.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Download the PhysioNet/CinC Challenge 2019 training data as described in
[`../data/README.md`](../data/README.md). Then place the `.psv` files at these paths:

```
data/raw/training_setA/*.psv   # 20,336 patients
data/raw/training_setB/*.psv   # 20,000 patients
```

`data/` is git-ignored apart from its README. Raw data is never committed.

## Pipeline

| Step | Command | Needs | Produces |
|---|---|---|---|
| 1. Train | `python -m src.train` | raw data, `configs/splits.json`, `configs/lightgbm.yaml` | `outputs/lightgbm_baseline.txt` and `.features.json`, plus val/test metrics printed to the console |
| 2. Calibrate | `python -m src.calibrate --method isotonic` | step 1 | ECE/Brier before and after calibration, utility at 0.5 and at the validation-tuned threshold (console) |
| 3. Conformal | `python -m src.quantify_uncertainty [--alpha 0.1 ...]` | step 1 | LAC and APS coverage, set sizes and composition (console) |
| 4. Explain | `python -m src.explain [--faithfulness-samples 200] [--stability-k 5]` | step 1 | global SHAP importance, faithfulness, stability (console) |
| 5. Decision curve | `python -m src.decision_curve_analysis` | step 1 | net-benefit summary (console) |
| 6. Cross-site | `python -m src.cross_site [--alpha 0.1]` | raw data, `configs/splits_hospitalA.json` | trains its own hospital-A model in memory and prints the A-vs-B comparison |
| 7. Figures | `python -m src.generate_figures` and `python -m src.demo.make_figures --all` | step 1 | PNGs in `outputs/figures/` |
| 8. Data figures | `python -m src.demo.make_ch3_figures` | raw data | example record and missingness figures |
| 9. Demo | `python -m src.demo.export_demo_data` then `python -m src.demo.build_standalone` | step 1 | `outputs/demo_data.json`, `outputs/demo/index.html` |

Notes:

- `outputs/` is git-ignored except for `outputs/figures/`, `outputs/demo/` and
  `outputs/demo_data.json`. The trained model is **not** committed, so you must run step 1
  before steps 2–5, 7 and 9.
- `configs/splits.json` and `configs/splits_hospitalA.json` are committed. If either file
  is missing, the scripts create a new split with seed 42. If they exist, they are reused.
- Every script reloads and re-preprocesses the full dataset from raw files. Nothing caches
  processed data.
- `src.calibrate` defaults to `--method temperature`. The other scripts default to
  `--calibration-method isotonic`, which matches the reported analysis. Pass the method
  explicitly.
- No script writes metrics to disk. Capture the console output if you need a record:
  `python -m src.calibrate --method isotonic | Tee-Object outputs\calibrate.log`.
- `src.demo.make_figures` plots Figures 4.6, 4.7 and 5.1 from hard-coded constants
  transcribed from the dissertation. It does not recompute them. See
  [`evaluation-notes.md`](evaluation-notes.md).

## Tests

```powershell
python -m pytest
```

The suite has 135 unit tests across 15 files. They use small synthetic inputs and need
neither the dataset nor a trained model. There is no CI configuration, so test status is
not tracked automatically.
