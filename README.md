# Explainable Sepsis Prediction with Calibrated Uncertainty for ICU Time Series

MSc dissertation project. Early sepsis prediction on ICU time series that combines
strong discrimination, calibrated probabilities, principled uncertainty quantification,
and faithful, stable explanations — with an explicit stress test under hospital-site shift.

**Dataset:** [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
**Author:** Shaidarov Daryn
**Supervisor:** Honghai Liu
**Course:** MSc Machine Learning and Artificial Intelligence, 2025–26

---

## Project structure

```
sepsis-prediction-msc/
├── data/                   # raw + processed data (gitignored, not committed)
├── notebooks/              # exploration and analysis notebooks
├── src/
│   ├── data/                # loading, preprocessing, Sepsis-3 labelling
│   ├── models/               # LightGBM / GRU / Transformer implementations
│   ├── evaluation/           # metrics: AUROC, AUPRC, utility score, ECE, coverage
│   └── utils/                 # shared helpers
├── configs/                 # Hydra configuration files
├── tests/                    # unit tests (data pipeline, splits, metrics)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

This project uses the **PhysioNet/CinC Challenge 2019** dataset (openly downloadable,
no credentialing required for the training data). See `data/README.md` for download
instructions and expected directory layout. Data is **not committed to this repository**
— see `.gitignore`.

## Status

Project just started. See `diary.md` for day-to-day progress notes.

## Ethics & data use

This is a retrospective analysis of secondary, de-identified data. No participants are
recruited or observed. Data use is governed by the PhysioNet Credentialed Health Data
License / Challenge terms of use. University ethics approval obtained via the
secondary-data route before any analysis began.
