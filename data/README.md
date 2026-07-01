# Data

This project uses the **PhysioNet/Computing in Cardiology Challenge 2019** dataset.

- Source: https://physionet.org/content/challenge-2019/1.0.0/
- License: open access (no credentialing required for the training set)
- Format: one `.psv` (pipe-separated values) file per patient, one row per hour

## Download

```bash
cd data/
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/
```

Or via the PhysioNet web interface if `wget` access is restricted on your network.

## Expected layout after download

```
data/
├── raw/
│   ├── training_setA/       # hospital system A, one .psv per patient
│   └── training_setB/       # hospital system B, one .psv per patient
└── processed/                # created by src/data/preprocess.py — gitignored
```

## Columns (per hour, per patient)

Vitals: `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2`
Labs: `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`,
`Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`,
`Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`,
`PTT`, `WBC`, `Fibrinogen`, `Platelets`
Demographics: `Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`, `ICULOS`
Label: `SepsisLabel` (1 if sepsis within the next 6 hours, per Sepsis-3 criteria)

## Note on hospital system C

The original Challenge held out a third hospital system (C) as a hidden test set for
the live competition. It is **not** part of the public release. Any cross-site /
distribution-shift analysis in this project uses a held-out partition of hospitals
A and B as a proxy for site shift — this limitation is documented explicitly in the
dissertation.
