# Data

This project uses the **PhysioNet/Computing in Cardiology Challenge 2019** dataset.

- Source: https://physionet.org/content/challenge-2019/1.0.0/
- Access: listed by PhysioNet as "Challenge Open Access". Check the licence terms on the dataset page before use.
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
```

No processed data is written to disk. Every entry point preprocesses the raw files in memory.

## Columns (per hour, per patient)

Vitals: `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2`
Labs: `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`,
`Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`,
`Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`,
`PTT`, `WBC`, `Fibrinogen`, `Platelets`
Demographics: `Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`, `ICULOS`
Label: `SepsisLabel`. For septic patients it is 1 when t ≥ t_sepsis − 6, so labels are shifted
six hours ahead of the Sepsis-3-based onset time. Otherwise it is 0.

## Note on hospital system C

The original Challenge held out a third hospital system (C) as a hidden test set for
the live competition. It is **not** part of the public release. The cross-site analysis
(`src/cross_site.py`) therefore trains on hospital A and evaluates on all of hospital B.
With only two sites, this is a limited test of site shift.
