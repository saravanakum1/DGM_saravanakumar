# Preprocessing Overview

Preprocessing is **offline** — run once per dataset, then training loads `.npy` files directly with no signal processing at training time.

## Pipeline

Each raw WFDB record is processed as follows:

```
WFDB record (500 Hz)
    │
    ▼
Bandpass filter   0.5 – 40 Hz   (Butterworth order 4, zero-phase)
    │
    ▼
Notch filter      50 Hz (PTB-XL) / 60 Hz (MIMIC)   (powerline noise)
    │
    ▼
Z-score per lead  (mean=0, std=1)
    │
    ▼
Save as .npy      shape (12, 5000)  float32
```

## Output

| Property | Value |
|---|---|
| Shape | `(12, 5000)` — 12 leads × 5000 samples |
| Dtype | `float32` |
| Duration | 10 seconds at 500 Hz |
| Location | `<dataset_dir>/npy/<record_id>.npy` |

## Why offline?

- Training startup is just `np.load()` — no WFDB or scipy at training time
- Filters and z-score are applied identically every run — fully reproducible
- Safe to interrupt and resume — already-processed files are skipped

## Dataset differences

| | PTB-XL | MIMIC-IV-ECG |
|---|---|---|
| Notch frequency | 50 Hz (European powerline) | 60 Hz (US powerline) |
| Record naming | `<stem>.npy` e.g. `00001_hr.npy` | `<study_id>.npy` e.g. `800001234.npy` |
| Metadata CSV | `ptbxl_database.csv` | `machine_measurements.csv` |
