# Preprocessing — MIMIC-IV-ECG

Script: `scripts/process_mimic.py`

## Dataset layout expected

```
<dataset_dir>/
├── machine_measurements.csv
└── files/
    └── p0000/
        └── p00000001/
            └── s800001234/
                ├── 800001234.dat
                └── 800001234.hea
```

Path pattern: `files/p{subj[:4]}/p{subj}/s{study_id}/{study_id}`

## Download

MIMIC-IV-ECG requires credentialed PhysioNet access. Set credentials as environment variables:

```bash
export PHYSIONET_USER=your_username
export PHYSIONET_PASS=your_password
```

Then download with `wget` (recursive):

```bash
wget -r -N -c -np \
  --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
  https://physionet.org/files/mimic-iv-ecg/1.0/
```

Or use the Kaggle dataset if available.

## Run preprocessing

```bash
uv run python scripts/process_mimic.py \
  --dataset-dir /path/to/mimic-iv-ecg \
  --workers 8
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--dataset-dir` | from `config.toml` | Path to MIMIC-IV-ECG root |
| `--workers` | `min(cpu_count, 8)` | Parallel worker processes |
| `--overwrite` | off | Re-process already-completed records |

!!! note
    Records whose `.hea` file is not on disk are skipped automatically. Safe to run on a partial download.

## Output

```
<dataset_dir>/npy/
├── 800001234.npy    # shape (12, 5000)  float32
├── 800001235.npy
└── ...
```

## Lead ordering

MIMIC records may have leads in a non-standard order. The script maps by lead name to the canonical 12-lead order:

`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`

Missing leads (if any) are filled with zeros.

## Label mapping

Labels are derived from `machine_measurements.csv` report fields:

| Label | Condition |
|---|---|
| `NORMAL` | `report_0 == "Sinus rhythm"` AND `report_2 == "Normal ECG"` |
| `AF` | Report contains "atrial fibrillation" or "atrial flutter" |
| `ST` | Report contains "st elevation" or "st depression" |
| `OTHER` | Any other non-normal record with report text |

- **Train/Val**: NORMAL records from non-test subjects
- **Test**: All classes from the last 10% of subjects (sorted by subject ID — no patient leakage)
