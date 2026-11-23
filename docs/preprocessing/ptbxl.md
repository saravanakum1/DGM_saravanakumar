# Preprocessing — PTB-XL

Script: `scripts/process_ptbxl.py`

## Dataset layout expected

```
<dataset_dir>/
├── ptbxl_database.csv
├── scp_statements.csv
└── records500/
    ├── 00000/
    │   ├── 00001_hr.dat
    │   ├── 00001_hr.hea
    │   └── ...
    └── ...
```

## Download (500 Hz records only)

```bash
aws s3 sync --no-sign-request \
  s3://physionet-open/ptb-xl/1.0.3/records500/ \
  /path/to/ptbxl/records500/

# Also download the metadata CSVs
aws s3 cp --no-sign-request \
  s3://physionet-open/ptb-xl/1.0.3/ptbxl_database.csv /path/to/ptbxl/
aws s3 cp --no-sign-request \
  s3://physionet-open/ptb-xl/1.0.3/scp_statements.csv /path/to/ptbxl/
```

!!! note
    Download only `records500/` — the 100 Hz `records100/` folder is not used.

## Run preprocessing

```bash
uv run python scripts/process_ptbxl.py \
  --dataset-dir /path/to/ptbxl \
  --workers 8
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--dataset-dir` | from `config.toml` | Path to PTB-XL root |
| `--workers` | `min(cpu_count, 8)` | Parallel worker processes |
| `--overwrite` | off | Re-process already-completed records |

## Output

```
<dataset_dir>/npy/
├── 00001_hr.npy    # shape (12, 5000)  float32
├── 00002_hr.npy
└── ...
```

~21,000 files, ~500 MB total.

## Smoke test

After running, verify a sample file:

```python
import numpy as np
sig = np.load("path/to/npy/00009_hr.npy")
print(sig.shape)   # (12, 5000)
print(sig.dtype)   # float32
print(sig.mean())  # ≈ 0.0
print(sig.std())   # ≈ 1.0
```

## Label mapping

Labels are derived from `scp_statements.csv` diagnostic superclasses:

| Label | Meaning |
|---|---|
| `NORM` | Normal sinus rhythm |
| `MI` | Myocardial infarction |
| `STTC` | ST/T-wave changes |
| `CD` | Conduction disturbance |
| `HYP` | Hypertrophy |

- **Train/Val**: NORM records from folds 1–9
- **Test**: All superclasses from fold 10 (stratified)
