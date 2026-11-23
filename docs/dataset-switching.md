# Dataset Switching

Switching between PTB-XL and MIMIC-IV-ECG requires changing **one line** in `config.toml`.

## How to switch

=== "PTB-XL"

    ```toml
    [data]
    name         = "ptbxl"
    dataset_dir  = "/path/to/ptbxl"
    sampling_rate = 500
    val_split    = 0.1
    ```

=== "MIMIC-IV-ECG"

    ```toml
    [data]
    name         = "mimic-iv-ecg"
    dataset_dir  = "/path/to/mimic-iv-ecg"
    sampling_rate = 500
    val_split    = 0.1
    ```

Then re-run training and evaluation — no code changes needed.

## What changes automatically

| Aspect | PTB-XL | MIMIC-IV-ECG |
|---|---|---|
| Metadata CSV | `ptbxl_database.csv` | `machine_measurements.csv` |
| Normal label | `NORM` | `NORMAL` |
| Anomaly classes | MI, STTC, CD, HYP | AF, ST, OTHER |
| Test split strategy | Stratified fold 10 | Last 10% of subjects by ID |
| Notch filter (offline) | 50 Hz | 60 Hz |
| Kaggle input path | `sharav95/ecg-ptbxl-data` | `sharav95/ecg-mimic-iv-data` |

## Architecture

The registry pattern (`backend/preprocess/registry.py`) wires everything together:

```
config.toml  [data] name = "ptbxl"
        │
        ▼
    get_spec(config)
        │
        ▼
    DatasetSpec
    ├── dataset_class       → PTBXLDataset / MIMICDataset
    ├── build_splits_fn     → ptbxl.build_splits / mimic.build_splits
    ├── fit_norm_stats_fn   → ptbxl.fit_norm_stats / mimic.fit_norm_stats
    ├── normal_label        → "NORM" / "NORMAL"
    ├── class_colors        → per-dataset color map
    └── seq_len / sampling_rate
```

Train and eval scripts call `spec.*` throughout — they never import `ptbxl` or `mimic` directly.

## Adding a new dataset

1. Create `backend/preprocess/<name>.py` following the same structure as `ptbxl.py`
2. Register it in `backend/preprocess/registry.py` under `_REGISTRY`
3. Write an offline preprocessing script `scripts/process_<name>.py`
4. Set `name = "<name>"` in `config.toml`
