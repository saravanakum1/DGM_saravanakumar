## ADDED Requirements

### Requirement: Dataset is acquired via kagglehub
The system SHALL provide `src/data/download_dataset.py` that calls `kagglehub.dataset_download("physionet/ptbxl-electrocardiography-database")`, prints the resolved local path, and exits. The resolved path SHALL be set as `data.dataset_dir` in `config.toml`. Kagglehub caches downloads; re-running SHALL not re-download if already cached.

#### Scenario: Download script resolves and prints a valid local path
- **WHEN** `uv run python src/data/download_dataset.py` is executed with valid Kaggle credentials
- **THEN** a local directory path containing `ptbxl_database.csv` and `records100/` is printed to stdout

#### Scenario: Re-running download does not repeat network transfer
- **WHEN** `download_dataset.py` is run a second time after a successful first run
- **THEN** the script completes without downloading files again (kagglehub cache hit)

---

### Requirement: Preprocessing pipeline loads PTB-XL records at 100 Hz
The preprocessing pipeline at `backend/preprocess/pipeline.py` SHALL load ECG records from the path configured in `config.toml` using the `wfdb` library at 100 Hz sampling rate (folder `records100/`). Output shape per record SHALL be `(12, 1000)` — 12 leads, 1000 timesteps.

#### Scenario: Loaded record has correct shape
- **WHEN** a single PTB-XL record is loaded via the pipeline
- **THEN** the resulting numpy array has shape `(12, 1000)`

#### Scenario: Missing leads are handled via zero-fill
- **WHEN** a record has one or more NaN values in any lead
- **THEN** those positions are filled with 0.0 (zero-fill strategy)

---

### Requirement: Per-lead normalization to zero mean, unit variance
Each lead SHALL be normalized independently using the global mean and standard deviation computed from the NORM training split. Normalization stats SHALL be saved to `artifacts/norm_stats.json` as `{"mean": [...], "std": [...]}` with one value per lead (shape 12 each). At inference time, the same stats SHALL be loaded from `norm_stats.json`.

#### Scenario: Normalization stats are computed from NORM train split only
- **WHEN** `pipeline.fit_norm_stats(norm_train_records)` is called
- **THEN** mean and std are computed per lead over all NORM training records and saved to `artifacts/norm_stats.json`

#### Scenario: Normalized NORM train signals have approximately zero mean per lead
- **WHEN** normalization is applied to the NORM training split using the fitted stats
- **THEN** the per-lead mean of the normalized records is within 1e-3 of 0.0

---

### Requirement: Bandpass filter removes baseline wander and muscle artifact
A Butterworth bandpass filter (0.5–40 Hz, order 4) SHALL be applied to each lead before normalization. The filter SHALL use `scipy.signal.butter` and `sosfiltfilt` (zero-phase).

#### Scenario: Bandpass filter is applied before normalization
- **WHEN** the preprocessing pipeline processes a record
- **THEN** filtering is performed before per-lead normalization

#### Scenario: Filtered signal has reduced low-frequency content
- **WHEN** a synthetic signal with a 0.1 Hz component is passed through the bandpass filter
- **THEN** the power of the 0.1 Hz component in the output is at least 20 dB lower than the input

---

### Requirement: Train/val/test split uses PTB-XL predefined folds
The pipeline SHALL use the `strat_fold` column in `ptbxl_database.csv` to assign splits: fold 10 = test (all classes), folds 1–9 = pool. NORM records from folds 1–9 SHALL be split into train (90%) and val (10%) using `config.toml` `data.val_split`. The test set includes all classes (NORM + MI + STTC + CD + HYP).

Superclass labels SHALL be derived from the `scp_codes` column using the SCP-ECG label mapping in `scp_statements.csv` (diagnostic superclass column).

#### Scenario: Test set contains all five superclasses
- **WHEN** the pipeline generates the test split from fold 10
- **THEN** records from all five superclasses (NORM, MI, STTC, CD, HYP) are present

#### Scenario: Train set contains only NORM records
- **WHEN** the pipeline generates the training split
- **THEN** all records in the training split have superclass label NORM
