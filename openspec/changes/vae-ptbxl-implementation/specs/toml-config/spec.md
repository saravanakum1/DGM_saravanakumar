## ADDED Requirements

### Requirement: config.toml is the single source of truth for all parameters
A `config.toml` file at the repo root SHALL define all model hyperparameters, training settings, data paths, and artifact paths. Python code SHALL load it using `tomllib` (Python 3.11+ stdlib). No hyperparameters SHALL be hardcoded in any source file. Command-line overrides are out of scope.

#### Scenario: Config loads without error on a fresh checkout
- **WHEN** `tomllib.load(open("config.toml", "rb"))` is called from the repo root
- **THEN** no exception is raised and all required top-level sections are present

#### Scenario: Missing config.toml raises a clear error
- **WHEN** a training script is run and `config.toml` does not exist
- **THEN** a `FileNotFoundError` with a message indicating the missing config file is raised before any training begins

---

### Requirement: config.toml defines the [data] section
The `[data]` section SHALL include:
- `dataset_dir` (string): local path to downloaded PTB-XL dataset (set after running `download_dataset.py`)
- `sampling_rate` (int): 100
- `val_split` (float): fraction of NORM folds 1–9 held out for validation (default 0.1)

#### Scenario: dataset_dir points to a directory containing ptbxl_database.csv
- **WHEN** `config.toml` is loaded and `data.dataset_dir` is resolved
- **THEN** the path `{dataset_dir}/ptbxl_database.csv` exists on disk

---

### Requirement: config.toml defines the [model] section
The `[model]` section SHALL include:
- `latent_dim` (int): size of the VAE latent space (default 128)
- `encoder_channels` (list of int): output channels per encoder Conv1d layer (default [32, 64, 128])
- `encoder_kernels` (list of int): kernel sizes per layer (default [7, 5, 3])
- `encoder_strides` (list of int): strides per layer (default [2, 2, 2])
- `beta` (float): KL loss weight in total loss (default 1.0)

#### Scenario: Model instantiated with config values produces correct latent dim
- **WHEN** the VAE is instantiated using values from the `[model]` section of `config.toml`
- **THEN** the latent vectors produced by the encoder have dimension equal to `model.latent_dim`

---

### Requirement: config.toml defines the [training] section
The `[training]` section SHALL include:
- `epochs` (int): number of training epochs (default 50)
- `batch_size` (int): mini-batch size (default 64)
- `learning_rate` (float): Adam optimizer LR (default 1e-3)
- `device` (string): PyTorch device string — `"cpu"`, `"cuda"`, or `"mps"` (default `"cpu"`)

#### Scenario: Training loop uses batch_size from config
- **WHEN** training is started with `batch_size = 32` in `config.toml`
- **THEN** each DataLoader batch has at most 32 samples

---

### Requirement: config.toml defines the [anomaly] section
The `[anomaly]` section SHALL include:
- `threshold_percentile` (int): percentile of NORM val reconstruction errors used as anomaly threshold (default 95; allowed values: 95 or 99)

#### Scenario: Threshold is computed at the configured percentile
- **WHEN** `threshold_percentile = 99` in `config.toml` and the threshold is computed over NORM val errors
- **THEN** the stored threshold equals `np.percentile(norm_val_errors, 99)`

---

### Requirement: config.toml defines the [artifacts] section
The `[artifacts]` section SHALL include:
- `output_dir` (string): directory to save all artifacts (default `"artifacts"`)
- `weights_file` (string): filename for trained VAE weights (default `"vae_weights.pt"`)
- `loss_file` (string): filename for loss history JSON (default `"vae_loss.json"`)
- `norm_stats_file` (string): filename for normalization stats JSON (default `"norm_stats.json"`)

#### Scenario: Artifacts are saved to the configured output_dir
- **WHEN** training completes with `output_dir = "artifacts"` in `config.toml`
- **THEN** `artifacts/vae_weights.pt`, `artifacts/vae_loss.json`, and `artifacts/norm_stats.json` all exist
