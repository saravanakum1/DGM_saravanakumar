# Training

All train scripts read from `config.toml` and use `get_spec(config)` internally to load the correct dataset class and split logic.

## Prerequisites

1. Preprocessing complete — `<dataset_dir>/npy/` populated
2. `config.toml` pointing to the right dataset and `dataset_dir`

## Run

=== "DDPM"

    ```bash
    uv run python src/train/train_ddpm.py
    ```

    Saves:
    - `artifacts/ddpm_ema_weights.pt` — EMA model weights
    - `artifacts/norm_stats.json` — per-lead mean/std + `ddpm_threshold`
    - `artifacts/splits_cache.json` — train/val/test split cache

=== "Flow Matching"

    ```bash
    uv run python src/train/train_flow_matching.py
    ```

    Saves:
    - `artifacts/fm_ema_weights.pt`
    - `artifacts/norm_stats.json` (adds `fm_threshold`)
    - `artifacts/splits_cache.json`

=== "VAE"

    ```bash
    uv run python src/train/train_vae.py
    ```

    Saves:
    - `artifacts/vae_weights.pt`
    - `artifacts/norm_stats.json` (adds `threshold`)
    - `artifacts/splits_cache.json`

## What happens on first run

1. `build_splits()` reads the metadata CSV and returns `(train, val, test)` record lists
2. `fit_norm_stats()` computes per-lead mean/std over NORMAL training records
3. Split cache is written to `artifacts/splits_cache.json` — subsequent runs (and eval) load this directly
4. Training loop runs for `config.training.epochs` epochs
5. EMA weights and anomaly threshold are saved at the end

## Kaggle

On Kaggle, the split cache from a prior training run can be bundled as a dataset and loaded automatically. The scripts check `/kaggle/input/datasets/sharav95/ecg-ad-comparison/artifacts/splits_cache.json` before rebuilding splits.

## Training configuration

Key `config.toml` settings:

```toml
[training]
batch_size    = 64
epochs        = 100
learning_rate = 1e-4
device        = "cuda"   # or "cpu" / "mps"

[anomaly]
threshold_percentile = 95
```
