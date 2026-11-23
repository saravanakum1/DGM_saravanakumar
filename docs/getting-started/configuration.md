# Configuration

All settings live in `config.toml` at the project root.

## `[data]` section

```toml
[data]
name         = "ptbxl"       # "ptbxl" or "mimic-iv-ecg"
dataset_dir  = "/path/to/dataset"
sampling_rate = 500
val_split    = 0.1
```

| Key | Description |
|---|---|
| `name` | Dataset to use. Switching this one value is all that's needed to change datasets. |
| `dataset_dir` | Absolute path to the dataset root. Must contain the `npy/` subfolder produced by preprocessing. |
| `sampling_rate` | Fixed at 500 Hz for both datasets. Do not change. |
| `val_split` | Fraction of NORMAL training records held out for validation. |

## `[training]` section

```toml
[training]
batch_size   = 64
epochs       = 100
learning_rate = 1e-4
device       = "cpu"    # "cpu", "cuda", or "mps"
```

## `[artifacts]` section

```toml
[artifacts]
output_dir       = "artifacts"
weights_file     = "vae_weights.pt"
norm_stats_file  = "norm_stats.json"
```

## `[anomaly]` section

```toml
[anomaly]
threshold_percentile = 95   # percentile of NORMAL val scores used as anomaly threshold
```

## Kaggle environment

On Kaggle, `dataset_dir` is automatically resolved from the mounted input path — no change to `config.toml` needed. The train scripts check for the Kaggle path first before falling back to `dataset_dir`.
