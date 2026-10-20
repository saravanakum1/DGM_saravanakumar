## Context

The project uses PTB-XL (21,799 records, 12-lead, 100 Hz, 1000 timesteps) for unsupervised ECG anomaly detection. The VAE is the first of three models (VAE, LSTM, DDPM). Trained artifacts (weights, loss history, norm stats) are consumed by the Streamlit dashboard at inference time. The dataset is not bundled — it is downloaded on demand via `kagglehub`.

Training is expected to run either locally (Mac, GPU optional) or in Google Colab. The same `config.toml` controls both environments by pointing `data.dataset_dir` at the appropriate path.

## Goals / Non-Goals

**Goals:**
- Implement a working 1D CNN VAE that trains on PTB-XL NORM records only
- Make all hyperparameters and paths configurable via `config.toml` (no hardcoded values)
- Produce artifacts (`vae_weights.pt`, `vae_loss.json`, `norm_stats.json`) in `artifacts/` consumable by the dashboard
- Support dataset acquisition via `kagglehub.dataset_download("physionet/ptbxl-electrocardiography-database")` into `dataset/`
- Follow the project directory layout from `DASHBOARD_ARCHITECTURE.md` (with `src/train/`, `src/eval/`, `backend/preprocess/` additions)

**Non-Goals:**
- LSTM or DDPM implementation (separate changes)
- Dashboard UI (separate change)
- Real-time inference or streaming
- Model retraining from within the dashboard

## Decisions

### D1: Dataset acquisition via kagglehub
**Decision**: Use `kagglehub.dataset_download("physionet/ptbxl-electrocardiography-database")` as the sole dataset download mechanism. The returned path is written into `config.toml` under `[data] dataset_dir`. A helper script `src/data/download_dataset.py` wraps this call and prints the resolved path.

**Why over alternatives**:
- `wfdb.dl_database` requires manual PhysioNet credentialing; kagglehub works with a single Kaggle API key
- Direct download from PhysioNet is slow and requires wget/curl setup in Colab
- kagglehub caches across runs; re-running doesn't re-download

**Colab compatibility**: In Colab, run `!pip install kagglehub` and upload `kaggle.json`; the same `download_dataset.py` script works unchanged.

### D2: TOML configuration (`config.toml`)
**Decision**: All hyperparameters, paths, and thresholds live in `config.toml` at the repo root. Python code loads it with `tomllib` (stdlib in Python 3.11+). No `argparse` or env-var overrides for primary configuration — TOML is the single source of truth.

**Sections**:
```toml
[data]
dataset_dir = "dataset/ptb-xl"   # resolved from kagglehub download
sampling_rate = 100
val_split = 0.1

[model]
latent_dim = 128
encoder_channels = [32, 64, 128]
encoder_kernels = [7, 5, 3]
encoder_strides = [2, 2, 2]
beta = 1.0

[training]
epochs = 50
batch_size = 64
learning_rate = 1e-3
device = "cpu"           # or "cuda", "mps"

[anomaly]
threshold_percentile = 95   # 95 or 99

[artifacts]
output_dir = "artifacts"
weights_file = "vae_weights.pt"
loss_file = "vae_loss.json"
norm_stats_file = "norm_stats.json"
```

### D3: Project structure additions to DASHBOARD_ARCHITECTURE.md
**Decision**: Extend the existing layout with:
```
src/
  data/
    download_dataset.py       ← kagglehub download + prints resolved path
  train/
    train_vae.py              ← VAE training entry point
  eval/
    eval_vae.py               ← VAE evaluation (AUROC, PR curve, score dist)
backend/
  preprocess/
    __init__.py
    pipeline.py               ← load, filter, normalize, split
  models/
    vae.py
config.toml                   ← hyperparameters + paths
dataset/                      ← kagglehub download target (gitignored)
artifacts/                    ← trained weights + metrics (gitignored for weights)
```

### D4: VAE architecture — 1D CNN matching the spec exactly
**Decision**: Implement the exact architecture from `PTB_XL_VAE_Architecture.md`. Decoder output is interpolated to `(batch, 12, 1000)` to correct off-by-one from strided convolutions. No architectural changes or experiments at this stage.

### D5: Anomaly scoring
**Decision**: Reconstruction MSE per sample (mean over all timesteps and leads). Threshold set at the `anomaly.threshold_percentile` percentile of NORM val reconstruction errors. This threshold is saved in `norm_stats.json` alongside global mean/std so the dashboard can apply it without recomputing.

## Risks / Trade-offs

- **kagglehub Colab auth**: Requires `kaggle.json` upload or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars in Colab. → Mitigation: document in `src/data/download_dataset.py` docstring.
- **MPS device (Apple Silicon)**: Some ops may fall back to CPU. → Mitigation: `device` is configurable; default to `"cpu"` for portability.
- **Decoder shape mismatch**: Strided ConvTranspose1d produces sizes that differ by ±1 from target. → Mitigation: explicit `F.interpolate(..., size=1000)` as final decoder step (per spec).
- **Long training time locally**: 50 epochs × ~14k NORM records may be slow on CPU. → Mitigation: `training.device = "mps"` or run in Colab with GPU; epochs are configurable.

## Open Questions

_(none — design is fully constrained by the VAE architecture spec and dashboard architecture)_
