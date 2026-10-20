## Why

The ECG-ADComparison project requires a trained VAE model as the first of three anomaly detection backends. Without the VAE implementation, the dashboard has no model to run inference against, and end-to-end evaluation metrics cannot be produced.

## What Changes

- Add a TOML-based configuration file (`config.toml`) for all model hyperparameters, training settings, and data paths — making experiments reproducible and adjustable without code changes
- Implement shared preprocessing pipeline under `backend/preprocess/` (data loading from local PTB-XL at `physionet.org/files/ptb-xl/1.0.3/records100/`, bandpass filtering, per-lead normalization, train/val/test splitting)
- Implement 1D CNN VAE at `backend/models/vae.py`: encoder, reparameterization, decoder, forward pass
- Implement training entry point at `src/train/train_vae.py` with per-epoch loss logging, saving history to `artifacts/vae_loss.json` and weights to `artifacts/vae_weights.pt`
- Implement evaluation entry point at `src/eval/eval_vae.py`: AUROC, precision-recall curve, per-class anomaly score distribution
- Save normalization stats to `artifacts/norm_stats.json` for dashboard inference reuse
- Update `DASHBOARD_ARCHITECTURE.md` project structure to reflect the `backend/preprocess/` directory layout and `src/train/` / `src/eval/` layout
- No changes to `pyproject.toml` — dependencies are already declared

## Capabilities

### New Capabilities

- `vae-model`: 1D CNN Variational Autoencoder — encoder/decoder architecture, reparameterization, forward pass, anomaly scoring via reconstruction error
- `shared-preprocessing`: PTB-XL data loading from local `records100/`, bandpass filtering, per-lead normalization, train/val/test split using PTB-XL predefined folds (fold 10 = test)
- `toml-config`: TOML-based hyperparameter configuration covering model architecture, optimizer, training schedule, anomaly thresholding, and data paths
- `vae-training`: Training loop, MSE + beta-KL loss, per-epoch logging, loss history export to `artifacts/vae_loss.json`, model checkpoint to `artifacts/vae_weights.pt`
- `vae-evaluation`: AUROC, precision-recall curve, anomaly score distribution by class (NORM / MI / STTC / CD / HYP)

### Modified Capabilities

_(none — initial implementation)_

## Impact

- **New files**: `config.toml`, `src/train/train_vae.py`, `src/eval/eval_vae.py`, `backend/preprocess/__init__.py`, `backend/preprocess/pipeline.py`, `backend/models/vae.py`, `artifacts/vae_weights.pt`, `artifacts/vae_loss.json`, `artifacts/norm_stats.json`
- **Updated files**: `dashboard_architecture/DASHBOARD_ARCHITECTURE.md` (project structure section)
- **Dataset**: PTB-XL available locally at `physionet.org/files/ptb-xl/1.0.3/records100/`; path is configurable in `config.toml` so training can also be run in Google Colab by pointing to a Colab-mounted or Drive path
- **Downstream**: Dashboard's `backend/inference.py` will load `vae_weights.pt` and `norm_stats.json`; `artifacts/vae_loss.json` feeds Page 3 Training Diagnostics
