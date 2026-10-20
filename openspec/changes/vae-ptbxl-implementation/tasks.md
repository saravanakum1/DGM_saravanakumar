## 1. Project Structure & Configuration

- [x] 1.1 Create `config.toml` at repo root with all sections: `[data]`, `[model]`, `[training]`, `[anomaly]`, `[artifacts]`
- [x] 1.2 Create `src/data/`, `src/train/`, `src/eval/` directories with `__init__.py` stubs
- [x] 1.3 Create `backend/preprocess/` directory with `__init__.py`
- [x] 1.4 Create `backend/models/` directory with `__init__.py`
- [x] 1.5 Create `artifacts/` directory and add `.gitkeep`; add `*.pt` to `.gitignore`
- [x] 1.6 Update `dashboard_architecture/DASHBOARD_ARCHITECTURE.md` project structure section to reflect `src/data/`, `src/train/`, `src/eval/`, `backend/preprocess/`, and `dataset/` directories
- [x] 1.7 Add `kagglehub` to `pyproject.toml` dependencies

## 2. Dataset Acquisition

- [x] 2.1 Implement `src/data/download_dataset.py`: call `kagglehub.dataset_download("physionet/ptbxl-electrocardiography-database")`, print resolved path, and exit
- [x] 2.2 Run `uv run python src/data/download_dataset.py` locally, confirm download resolves to `dataset/` (or kagglehub cache), update `config.toml` `data.dataset_dir` with the resolved path
- [x] 2.3 Verify `ptbxl_database.csv` and `records100/` are present under the resolved path

## 3. Shared Preprocessing Pipeline

- [x] 3.1 Implement `backend/preprocess/pipeline.py`: `load_record(record_id, dataset_dir) -> np.ndarray (12, 1000)` using `wfdb.rdsamp`; zero-fill NaN values
- [x] 3.2 Implement bandpass filter in `pipeline.py`: Butterworth order-4, 0.5–40 Hz, using `scipy.signal.butter` + `sosfiltfilt`; apply per lead
- [x] 3.3 Implement `build_splits(config) -> (train_ids, val_ids, test_ids_with_labels)`: read `ptbxl_database.csv`, parse `scp_codes` + `scp_statements.csv` for superclass labels, apply fold-based splitting (fold 10 = test, folds 1–9 NORM → 90/10 train/val)
- [x] 3.4 Implement `fit_norm_stats(train_records) -> dict` and `normalize(record, stats) -> np.ndarray`: compute per-lead mean/std over NORM train split; save stats to `artifacts/norm_stats.json`
- [x] 3.5 Implement `PTBXLDataset(torch.utils.data.Dataset)` wrapping the above: loads, filters, normalizes, and returns `(signal_tensor, label_str)` pairs
- [x] 3.6 Smoke test: load 5 NORM records end-to-end, verify shape `(12, 1000)`, check no NaNs

## 4. VAE Model

- [x] 4.1 Implement `backend/models/vae.py`: `Encoder` class — 3× Conv1d + ReLU, Flatten, `fc_mu`, `fc_logvar` Linear projections
- [x] 4.2 Implement `Decoder` class in `vae.py`: Linear + ReLU, reshape, 3× ConvTranspose1d, Tanh, `F.interpolate(size=1000)`
- [x] 4.3 Implement `VAE` class in `vae.py`: `__init__` reads from config dict, `encode`, `reparameterize` (training vs eval), `decode`, `forward`, `anomaly_score`
- [x] 4.4 Unit test: instantiate VAE from config, pass `(4, 12, 1000)` tensor, assert output shapes `(4, 12, 1000)`, `(4, 128)`, `(4, 128)`
- [x] 4.5 Unit test: `anomaly_score` returns tensor of shape `(4,)` with non-negative values

## 5. Training Loop

- [x] 5.1 Implement `src/train/train_vae.py`: load `config.toml` with `tomllib`, build dataloaders (train NORM, val NORM), instantiate VAE, instantiate Adam optimizer
- [x] 5.2 Implement MSE + beta-KL loss function in `train_vae.py`
- [x] 5.3 Implement epoch loop: forward pass, loss backward, optimizer step, accumulate train/val loss per epoch, print `Epoch {n}/{total} | train_loss={:.4f} | val_loss={:.4f}`
- [x] 5.4 After training loop: save `model.state_dict()` to `artifacts/vae_weights.pt`
- [x] 5.5 After training loop: save `{"train_loss": [...], "val_loss": [...]}` to `artifacts/vae_loss.json`
- [x] 5.6 After training loop: compute NORM val reconstruction errors, compute threshold at configured percentile, update and save `norm_stats.json` with `threshold` key added
- [x] 5.7 Run training for 2 epochs locally to confirm no errors; check artifacts are written

## 6. Evaluation

- [x] 6.1 Implement `src/eval/eval_vae.py`: load config, load VAE weights, load norm stats, build test DataLoader (all classes)
- [x] 6.2 Run inference on full test set: collect `(anomaly_score, true_label)` pairs; binary labels NORM=0, others=1
- [x] 6.3 Compute AUROC using `sklearn.metrics.roc_auc_score`; print to stdout; save to `artifacts/metrics.json` under `VAE.auroc`
- [x] 6.4 Compute PR curve using `sklearn.metrics.precision_recall_curve`; compute PR-AUC; save `artifacts/vae_pr_curve.png`; add `VAE.pr_auc` to `metrics.json`
- [x] 6.5 Generate per-class score distribution histogram: one overlapping histogram per class (NORM, MI, STTC, CD, HYP); save to `artifacts/vae_score_distribution.png`
- [x] 6.6 Verify `metrics.json` contains `VAE` key with `auroc`, `pr_auc`, `threshold`, `threshold_percentile`

## 7. Colab Compatibility

- [x] 7.1 Document in `src/data/download_dataset.py` docstring: how to install `kagglehub` in Colab, how to upload `kaggle.json`, and how to update `config.toml` `data.dataset_dir` to the Colab-resolved path
- [x] 7.2 Confirm `config.toml` `training.device` can be set to `"cuda"` for Colab GPU training with no code changes
