## ADDED Requirements

### Requirement: Training entry point is src/train/train_vae.py
The system SHALL provide `src/train/train_vae.py` as the sole training entry point, runnable via `uv run python src/train/train_vae.py`. It SHALL load `config.toml`, instantiate the preprocessing pipeline and VAE, run the training loop, and save all artifacts on completion.

#### Scenario: Training script runs end-to-end without error
- **WHEN** `uv run python src/train/train_vae.py` is executed with a valid `config.toml` and accessible dataset
- **THEN** the script exits with code 0 and artifacts are present in `artifacts/`

---

### Requirement: Loss function combines MSE reconstruction and beta-weighted KL divergence
The VAE loss SHALL be computed as:
```
recon_loss = F.mse_loss(x_recon, x, reduction='mean')
kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
total_loss = recon_loss + beta * kl_loss
```
where `beta` is read from `config.toml` `[model] beta`.

#### Scenario: Loss decreases over training epochs on NORM data
- **WHEN** the VAE is trained for at least 10 epochs on NORM records
- **THEN** the mean training loss at epoch 10 is less than the mean training loss at epoch 1

#### Scenario: KL loss is non-negative
- **WHEN** `kl_loss` is computed for any batch
- **THEN** `kl_loss >= 0`

---

### Requirement: Training loop logs loss per epoch to stdout
The training loop SHALL print train loss and validation loss to stdout after each epoch in the format:
```
Epoch {n}/{total} | train_loss={:.4f} | val_loss={:.4f}
```

#### Scenario: Loss is printed for every epoch
- **WHEN** training runs for `epochs = 5`
- **THEN** exactly 5 lines matching the epoch log format are printed to stdout

---

### Requirement: Loss history is saved to artifacts/vae_loss.json
After training completes, the system SHALL save loss history as JSON to `{artifacts.output_dir}/{artifacts.loss_file}` in the format:
```json
{
  "train_loss": [<float>, ...],
  "val_loss":   [<float>, ...]
}
```
with one value per epoch.

#### Scenario: Loss JSON contains correct number of epochs
- **WHEN** training runs for `epochs = 50`
- **THEN** `vae_loss.json` contains lists of length 50 for both `train_loss` and `val_loss`

---

### Requirement: Model weights are saved to artifacts/vae_weights.pt
After training completes, the system SHALL save the full model state dict using `torch.save(model.state_dict(), ...)` to `{artifacts.output_dir}/{artifacts.weights_file}`.

#### Scenario: Saved weights can be reloaded into a fresh VAE instance
- **WHEN** a new VAE instance is created with the same config and `load_state_dict` is called with the saved weights
- **THEN** the model produces identical outputs to the trained model for the same input

---

### Requirement: NORM val reconstruction threshold is saved in norm_stats.json
After training, the system SHALL compute reconstruction MSE for all NORM validation records, compute the threshold at `config.toml` `anomaly.threshold_percentile`, and save to `{artifacts.output_dir}/{artifacts.norm_stats_file}` as:
```json
{
  "mean": [<float x 12>],
  "std":  [<float x 12>],
  "threshold": <float>
}
```

#### Scenario: norm_stats.json contains mean, std, and threshold keys
- **WHEN** training completes successfully
- **THEN** `norm_stats.json` contains exactly the keys `mean`, `std`, and `threshold`
