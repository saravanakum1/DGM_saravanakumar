"""
VAE training entry point.

Usage:
    uv run python src/train/train_vae.py

Reads all settings from config.toml at the repo root.
Saves artifacts to the directory configured under [artifacts] output_dir.

Kaggle Notebook usage:
    - Set config.toml [data] dataset_dir to the Kaggle input path, e.g.:
        dataset_dir = "/kaggle/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset"
    - Set [training] device = "cuda"
    - Run: !python src/train/train_vae.py
    - Download output files from /kaggle/working/artifacts/
"""

import json
import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.vae import VAE
from backend.preprocess.pipeline import (
    PTBXLDataset,
    build_splits,
    fit_norm_stats,
    load_norm_stats,
    save_norm_stats,
)


def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def main():
    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found at {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    dataset_dir = config["data"]["dataset_dir"]
    fs = config["data"]["sampling_rate"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    device = torch.device(config["training"]["device"])
    beta = config["model"]["beta"]
    percentile = config["anomaly"]["threshold_percentile"]
    output_dir = Path(config["artifacts"]["output_dir"])
    weights_file = output_dir / config["artifacts"]["weights_file"]
    loss_file = output_dir / config["artifacts"]["loss_file"]
    norm_stats_path = output_dir / config["artifacts"]["norm_stats_file"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset splits...")
    train_records, val_records, _ = build_splits(config)
    print(f"  Train: {len(train_records)} NORM | Val: {len(val_records)} NORM")

    # Fit or load norm stats
    if norm_stats_path.exists():
        print(f"Loading existing norm stats from {norm_stats_path}")
        norm_stats = load_norm_stats(str(norm_stats_path))
    else:
        print("Computing normalization stats from training split...")
        norm_stats = fit_norm_stats(train_records, dataset_dir, fs=fs)
        save_norm_stats(norm_stats, str(norm_stats_path))
        print(f"  Saved to {norm_stats_path}")

    train_ds = PTBXLDataset(train_records, dataset_dir, norm_stats=norm_stats, fs=fs)
    val_ds = PTBXLDataset(val_records, dataset_dir, norm_stats=norm_stats, fs=fs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        epoch_train_loss = 0.0
        for signals, _ in train_loader:
            signals = signals.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(signals)
            loss, _, _ = vae_loss(x_recon, signals, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * signals.size(0)
        epoch_train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for signals, _ in val_loader:
                signals = signals.to(device)
                x_recon, mu, logvar = model(signals)
                loss, _, _ = vae_loss(x_recon, signals, mu, logvar, beta)
                epoch_val_loss += loss.item() * signals.size(0)
        epoch_val_loss /= len(val_ds)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(f"Epoch {epoch}/{epochs} | train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f}")

    # --- Save weights ---
    torch.save(model.state_dict(), weights_file)
    print(f"\nWeights saved to {weights_file}")

    # --- Save loss history ---
    with open(loss_file, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)
    print(f"Loss history saved to {loss_file}")

    # --- Compute anomaly threshold on val set ---
    print(f"Computing anomaly threshold at {percentile}th percentile of NORM val errors...")
    model.eval()
    val_errors = []
    with torch.no_grad():
        for signals, _ in val_loader:
            signals = signals.to(device)
            scores = model.anomaly_score(signals)
            val_errors.extend(scores.cpu().numpy().tolist())

    threshold = float(np.percentile(val_errors, percentile))
    norm_stats["threshold"] = threshold
    norm_stats["threshold_percentile"] = percentile
    save_norm_stats(norm_stats, str(norm_stats_path))
    print(f"Threshold ({percentile}th pct): {threshold:.6f} — saved to {norm_stats_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
