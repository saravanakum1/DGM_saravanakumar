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
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.vae import VAE
from src.data.download_dataset import get_dataset_path
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


def kl_beta(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """Linear KL warmup: beta ramps from 0 to beta_max over warmup_epochs."""
    return min(beta_max, beta_max * epoch / warmup_epochs)


def main():
    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found at {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    dataset_dir = get_dataset_path()
    fs = config["data"]["sampling_rate"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    _device_cfg = config["training"]["device"]
    if _device_cfg == "cpu" and torch.cuda.is_available():
        _device_cfg = "cuda"
        print("CUDA available — overriding config device to 'cuda'")
    device = torch.device(_device_cfg)
    beta = config["model"]["beta"]
    percentile = config["anomaly"]["threshold_percentile"]
    output_dir = Path(config["artifacts"]["output_dir"])
    weights_file = output_dir / config["artifacts"]["weights_file"]
    loss_file = output_dir / config["artifacts"]["loss_file"]
    norm_stats_path = output_dir / config["artifacts"]["norm_stats_file"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or build dataset splits
    KAGGLE_CODE_PATH = Path("/kaggle/input/datasets/sharav95/ecg-ad-comparison/artifacts")
    splits_cache = output_dir / "splits_cache.json"
    bundled_splits = KAGGLE_CODE_PATH / "splits_cache.json"

    if bundled_splits.exists():
        print(f"Loading cached splits from {bundled_splits}")
        with open(bundled_splits) as f:
            _splits = json.load(f)
        train_records = [tuple(r) for r in _splits["train"]]
        val_records = [tuple(r) for r in _splits["val"]]
    elif splits_cache.exists():
        print(f"Loading cached splits from {splits_cache}")
        with open(splits_cache) as f:
            _splits = json.load(f)
        train_records = [tuple(r) for r in _splits["train"]]
        val_records = [tuple(r) for r in _splits["val"]]
    else:
        print("Building dataset splits...")
        train_records, val_records, test_records = build_splits(config, dataset_dir=dataset_dir)
        with open(splits_cache, "w") as f:
            json.dump({"train": train_records, "val": val_records, "test": test_records}, f)
        print(f"  Splits cached to {splits_cache}")
    print(f"  Train: {len(train_records)} NORM | Val: {len(val_records)} NORM")

    # Load norm stats: bundled in code dataset → output dir → compute from scratch
    bundled_norm_stats = KAGGLE_CODE_PATH / "norm_stats.json"
    if bundled_norm_stats.exists():
        print(f"Loading bundled norm stats from {bundled_norm_stats}")
        norm_stats = load_norm_stats(str(bundled_norm_stats))
        save_norm_stats(norm_stats, str(norm_stats_path))  # copy to output dir
    elif norm_stats_path.exists():
        print(f"Loading existing norm stats from {norm_stats_path}")
        norm_stats = load_norm_stats(str(norm_stats_path))
    else:
        print("Computing normalization stats from training split...")
        norm_stats = fit_norm_stats(train_records, dataset_dir, fs=fs)
        save_norm_stats(norm_stats, str(norm_stats_path))
        print(f"  Saved to {norm_stats_path}")

    print("Preloading train signals into RAM...")
    train_ds = PTBXLDataset(train_records, dataset_dir, norm_stats=norm_stats, fs=fs)
    print(f"  Train: {len(train_ds)} signals loaded ({train_ds.signals.nbytes / 1e6:.0f} MB)")
    print("Preloading val signals into RAM...")
    val_ds = PTBXLDataset(val_records, dataset_dir, norm_stats=norm_stats, fs=fs)
    print(f"  Val: {len(val_ds)} signals loaded ({val_ds.signals.nbytes / 1e6:.0f} MB)")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = VAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    train_losses, val_losses = [], []
    warmup_epochs = max(1, epochs // 5)  # ramp KL over first 20% of training

    train_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        beta_now = kl_beta(epoch, warmup_epochs, beta)

        # --- Train ---
        model.train()
        epoch_train_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        for signals, _ in train_loader:
            signals = signals.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(signals)
            loss, recon, kl = vae_loss(x_recon, signals, mu, logvar, beta_now)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            n = signals.size(0)
            epoch_train_loss += loss.item() * n
            epoch_recon_loss += recon.item() * n
            epoch_kl_loss += kl.item() * n
        epoch_train_loss /= len(train_ds)
        epoch_recon_loss /= len(train_ds)
        epoch_kl_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for signals, _ in val_loader:
                signals = signals.to(device)
                x_recon, mu, logvar = model(signals)
                loss, _, _ = vae_loss(x_recon, signals, mu, logvar, beta_now)
                epoch_val_loss += loss.item() * signals.size(0)
        epoch_val_loss /= len(val_ds)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        eta = (elapsed / epoch) * (epochs - epoch)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(
            f"Epoch {epoch}/{epochs} | lr={current_lr:.2e} | beta={beta_now:.3f} | "
            f"train={epoch_train_loss:.4f} (recon={epoch_recon_loss:.4f} kl={epoch_kl_loss:.4f}) | "
            f"val={epoch_val_loss:.4f} | t={epoch_time:.1f}s | eta={eta/60:.1f}m",
            flush=True,
        )

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
