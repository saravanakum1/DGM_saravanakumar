"""
DDPM training entry point.

Usage:
    uv run python src/train/train_ddpm.py

Reads all settings from config.toml at the repo root.
Saves artifacts to the directory configured under [artifacts] output_dir.

Kaggle Notebook usage:
    - Set config.toml [data] dataset_dir to the Kaggle input path, e.g.:
        dataset_dir = "/kaggle/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset"
    - Set [training] device = "cuda"
    - Run: !python src/train/train_ddpm.py
    - Download output files from /kaggle/working/artifacts/
"""

import copy
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.ddpm import DDPM
from src.data.download_dataset import get_dataset_path
from backend.preprocess.pipeline import (
    PTBXLDataset,
    build_splits,
    fit_norm_stats,
    load_norm_stats,
    save_norm_stats,
)


def ema_update(ema_model: DDPM, model: DDPM, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


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

    ema_decay = config["ddpm"].get("ema_decay", 0.9999)
    percentile = config["anomaly"]["threshold_percentile"]
    output_dir = Path(config["artifacts"]["output_dir"])
    ema_weights_file = output_dir / "ddpm_ema_weights.pt"
    loss_file = output_dir / "ddpm_loss.json"
    norm_stats_path = output_dir / config["artifacts"]["norm_stats_file"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset splits
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Norm stats
    # ------------------------------------------------------------------
    bundled_norm_stats = KAGGLE_CODE_PATH / "norm_stats.json"
    if bundled_norm_stats.exists():
        print(f"Loading bundled norm stats from {bundled_norm_stats}")
        norm_stats = load_norm_stats(str(bundled_norm_stats))
        save_norm_stats(norm_stats, str(norm_stats_path))
    elif norm_stats_path.exists():
        print(f"Loading existing norm stats from {norm_stats_path}")
        norm_stats = load_norm_stats(str(norm_stats_path))
    else:
        print("Computing normalization stats from training split...")
        norm_stats = fit_norm_stats(train_records, dataset_dir, fs=fs)
        save_norm_stats(norm_stats, str(norm_stats_path))
        print(f"  Saved to {norm_stats_path}")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    print("Preloading train signals into RAM...")
    train_ds = PTBXLDataset(train_records, dataset_dir, norm_stats=norm_stats, fs=fs)
    print(f"  Train: {len(train_ds)} signals loaded ({train_ds.signals.nbytes / 1e6:.0f} MB)")
    print("Preloading val signals into RAM...")
    val_ds = PTBXLDataset(val_records, dataset_dir, norm_stats=norm_stats, fs=fs)
    print(f"  Val: {len(val_ds)} signals loaded ({val_ds.signals.nbytes / 1e6:.0f} MB)")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    # ------------------------------------------------------------------
    # Model, EMA copy, optimizer
    # ------------------------------------------------------------------
    model = DDPM(config).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # Only the UNet parameters are trainable — schedule buffers are fixed
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    total_params = sum(p.numel() for p in model.unet.parameters())
    print(f"\nUNet parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_losses, val_losses = [], []
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        epoch_train_loss = 0.0
        for signals, _ in train_loader:
            signals = signals.to(device)
            optimizer.zero_grad()
            loss = model.loss(signals)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
            optimizer.step()
            ema_update(ema_model, model, ema_decay)
            epoch_train_loss += loss.item() * signals.size(0)
        epoch_train_loss /= len(train_ds)

        # --- Validate (loss only — no reverse pass, keeps val fast) ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for signals, _ in val_loader:
                signals = signals.to(device)
                epoch_val_loss += model.loss(signals).item() * signals.size(0)
        epoch_val_loss /= len(val_ds)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        eta = (elapsed / epoch) * (epochs - epoch)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        print(
            f"Epoch {epoch}/{epochs} | lr={current_lr:.2e} | "
            f"train={epoch_train_loss:.4f} | val={epoch_val_loss:.4f} | "
            f"t={epoch_time:.1f}s | eta={eta/60:.1f}m",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Save EMA weights
    # ------------------------------------------------------------------
    torch.save(ema_model.state_dict(), ema_weights_file)
    print(f"\nEMA weights saved to {ema_weights_file}")

    # --- Save loss history ---
    with open(loss_file, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)
    print(f"Loss history saved to {loss_file}")

    # ------------------------------------------------------------------
    # Anomaly threshold on val set (using EMA model)
    # ------------------------------------------------------------------
    print(f"Computing anomaly threshold at {percentile}th percentile of NORM val errors...")
    ema_model.eval()
    val_errors = []
    with torch.no_grad():
        for signals, _ in val_loader:
            signals = signals.to(device)
            scores = ema_model.anomaly_score(signals, ema_unet=ema_model.unet)
            val_errors.extend(scores.cpu().numpy().tolist())

    threshold = float(np.percentile(val_errors, percentile))
    norm_stats["ddpm_threshold"] = threshold
    norm_stats["ddpm_threshold_percentile"] = percentile
    save_norm_stats(norm_stats, str(norm_stats_path))
    print(f"Threshold ({percentile}th pct): {threshold:.6f} — saved to {norm_stats_path}")

    # ------------------------------------------------------------------
    # Loss curve
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, epochs + 1), train_losses, label="train")
        ax.plot(range(1, epochs + 1), val_losses,   label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Noise loss (L_simple)")
        ax.set_title("DDPM Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = output_dir / "ddpm_loss_curve.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping loss curve plot")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
