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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.ddpm import DDPM
from src.data.download_dataset import get_dataset_path
from backend.preprocess.registry import get_spec
from backend.preprocess.pipeline import load_norm_stats, save_norm_stats
from backend.preprocess.mimic import sort_records_by_npz


def ema_update(ema_model: DDPM, model: DDPM, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found at {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    spec = get_spec(config)
    dataset_dir = get_dataset_path(config)
    fs = config["data"]["sampling_rate"]
    if config["data"].get("name") == "mimic-iv-ecg":
        _train_cfg = config.get("ddpm_training", config["training"])
    else:
        _train_cfg = config["training"]
    epochs = _train_cfg["epochs"]
    batch_size = _train_cfg["batch_size"]
    lr = _train_cfg.get("learning_rate", config["training"]["learning_rate"])
    _device_cfg = config["training"]["device"]
    if _device_cfg == "cpu" and torch.cuda.is_available():
        _device_cfg = "cuda"
        print("CUDA available — overriding config device to 'cuda'")
    device = torch.device(_device_cfg)

    ema_decay = config["ddpm"].get("ema_decay", 0.9999)
    percentile = config["anomaly"]["threshold_percentile"]
    output_dir = Path(config["artifacts"]["output_dir"])
    if config["data"].get("name") == "mimic-iv-ecg":
        output_dir = output_dir / "mimic"
    output_dir.mkdir(parents=True, exist_ok=True)
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
        train_records, val_records, test_records = spec.build_splits_fn(config, dataset_dir=dataset_dir)
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
        norm_stats = spec.fit_norm_stats_fn(train_records, dataset_dir)
        save_norm_stats(norm_stats, str(norm_stats_path))
        print(f"  Saved to {norm_stats_path}")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    # Sort by npz path so prefetch reads each file sequentially (801 opens vs 74K)
    train_records = sort_records_by_npz(train_records)
    train_ds = spec.dataset_class(train_records, dataset_dir, norm_stats=norm_stats)
    val_ds = spec.dataset_class(val_records, dataset_dir, norm_stats=norm_stats)
    if hasattr(train_ds, "prefetch"):
        train_ds.prefetch()
    if hasattr(val_ds, "prefetch"):
        val_ds.prefetch()

    pin_memory = torch.cuda.is_available()
    # shuffle=False because records are pre-sorted by npz file for sequential I/O
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    # ------------------------------------------------------------------
    # Model, EMA copy, optimizer
    # ------------------------------------------------------------------
    model = DDPM(config).to(device)

    # DataParallel — use all available GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # AMP disabled — float32 for numerical stability
    use_amp = False
    scaler = GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision (AMP): disabled (float32)")

    _raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    optimizer = torch.optim.Adam(_raw_model.unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    total_params = sum(p.numel() for p in _raw_model.unet.parameters())
    print(f"UNet parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    checkpoint_file = output_dir / "ddpm_checkpoint.pt"
    best_weights_file = output_dir / "ddpm_ema_weights_best.pt"
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    start_epoch = 1

    if checkpoint_file.exists():
        print(f"Resuming from checkpoint {checkpoint_file}...")
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        _raw_model.load_state_dict(ckpt["model"])
        _raw_ema_tmp = ema_model.module if isinstance(ema_model, torch.nn.DataParallel) else ema_model
        _raw_ema_tmp.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if use_amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        epoch_train_loss = 0.0
        n_batches = len(train_loader)
        log_every = max(1, n_batches // 10)
        for batch_idx, (signals, _) in enumerate(train_loader, 1):
            signals = signals.to(device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                loss = model.loss(signals) if not isinstance(model, torch.nn.DataParallel) \
                    else model.module.loss(signals)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_raw_model.unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema_update(ema_model, model, ema_decay)
            epoch_train_loss += loss.item() * signals.size(0)
            if batch_idx % log_every == 0 or batch_idx == n_batches:
                print(
                    f"  Epoch {epoch}/{epochs} batch {batch_idx}/{n_batches} loss={loss.item():.4f}",
                    flush=True,
                )
        epoch_train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for signals, _ in val_loader:
                signals = signals.to(device)
                with autocast("cuda", enabled=use_amp):
                    loss = model.module.loss(signals) if isinstance(model, torch.nn.DataParallel) \
                        else model.loss(signals)
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
            f"Epoch {epoch}/{epochs} | lr={current_lr:.2e} | "
            f"train={epoch_train_loss:.4f} | val={epoch_val_loss:.4f} | "
            f"t={epoch_time:.1f}s | eta={eta/60:.1f}m",
            flush=True,
        )

        _raw_ema = ema_model.module if isinstance(ema_model, torch.nn.DataParallel) else ema_model
        _raw_model_ckpt = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(_raw_ema.state_dict(), ema_weights_file)
        with open(loss_file, "w") as f:
            json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)

        # Only save resume checkpoint when val loss improves — prevents resuming from corrupted state
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(_raw_ema.state_dict(), best_weights_file)
            torch.save({
                "epoch": epoch,
                "model": _raw_model_ckpt.state_dict(),
                "ema": _raw_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
            }, checkpoint_file)
            print(f"  Best weights saved (epoch {epoch}, val={best_val_loss:.4f})", flush=True)
        print(f"  Checkpoint saved (epoch {epoch})", flush=True)

    # ------------------------------------------------------------------
    # Save EMA weights (final)
    # ------------------------------------------------------------------
    _raw_ema = ema_model.module if isinstance(ema_model, torch.nn.DataParallel) else ema_model
    torch.save(_raw_ema.state_dict(), ema_weights_file)
    print(f"\nEMA weights saved to {ema_weights_file}")

    # --- Save loss history ---
    with open(loss_file, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)
    print(f"Loss history saved to {loss_file}")

    # ------------------------------------------------------------------
    # Anomaly threshold on val set (using EMA model)
    # ------------------------------------------------------------------
    print(f"Computing anomaly threshold at {percentile}th percentile of NORM val errors...")
    _raw_ema.eval()
    val_errors = []
    # Use larger batch for threshold computation — data already in RAM, bigger batch = fewer UNet calls
    threshold_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=pin_memory)
    n_threshold_batches = len(threshold_loader)
    with torch.no_grad():
        for i, (signals, _) in enumerate(threshold_loader, 1):
            signals = signals.to(device)
            scores = _raw_ema.anomaly_score(signals, ema_unet=_raw_ema.unet)
            val_errors.extend(scores.cpu().numpy().tolist())
            if i % 5 == 0 or i == n_threshold_batches:
                print(f"  Threshold progress: {i}/{n_threshold_batches} batches ({len(val_errors)}/{len(val_ds)} samples)", flush=True)

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
