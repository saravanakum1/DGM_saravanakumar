"""Train Flow Matching model on PTB-XL NORM ECGs.

Usage (local):
    uv run python src/train/train_flow_matching.py

Usage (Kaggle):
    python src/train/train_flow_matching.py --dataset-dir /kaggle/input/ptb-xl.../
"""
import argparse
import copy
import json
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.models.flow_matching import FlowMatching, sample_euler, cfm_score
from backend.preprocess.registry import get_spec
from backend.preprocess.pipeline import load_norm_stats, save_norm_stats
from backend.preprocess.mimic import sort_records_by_npz
from src.data.download_dataset import get_dataset_path


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

FM_DEFAULTS = {
    "epochs": 100,
    "batch_size": 64,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.9999,
    "time_dim": 128,
    "num_steps": 50,
    "n_hutchinson": 1,
    "threshold_percentile": 95,
    "val_vis_samples": 4,
}


# ---------------------------------------------------------------------------
# EMA helpers
# ---------------------------------------------------------------------------

def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

KAGGLE_CODE_PATH = Path("/kaggle/input/datasets/sharav95/ecg-ad-comparison/artifacts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found at {config_path}")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    fm_cfg = {**FM_DEFAULTS, **config.get("flow_matching", {})}
    if args.epochs is not None:
        fm_cfg["epochs"] = args.epochs

    spec = get_spec(config)
    dataset_dir = args.dataset_dir or get_dataset_path(config)
    artifacts_dir = Path(config["artifacts"]["output_dir"])
    if config["data"].get("name") == "mimic-iv-ecg":
        artifacts_dir = artifacts_dir / "mimic"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch_xla
        device = torch_xla.device()
        _use_xla = True
        print("Device: TPU (torch_xla)")
    except ImportError:
        _use_xla = False
        device_str = config["training"].get("device", "cpu")
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        device = torch.device(device_str)
        print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Step 0 — Splits (load from cache or build once and save)
    # ------------------------------------------------------------------
    splits_cache = artifacts_dir / "splits_cache.json"
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
    print(f"  NORM train: {len(train_records)}  val: {len(val_records)}")

    # ------------------------------------------------------------------
    # Norm stats (load bundled → local cache → compute)
    # ------------------------------------------------------------------
    norm_stats_path = artifacts_dir / config["artifacts"]["norm_stats_file"]
    bundled_norm_stats = KAGGLE_CODE_PATH / "norm_stats.json"

    if bundled_norm_stats.exists():
        print(f"Loading bundled norm stats from {bundled_norm_stats}")
        norm_stats = load_norm_stats(str(bundled_norm_stats))
        save_norm_stats(norm_stats, str(norm_stats_path))
    elif norm_stats_path.exists():
        norm_stats = load_norm_stats(str(norm_stats_path))
        print(f"Loaded existing norm stats from {norm_stats_path}")
    else:
        print("Computing norm stats from NORM train set...")
        norm_stats = spec.fit_norm_stats_fn(train_records, dataset_dir)
        save_norm_stats(norm_stats, str(norm_stats_path))
        print(f"Saved norm stats to {norm_stats_path}")

    # ------------------------------------------------------------------
    # Step 1 — Datasets
    # ------------------------------------------------------------------
    fs = config["data"]["sampling_rate"]
    # Sort by npz path so prefetch reads each file sequentially (801 opens vs 74K)
    train_records = sort_records_by_npz(train_records)
    # TPU: disable multiprocessing in dataset loader (XLA forks conflict with TPU runtime)
    _ds_workers = 1 if _use_xla else None
    train_ds = spec.dataset_class(train_records, dataset_dir, norm_stats=norm_stats, num_workers=_ds_workers)
    val_ds = spec.dataset_class(val_records, dataset_dir, norm_stats=norm_stats, num_workers=_ds_workers)
    if hasattr(train_ds, "prefetch"):
        train_ds.prefetch()
    if hasattr(val_ds, "prefetch"):
        val_ds.prefetch()

    train_loader = DataLoader(
        train_ds,
        batch_size=fm_cfg["batch_size"],
        shuffle=False,  # pre-sorted by npz file for sequential I/O
        num_workers=0,
        pin_memory=(not _use_xla and device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=fm_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(not _use_xla and device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Step 2 — Model + EMA
    # ------------------------------------------------------------------
    seq_len = config.get("data", {}).get("sampling_rate", 500) * 10
    model = FlowMatching(
        time_dim=fm_cfg["time_dim"],
        num_steps=fm_cfg["num_steps"],
        n_hutchinson=fm_cfg["n_hutchinson"],
        seq_len=seq_len,
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=fm_cfg["lr"],
        weight_decay=fm_cfg["weight_decay"],
    )

    # ------------------------------------------------------------------
    # Step 3 — Resume from checkpoint if available
    # ------------------------------------------------------------------
    checkpoint_file = artifacts_dir / "fm_checkpoint.pt"
    history: dict[str, list] = {"train": [], "val": []}
    best_val_loss = float("inf")
    start_epoch = 1

    if checkpoint_file.exists():
        print(f"Resuming from checkpoint {checkpoint_file}...")
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt.get("history", {"train": [], "val": []})
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, fm_cfg["epochs"] + 1):
        model.train()
        train_losses: list[float] = []
        t0 = time.time()

        for batch, _ in train_loader:
            batch = batch.to(device)
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fm_cfg["grad_clip"])
            optimizer.step()
            if _use_xla:
                torch_xla.sync()
            ema_update(ema_model, model, fm_cfg["ema_decay"])
            train_losses.append(loss.item())

        # Validation loss
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch, _ in val_loader:
                batch = batch.to(device)
                loss_val = model.loss(batch)
                if _use_xla:
                    torch_xla.sync()
                val_losses.append(loss_val.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        elapsed = time.time() - t0
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(
            f"Epoch {epoch:3d}/{fm_cfg['epochs']}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Save full checkpoint every epoch — safe to kill and resume at any point
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_val_loss": best_val_loss,
        }, checkpoint_file)
        torch.save(ema_model.state_dict(), artifacts_dir / "fm_ema_weights_latest.pt")
        with open(artifacts_dir / "fm_loss.json", "w") as f:
            json.dump(history, f, indent=2)

        # Save best EMA weights separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ema_model.state_dict(), artifacts_dir / "fm_ema_weights.pt")
            print(f"  → Best weights saved (epoch {epoch}, val={val_loss:.4f})", flush=True)
        print(f"  Checkpoint saved (epoch {epoch})", flush=True)

        # Visualize samples every 10 epochs (Euler, fast)
        if epoch % 10 == 0:
            with torch.no_grad():
                z = torch.randn(fm_cfg["val_vis_samples"], 12, spec.seq_len, device=device)
                samples = sample_euler(ema_model.unet, z, num_steps=100)
            print(
                f"  [vis] generated {fm_cfg['val_vis_samples']} samples "
                f"| mean={samples.mean():.3f} std={samples.std():.3f}"
            )

    print("Training complete. Run scripts/compute_fm_threshold.py to compute the anomaly threshold.")


if __name__ == "__main__":
    main()
