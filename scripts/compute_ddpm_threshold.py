"""Compute DDPM anomaly threshold from saved EMA weights.

Run this when training completed but the session was killed before
threshold computation finished. Loads best weights, scores the NORM
val set, saves ddpm_threshold to norm_stats.json.

Usage (Kaggle):
    !python scripts/compute_ddpm_threshold.py
"""

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.models.ddpm import DDPM
from backend.preprocess.pipeline import load_norm_stats, save_norm_stats
from backend.preprocess.mimic import sort_records_by_npz
from backend.preprocess.registry import get_spec
from src.data.download_dataset import get_dataset_path

KAGGLE_CODE_PATH = Path("/kaggle/input/datasets/sharav95/ecg-ad-comparison/artifacts")


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    spec = get_spec(config)
    dataset_dir = get_dataset_path(config)
    percentile = config["anomaly"]["threshold_percentile"]

    _device_cfg = config["training"]["device"]
    if _device_cfg == "cuda" and not torch.cuda.is_available():
        _device_cfg = "mps" if torch.backends.mps.is_available() else "cpu"
    elif _device_cfg == "cpu" and torch.cuda.is_available():
        _device_cfg = "cuda"
    device = torch.device(_device_cfg)
    print(f"Device: {device}")

    output_dir = Path(config["artifacts"]["output_dir"])
    if config["data"].get("name") == "mimic-iv-ecg":
        output_dir = output_dir / "mimic"
    norm_stats_path = output_dir / config["artifacts"]["norm_stats_file"]
    best_weights = output_dir / "ddpm_ema_weights_best.pt"
    if not best_weights.exists():
        best_weights = output_dir / "ddpm_ema_weights.pt"

    if not best_weights.exists():
        raise FileNotFoundError(f"No weights found at {best_weights}")

    print(f"Loading weights from {best_weights}")
    model = DDPM(config).to(device)
    model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))
    model.eval()

    # Load val split
    for cache_path in (KAGGLE_CODE_PATH / "splits_cache.json", output_dir / "splits_cache.json"):
        if cache_path.exists():
            print(f"Loading splits from {cache_path}")
            with open(cache_path) as f:
                splits = json.load(f)
            val_records = [tuple(r) for r in splits["val"]]
            break
    else:
        raise FileNotFoundError("splits_cache.json not found")

    val_records = sort_records_by_npz(val_records)
    print(f"Val set: {len(val_records)} NORM records")

    norm_stats = load_norm_stats(str(norm_stats_path))
    val_ds = spec.dataset_class(val_records, dataset_dir, norm_stats=norm_stats)
    if hasattr(val_ds, "prefetch"):
        val_ds.prefetch()

    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    n_batches = len(val_loader)

    print(f"Scoring {len(val_ds)} val samples ({n_batches} batches of 256)...")
    scores = []
    with torch.no_grad():
        for i, (signals, _) in enumerate(val_loader, 1):
            signals = signals.to(device)
            batch_scores = model.anomaly_score(signals, ema_unet=model.unet)
            scores.extend(batch_scores.cpu().numpy().tolist())
            if i % 5 == 0 or i == n_batches:
                print(f"  {i}/{n_batches} batches ({len(scores)}/{len(val_ds)} samples)", flush=True)

    threshold = float(np.percentile(scores, percentile))
    print(f"Threshold ({percentile}th pct): {threshold:.6f}")

    norm_stats["ddpm_threshold"] = threshold
    norm_stats["ddpm_threshold_percentile"] = percentile
    save_norm_stats(norm_stats, str(norm_stats_path))
    print(f"Saved to {norm_stats_path}")


if __name__ == "__main__":
    main()
