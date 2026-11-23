"""Compute Flow Matching anomaly threshold from saved EMA weights.

Run this after train_flow_matching.py completes. Loads best weights,
scores the NORM val set, saves fm_threshold to norm_stats.json.

Usage (Kaggle):
    !python scripts/compute_fm_threshold.py
"""

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.models.flow_matching import FlowMatching, cfm_score
from backend.preprocess.pipeline import load_norm_stats, save_norm_stats
from backend.preprocess.mimic import sort_records_by_npz
from backend.preprocess.registry import get_spec
from src.data.download_dataset import get_dataset_path

KAGGLE_CODE_PATH = Path("/kaggle/input/datasets/sharav95/ecg-ad-comparison/artifacts")


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    fm_cfg = config.get("flow_matching", {})
    spec = get_spec(config)
    dataset_dir = get_dataset_path(config)
    percentile = fm_cfg.get("threshold_percentile", config["anomaly"]["threshold_percentile"])
    n_score_samples = fm_cfg.get("score_samples", 20)

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
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = output_dir / config["artifacts"]["norm_stats_file"]

    # Find best weights
    bundled_dir = KAGGLE_CODE_PATH / ("mimic" if config["data"].get("name") == "mimic-iv-ecg" else "")
    weights_path = None
    for candidate in [
        output_dir / "fm_ema_weights.pt",
        output_dir / "fm_ema_weights_latest.pt",
        bundled_dir / "fm_ema_weights.pt",
    ]:
        if candidate.exists():
            weights_path = candidate
            break

    if weights_path is None:
        raise FileNotFoundError(f"No FM weights found in {output_dir}")

    print(f"Loading weights from {weights_path}")
    seq_len = config.get("data", {}).get("sampling_rate", 500) * 10
    model = FlowMatching(
        time_dim=fm_cfg.get("time_dim", 128),
        num_steps=fm_cfg.get("num_steps", 50),
        n_hutchinson=fm_cfg.get("n_hutchinson", 1),
        seq_len=seq_len,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
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

    print(f"Scoring {len(val_ds)} val samples ({n_batches} batches, {n_score_samples} CFM samples each)...")
    scores = []
    with torch.no_grad():
        for i, (batch, _) in enumerate(val_loader, 1):
            batch = batch.to(device)
            batch_scores = cfm_score(model.unet, batch, n_samples=n_score_samples)
            scores.extend(batch_scores.cpu().tolist())
            if i % 5 == 0 or i == n_batches:
                print(f"  {i}/{n_batches} batches ({len(scores)}/{len(val_ds)} samples)", flush=True)

    threshold = float(np.percentile(scores, percentile))
    print(f"FM threshold ({percentile}th pct): {threshold:.6f}")

    norm_stats["fm_threshold"] = threshold
    norm_stats["fm_threshold_percentile"] = percentile
    save_norm_stats(norm_stats, str(norm_stats_path))
    print(f"Saved to {norm_stats_path}")


if __name__ == "__main__":
    main()
