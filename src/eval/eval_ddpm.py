"""
DDPM evaluation entry point.

Usage:
    uv run python src/eval/eval_ddpm.py

Reads settings from config.toml. Requires trained artifacts:
    artifacts/ddpm_ema_weights.pt
    artifacts/norm_stats.json   (must contain ddpm_threshold)

Outputs:
    artifacts/metrics.json                  — updated with DDPM metrics (merged with VAE if present)
    artifacts/ddpm_pr_curve.png             — Precision-Recall curve
    artifacts/ddpm_score_distribution.png   — Per-class anomaly score histogram
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tomllib
import torch
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.ddpm import DDPM
from src.data.download_dataset import get_dataset_path
from backend.preprocess.pipeline import PTBXLDataset, build_splits, load_norm_stats

CLASS_COLORS = {
    "NORM": "#22C55E",
    "MI":   "#EF4444",
    "STTC": "#F97316",
    "CD":   "#EAB308",
    "HYP":  "#A855F7",
}


def main():
    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found at {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    dataset_dir = get_dataset_path()
    fs = config["data"]["sampling_rate"]
    batch_size = config["training"]["batch_size"]
    _device_cfg = config["training"]["device"]
    if _device_cfg == "cpu" and torch.cuda.is_available():
        _device_cfg = "cuda"
    device = torch.device(_device_cfg)
    output_dir = Path(config["artifacts"]["output_dir"])
    weights_path = output_dir / "ddpm_ema_weights.pt"
    norm_stats_path = output_dir / config["artifacts"]["norm_stats_file"]
    metrics_path = output_dir / "metrics.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"EMA weights not found at {weights_path}. Run train_ddpm.py first."
        )
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"Norm stats not found at {norm_stats_path}. Run train_ddpm.py first."
        )

    norm_stats = load_norm_stats(str(norm_stats_path))
    threshold = norm_stats.get("ddpm_threshold")
    if threshold is None:
        raise KeyError(
            "'ddpm_threshold' missing from norm_stats.json. Run train_ddpm.py to compute it."
        )
    threshold_percentile = norm_stats.get(
        "ddpm_threshold_percentile", config["anomaly"]["threshold_percentile"]
    )

    # Load model — DDPM constructor creates the schedule buffers, then we overwrite weights
    model = DDPM(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded EMA weights from {weights_path}")

    # ------------------------------------------------------------------
    # Test split — same cache as training
    # ------------------------------------------------------------------
    KAGGLE_CODE_PATH = Path("/kaggle/input/datasets/sharav95/ecg-ad-comparison/artifacts")
    splits_cache = output_dir / "splits_cache.json"
    bundled_splits = KAGGLE_CODE_PATH / "splits_cache.json"

    if bundled_splits.exists():
        print(f"Loading cached splits from {bundled_splits}")
        with open(bundled_splits) as f:
            _splits = json.load(f)
        test_records = [tuple(r) for r in _splits["test"]]
    elif splits_cache.exists():
        print(f"Loading cached splits from {splits_cache}")
        with open(splits_cache) as f:
            _splits = json.load(f)
        test_records = [tuple(r) for r in _splits["test"]]
    else:
        print("Building dataset splits (slow — run train_ddpm.py first to cache)...")
        _, _, test_records = build_splits(config, dataset_dir=dataset_dir)

    print(f"Test set: {len(test_records)} records")

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    print("Preloading test signals into RAM...")
    t0 = time.time()
    test_ds = PTBXLDataset(test_records, dataset_dir, norm_stats=norm_stats, fs=fs)
    print(f"  {len(test_ds)} signals loaded ({test_ds.signals.nbytes / 1e6:.0f} MB) in {time.time()-t0:.1f}s")

    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )

    # ------------------------------------------------------------------
    # Inference — anomaly_score() already uses torch.no_grad internally
    # ------------------------------------------------------------------
    print("Running DDPM anomaly scoring (partial diffusion t*→0 per batch)...")
    t0 = time.time()
    all_scores, all_labels = [], []
    for i, (signals, labels) in enumerate(test_loader):
        signals = signals.to(device)
        scores = model.anomaly_score(signals)
        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(test_loader) - i - 1)
            print(f"  batch {i+1}/{len(test_loader)} | elapsed {elapsed:.0f}s | eta {remaining:.0f}s", flush=True)

    print(f"Scoring complete in {time.time()-t0:.1f}s")

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Binary labels: NORM=0, everything else=1
    binary_labels = (all_labels != "NORM").astype(int)

    # --- AUROC ---
    auroc = float(roc_auc_score(binary_labels, all_scores))
    print(f"AUROC: {auroc:.4f}")

    # --- PR curve + PR-AUC ---
    precision, recall, _ = precision_recall_curve(binary_labels, all_scores)
    pr_auc = float(auc(recall, precision))
    print(f"PR-AUC: {pr_auc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#3B82F6", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"DDPM Precision-Recall Curve (PR-AUC = {pr_auc:.3f})")
    ax.grid(alpha=0.3)
    pr_path = output_dir / "ddpm_pr_curve.png"
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PR curve saved to {pr_path}")

    # --- Per-class score distribution ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls, color in CLASS_COLORS.items():
        mask = all_labels == cls
        if mask.sum() == 0:
            continue
        ax.hist(
            all_scores[mask],
            bins=60,
            alpha=0.55,
            label=f"{cls} (n={mask.sum()})",
            color=color,
            density=True,
        )
    ax.axvline(
        threshold, color="black", linestyle="--", lw=1.5,
        label=f"Threshold ({threshold_percentile}th pct)"
    )
    ax.set_xlabel("Anomaly Score (Partial Diffusion MSE)")
    ax.set_ylabel("Density")
    ax.set_title("DDPM Anomaly Score Distribution by Class")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    dist_path = output_dir / "ddpm_score_distribution.png"
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Score distribution saved to {dist_path}")

    # --- Save / merge metrics.json ---
    existing_metrics: dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing_metrics = json.load(f)

    existing_metrics["DDPM"] = {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
    }
    with open(metrics_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
