"""
Flow Matching evaluation entry point.

Usage:
    uv run python src/eval/eval_flow_matching.py

Reads settings from config.toml. Requires trained artifacts:
    artifacts/fm_ema_weights.pt
    artifacts/norm_stats.json   (must contain fm_threshold)

Outputs:
    artifacts/metrics.json                    — updated with FM metrics (merged with VAE/DDPM if present)
    artifacts/fm_pr_curve.png                 — Precision-Recall curve
    artifacts/fm_score_distribution.png       — Per-class anomaly score histogram
    artifacts/fm_per_class_auroc.png          — Per-class one-vs-rest AUROC bar chart
"""

import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

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

from backend.models.flow_matching import FlowMatching, cfm_score
from src.data.download_dataset import get_dataset_path
from backend.preprocess.registry import get_spec
from backend.preprocess.pipeline import load_norm_stats
from backend.preprocess.mimic import sort_records_by_npz

KAGGLE_CODE_PATH = Path("/kaggle/input/datasets/sharav95/ecg-ad-comparison/artifacts")


def _load_splits(output_dir: Path, config: dict, dataset_dir: str, spec) -> list:
    for cache_path in (KAGGLE_CODE_PATH / "splits_cache.json", output_dir / "splits_cache.json"):
        if cache_path.exists():
            print(f"Loading cached splits from {cache_path}")
            with open(cache_path) as f:
                splits = json.load(f)
            return [tuple(r) for r in splits["test"]]
    print("Building dataset splits (slow — run train_flow_matching.py first to cache)...")
    _, _, test_records = spec.build_splits_fn(config, dataset_dir=dataset_dir)
    return test_records


def main() -> None:
    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found at {config_path}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    spec = get_spec(config)
    dataset_dir = get_dataset_path(config)
    fs = config["data"]["sampling_rate"]
    fm_cfg = config.get("flow_matching", {})
    batch_size = fm_cfg.get("batch_size", config["training"]["batch_size"])

    _device_cfg = config["training"]["device"]
    if _device_cfg == "cpu" and torch.cuda.is_available():
        _device_cfg = "cuda"
    elif _device_cfg == "cpu" and torch.backends.mps.is_available():
        _device_cfg = "mps"
    device = torch.device(_device_cfg)

    output_dir = Path(config["artifacts"]["output_dir"])
    if config["data"].get("name") == "mimic-iv-ecg":
        output_dir = output_dir / "mimic"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "fm_ema_weights.pt"
    norm_stats_path = output_dir / config["artifacts"]["norm_stats_file"]
    metrics_path = output_dir / "metrics.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"EMA weights not found at {weights_path}. Run train_flow_matching.py first."
        )
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"Norm stats not found at {norm_stats_path}. Run train_flow_matching.py first."
        )

    norm_stats = load_norm_stats(str(norm_stats_path))
    threshold = norm_stats.get("fm_threshold")
    if threshold is None:
        raise KeyError(
            "'fm_threshold' missing from norm_stats.json. Run train_flow_matching.py to compute it."
        )
    threshold_percentile = fm_cfg.get(
        "threshold_percentile", config["anomaly"]["threshold_percentile"]
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = FlowMatching(
        time_dim=fm_cfg.get("time_dim", 128),
        num_steps=fm_cfg.get("num_steps", 50),
        n_hutchinson=fm_cfg.get("n_hutchinson", 1),
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded EMA weights from {weights_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Test split
    # ------------------------------------------------------------------
    test_records = sort_records_by_npz(_load_splits(output_dir, config, dataset_dir, spec))
    print(f"Test set: {len(test_records)} records (sorted by npz for sequential I/O)")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    test_ds = spec.dataset_class(test_records, dataset_dir, norm_stats=norm_stats)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # Inference — reverse ODE + Hutchinson log-likelihood
    # ------------------------------------------------------------------
    n_score_samples = fm_cfg.get("score_samples", 20)

    print(f"Running FM anomaly scoring (CFM reconstruction, {n_score_samples} samples per signal)...")
    all_scores: list[float] = []
    all_labels: list[str] = []

    t0 = time.time()
    for signals, labels in tqdm(test_loader, desc="FM eval", unit="batch"):
        signals = signals.to(device)
        batch_scores = cfm_score(model.unet, signals, n_samples=n_score_samples)
        all_scores.extend(batch_scores.cpu().numpy().tolist())
        all_labels.extend(labels)

    print(f"Scoring complete in {time.time()-t0:.1f}s")

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    binary_labels = (all_labels != spec.normal_label).astype(int)

    # ------------------------------------------------------------------
    # AUROC + PR-AUC
    # ------------------------------------------------------------------
    auroc = float(roc_auc_score(binary_labels, all_scores))
    print(f"AUROC: {auroc:.4f}")

    precision_arr, recall_arr, _ = precision_recall_curve(binary_labels, all_scores)
    pr_auc = float(auc(recall_arr, precision_arr))
    print(f"PR-AUC: {pr_auc:.4f}")

    # Per-class one-vs-rest AUROC
    per_class_auroc: dict[str, float] = {}
    for cls in spec.class_colors:
        if cls == spec.normal_label:
            continue
        mask = (all_labels == spec.normal_label) | (all_labels == cls)
        if mask.sum() == 0:
            continue
        y_true = (all_labels[mask] == cls).astype(int)
        y_score = all_scores[mask]
        if y_true.sum() == 0:
            continue
        per_class_auroc[cls] = float(roc_auc_score(y_true, y_score))
        print(f"  AUROC {cls} vs NORM: {per_class_auroc[cls]:.4f}")

    # ------------------------------------------------------------------
    # Plot 1 — Precision-Recall curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_arr, precision_arr, color="#8B5CF6", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Flow Matching Precision-Recall Curve (PR-AUC = {pr_auc:.3f})")
    ax.grid(alpha=0.3)
    pr_path = output_dir / "fm_pr_curve.png"
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PR curve saved to {pr_path}")

    # ------------------------------------------------------------------
    # Plot 2 — Per-class score distribution
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls, color in spec.class_colors.items():
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
        label=f"Threshold ({threshold_percentile}th pct = {threshold:.1f})",
    )
    ax.set_xlabel("Anomaly Score  −log p(x)")
    ax.set_ylabel("Density")
    ax.set_title("Flow Matching Anomaly Score Distribution by Class")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    dist_path = output_dir / "fm_score_distribution.png"
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Score distribution saved to {dist_path}")

    # ------------------------------------------------------------------
    # Plot 3 — Per-class AUROC bar chart
    # ------------------------------------------------------------------
    if per_class_auroc:
        fig, ax = plt.subplots(figsize=(7, 4))
        classes = list(per_class_auroc.keys())
        values = [per_class_auroc[c] for c in classes]
        colors = [spec.class_colors[c] for c in classes]
        bars = ax.bar(classes, values, color=colors, alpha=0.85, width=0.5)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random (0.5)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUROC (one-vs-NORM)")
        ax.set_title("Flow Matching Per-Class AUROC")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        per_class_path = output_dir / "fm_per_class_auroc.png"
        fig.savefig(per_class_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Per-class AUROC chart saved to {per_class_path}")

    # ------------------------------------------------------------------
    # Save / merge metrics.json
    # ------------------------------------------------------------------
    existing_metrics: dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing_metrics = json.load(f)

    existing_metrics["FM"] = {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "per_class_auroc": per_class_auroc,
    }
    with open(metrics_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n── Flow Matching Evaluation Summary ──")
    print(f"  AUROC:              {auroc:.4f}")
    print(f"  PR-AUC:             {pr_auc:.4f}")
    print(f"  Threshold:          {threshold:.4f}  ({threshold_percentile}th pct of NORM val)")
    for cls, v in per_class_auroc.items():
        print(f"  AUROC {cls} vs NORM: {v:.4f}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
