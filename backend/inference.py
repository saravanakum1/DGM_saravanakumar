import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend.models.vae import VAE
from backend.models.ddpm import DDPM
from backend.models.flow_matching import FlowMatching


def load_models(config: dict, artifacts_dir: Path, device: torch.device) -> dict:
    models = {}

    vae_path = artifacts_dir / config["artifacts"]["weights_file"]
    if vae_path.exists():
        vae = VAE(config)
        vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
        vae.eval()
        vae.to(device)
        models["VAE"] = vae

    ddpm_path = artifacts_dir / "ddpm_ema_weights.pt"
    if ddpm_path.exists():
        ddpm = DDPM(config)
        ddpm.load_state_dict(torch.load(ddpm_path, map_location=device, weights_only=True))
        ddpm.eval()
        ddpm.to(device)
        models["DDPM"] = ddpm

    fm_path = artifacts_dir / "fm_ema_weights.pt"
    if fm_path.exists():
        fm_cfg = config.get("flow_matching", {})
        seq_len = config.get("data", {}).get("sampling_rate", 500) * 10
        fm = FlowMatching(
            time_dim=fm_cfg.get("time_dim", 128),
            num_steps=fm_cfg.get("num_steps", 50),
            n_hutchinson=fm_cfg.get("n_hutchinson", 1),
            seq_len=seq_len,
        )
        fm.load_state_dict(torch.load(fm_path, map_location=device, weights_only=True))
        fm.eval()
        fm.to(device)
        models["FM"] = fm

    return models


def run_inference(
    models: dict,
    signal_tensor: torch.Tensor,
    norm_stats: dict,
    config: dict,
    device: torch.device,
) -> dict:
    results = {}
    x = signal_tensor.to(device)

    if "VAE" in models:
        model = models["VAE"]
        t0 = time.time()
        with torch.no_grad():
            recon, mu, logvar = model(x)
        score = float(F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2])[0].cpu())
        heatmap = (recon - x).pow(2).mean(dim=1)[0].cpu().numpy()
        elapsed_ms = int((time.time() - t0) * 1000)
        results["VAE"] = {
            "anomaly_score": score,
            "heatmap": heatmap,
            "inference_ms": elapsed_ms,
        }

    if "DDPM" in models:
        model = models["DDPM"]
        t0 = time.time()
        scores = model.anomaly_score(x)
        score = float(scores[0].cpu())
        recon = model.reconstruct(x)
        heatmap = (recon - x).pow(2).mean(dim=1)[0].cpu().numpy()
        elapsed_ms = int((time.time() - t0) * 1000)
        results["DDPM"] = {
            "anomaly_score": score,
            "heatmap": heatmap,
            "inference_ms": elapsed_ms,
        }

    if "FM" in models:
        model = models["FM"]
        t0 = time.time()
        scores, heatmap = model.anomaly_score(x)
        score = float(scores[0].cpu())
        elapsed_ms = int((time.time() - t0) * 1000)
        results["FM"] = {
            "anomaly_score": score,
            "heatmap": heatmap.numpy(),
            "inference_ms": elapsed_ms,
        }

    return results
