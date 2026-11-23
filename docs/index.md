# ECG Anomaly Detection — Model Comparison

A unified framework for training and evaluating deep generative models on ECG anomaly detection, supporting multiple datasets via a plugin architecture.

## Models

| Model | Type | Script |
|---|---|---|
| VAE | Variational Autoencoder | `src/train/train_vae.py` |
| DDPM | Denoising Diffusion Probabilistic Model | `src/train/train_ddpm.py` |
| Flow Matching | Continuous Normalizing Flow | `src/train/train_flow_matching.py` |

## Datasets

| Dataset | Source | Hz | Records |
|---|---|---|---|
| PTB-XL | PhysioNet | 500 | ~21,000 |
| MIMIC-IV-ECG | PhysioNet | 500 | ~800,000 |

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Preprocess PTB-XL (run once)
uv run python scripts/process_ptbxl.py --dataset-dir /path/to/ptbxl

# 3. Train
uv run python src/train/train_ddpm.py

# 4. Evaluate
uv run python src/eval/eval_ddpm.py
```

See [Getting Started](getting-started/installation.md) for full setup instructions.
