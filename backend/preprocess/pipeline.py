"""Shared preprocessing utilities.

Signal filtering and WFDB loading are done offline by preprocessing scripts.
This module provides only what training needs: npy loading, normalization, and
a picklable _load_one helper for multiprocessing.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

N_LEADS = 12
TARGET_TIMESTEPS = 5000  # 500 Hz × 10 s


# ---------------------------------------------------------------------------
# npy loading
# ---------------------------------------------------------------------------

def load_npy(path: str) -> np.ndarray:
    """Load a preprocessed .npy record. Returns (12, 5000) float32."""
    return np.load(path)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(signal: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize signal per lead using precomputed mean/std.

    Args:
        signal: (12, T) float32
        stats: dict with "mean" and "std" lists of length 12

    Returns:
        Normalized signal, same shape.
    """
    mean = np.array(stats["mean"], dtype=np.float32).reshape(12, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(12, 1)
    std = np.where(std < 1e-8, 1.0, std)
    return ((signal - mean) / std).astype(np.float32)


def save_norm_stats(stats: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parallel loading helper — module-level so multiprocessing can pickle it
# ---------------------------------------------------------------------------

def _load_one(args: tuple) -> np.ndarray:
    """Load a single .npy record and optionally normalize. Returns (12, 5000) float32."""
    npy_path, norm_stats = args
    sig = load_npy(npy_path)
    if norm_stats is not None:
        sig = normalize(sig, norm_stats)
    return sig


# ---------------------------------------------------------------------------
# Backward-compat re-exports — existing tests import from here
# ---------------------------------------------------------------------------

from backend.preprocess.ptbxl import (  # noqa: E402, F401
    PTBXLDataset,
    build_splits,
    fit_norm_stats,
    SUPERCLASS_MAP,
)
