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
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset

N_LEADS = 12
TARGET_TIMESTEPS = 5000  # 500 Hz × 10 s


# ---------------------------------------------------------------------------
# Signal loading and filtering
# ---------------------------------------------------------------------------

def load_record(record_path: str) -> np.ndarray:
    """Load a WFDB record and return (12, 5000) float32."""
    import wfdb
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal.T.astype(np.float32)  # (12, T)
    if sig.shape[1] > 5000:
        sig = sig[:, :5000]
    elif sig.shape[1] < 5000:
        sig = np.pad(sig, ((0, 0), (0, 5000 - sig.shape[1])))
    return sig


def bandpass_filter(signal: np.ndarray, fs: float = 500, lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """Apply a Butterworth bandpass filter to (12, T) signal."""
    nyq = fs / 2.0
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=-1).astype(np.float32)


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
