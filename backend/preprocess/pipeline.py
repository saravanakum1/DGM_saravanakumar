import ast
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPERCLASS_MAP = {
    "NORM": "NORM",
    "MI": "MI",
    "STTC": "STTC",
    "CD": "CD",
    "HYP": "HYP",
}

TARGET_TIMESTEPS = 1000
N_LEADS = 12


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def load_record(record_path: str) -> np.ndarray:
    """Load a single PTB-XL record and return shape (12, 1000).

    Args:
        record_path: Full path to record file without extension
                     (e.g. dataset_dir/records100/00000/00001_lr)

    Returns:
        np.ndarray of shape (12, 1000), float32. NaN values are zero-filled.
    """
    signal, _ = wfdb.rdsamp(record_path)  # shape: (1000, 12)
    signal = signal.T.astype(np.float32)  # (12, 1000)
    signal = np.nan_to_num(signal, nan=0.0)
    return signal


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, fs: int = 100) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter (0.5–40 Hz) per lead.

    Args:
        signal: np.ndarray of shape (12, 1000)
        fs: sampling rate in Hz

    Returns:
        Filtered signal, same shape.
    """
    sos = butter(4, [0.5, 40.0], btype="band", fs=fs, output="sos")
    filtered = np.stack(
        [sosfiltfilt(sos, signal[lead]) for lead in range(signal.shape[0])],
        axis=0,
    )
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Superclass label extraction
# ---------------------------------------------------------------------------

def _load_superclass_lookup(dataset_dir: str) -> dict[str, str]:
    """Build a dict mapping SCP code -> superclass from scp_statements.csv."""
    scp_path = Path(dataset_dir) / "scp_statements.csv"
    scp_df = pd.read_csv(scp_path, index_col=0)
    lookup: dict[str, str] = {}
    for code, row in scp_df.iterrows():
        sc = row.get("diagnostic_class", "")
        if isinstance(sc, str) and sc.strip() in SUPERCLASS_MAP:
            lookup[str(code).strip()] = sc.strip()
    return lookup


def _extract_superclass(scp_codes_str: str, lookup: dict[str, str]) -> str | None:
    """Return the dominant superclass for a record, or None if undetermined."""
    try:
        codes: dict = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return None

    # Pick the superclass with the highest likelihood score
    best_class, best_score = None, -1.0
    for code, score in codes.items():
        sc = lookup.get(code.strip())
        if sc and score > best_score:
            best_class, best_score = sc, score
    return best_class


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------

def build_splits(config: dict, dataset_dir: str | None = None) -> tuple[list, list, list]:
    """Build train / val / test record lists from PTB-XL metadata.

    Args:
        config: loaded config.toml dict
        dataset_dir: override dataset path (e.g. from get_dataset_path()).
                     Falls back to config["data"]["dataset_dir"] if not provided.

    Returns:
        train_records: list of (filename_lr, "NORM") — NORM only, folds 1-9
        val_records:   list of (filename_lr, "NORM") — NORM only, held-out fraction
        test_records:  list of (filename_lr, superclass) — all classes, fold 10
    """
    if dataset_dir is None:
        dataset_dir = config["data"]["dataset_dir"]
    val_split = config["data"]["val_split"]

    db_path = Path(dataset_dir) / "ptbxl_database.csv"
    df = pd.read_csv(db_path, index_col="ecg_id")

    lookup = _load_superclass_lookup(dataset_dir)
    df["superclass"] = df["scp_codes"].apply(
        lambda x: _extract_superclass(x, lookup)
    )
    df = df[df["superclass"].notna()].copy()

    # Test set: fold 10, all superclasses
    test_df = df[df["strat_fold"] == 10]
    test_records = [
        (row["filename_lr"], row["superclass"])
        for _, row in test_df.iterrows()
    ]

    # NORM pool: folds 1–9
    norm_pool = df[(df["strat_fold"] != 10) & (df["superclass"] == "NORM")]
    norm_pool = norm_pool.sample(frac=1, random_state=42)  # shuffle

    n_val = int(len(norm_pool) * val_split)
    val_df = norm_pool.iloc[:n_val]
    train_df = norm_pool.iloc[n_val:]

    train_records = [(row["filename_lr"], "NORM") for _, row in train_df.iterrows()]
    val_records = [(row["filename_lr"], "NORM") for _, row in val_df.iterrows()]

    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def fit_norm_stats(
    records: list[tuple[str, str]],
    dataset_dir: str,
    fs: int = 100,
) -> dict:
    """Compute per-lead mean and std over all provided records.

    Args:
        records: list of (filename_lr, label) — typically NORM train records
        dataset_dir: root path of PTB-XL dataset
        fs: sampling rate for bandpass filter

    Returns:
        dict with keys "mean" and "std", each a list of 12 floats
    """
    signals = []
    for filename_lr, _ in records:
        record_path = str(Path(dataset_dir) / filename_lr)
        sig = load_record(record_path)
        sig = bandpass_filter(sig, fs=fs)
        signals.append(sig)

    stack = np.stack(signals, axis=0)  # (N, 12, 1000)
    mean = stack.mean(axis=(0, 2)).tolist()  # (12,)
    std = stack.std(axis=(0, 2)).tolist()    # (12,)

    return {"mean": mean, "std": std}


def save_norm_stats(stats: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize(signal: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize signal per lead using precomputed mean/std.

    Args:
        signal: np.ndarray of shape (12, 1000)
        stats: dict with "mean" and "std" lists of length 12

    Returns:
        Normalized signal, same shape.
    """
    mean = np.array(stats["mean"], dtype=np.float32).reshape(12, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(12, 1)
    std = np.where(std < 1e-8, 1.0, std)  # avoid division by zero
    return ((signal - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Parallel loading helper (module-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _load_one(args: tuple) -> np.ndarray:
    """Load, filter, and optionally normalize a single record. Returns (12, 1000) float32."""
    record_path, norm_stats, fs = args
    sig = load_record(record_path)
    sig = bandpass_filter(sig, fs=fs)
    if norm_stats is not None:
        sig = normalize(sig, norm_stats)
    return sig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PTBXLDataset(Dataset):
    """PyTorch Dataset for PTB-XL records.

    Preloads all signals into RAM on init — bandpass filtering and normalization
    are applied once at load time so each epoch reads from memory, not disk.

    Args:
        records: list of (filename_lr, superclass_label)
        dataset_dir: root path of PTB-XL dataset
        norm_stats: dict with "mean" and "std" for normalization (or None to skip)
        fs: sampling rate
    """

    def __init__(
        self,
        records: list[tuple[str, str]],
        dataset_dir: str,
        norm_stats: dict | None = None,
        fs: int = 100,
        num_workers: int | None = None,
    ):
        self.labels = [label for _, label in records]
        args = [
            (str(Path(dataset_dir) / filename_lr), norm_stats, fs)
            for filename_lr, _ in records
        ]
        n_workers = num_workers if num_workers is not None else min(cpu_count(), 8)
        print(f"  Loading {len(args)} records using {n_workers} CPU cores...", flush=True)
        with Pool(processes=n_workers) as pool:
            signals = pool.map(_load_one, args)
        self.signals = torch.from_numpy(np.stack(signals, axis=0))  # (N, 12, 1000)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        return self.signals[idx], self.labels[idx]
