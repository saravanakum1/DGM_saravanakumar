"""PTB-XL dataset implementation.

Expects .npy files preprocessed offline by process_ptbxl():
  bandpass (0.5-40 Hz) + notch (50 Hz) + z-score per lead → (12, 5000) float32
  stored at: <dataset_dir>/npy/<record_stem>.npy
"""

import ast
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.preprocess.pipeline import _load_one, normalize

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

SUPERCLASS_MAP = {
    "NORM": "NORM",
    "MI": "MI",
    "STTC": "STTC",
    "CD": "CD",
    "HYP": "HYP",
}

PTBXL_NORMAL_LABEL = "NORM"

PTBXL_CLASS_COLORS = {
    "NORM": "#22C55E",
    "MI":   "#EF4444",
    "STTC": "#F97316",
    "CD":   "#EAB308",
    "HYP":  "#A855F7",
}

PTBXL_SEQ_LEN = 5000
PTBXL_SAMPLING_RATE = 500
PTBXL_KAGGLE_INPUT_PATH = "/kaggle/input/datasets/sharav95/ecg-ptbxl-data"
PTBXL_REQUIRED_FILES = ["ptbxl_database.csv", "scp_statements.csv", "npy"]


# ---------------------------------------------------------------------------
# Label extraction from scp_codes
# ---------------------------------------------------------------------------

def _load_superclass_lookup(dataset_dir: str) -> dict[str, str]:
    scp_path = Path(dataset_dir) / "scp_statements.csv"
    scp_df = pd.read_csv(scp_path, index_col=0)
    lookup: dict[str, str] = {}
    for code, row in scp_df.iterrows():
        sc = row.get("diagnostic_class", "")
        if isinstance(sc, str) and sc.strip() in SUPERCLASS_MAP:
            lookup[str(code).strip()] = sc.strip()
    return lookup


def _aggregate_superclass(scp_codes_str: str, lookup: dict[str, str]) -> str | None:
    try:
        codes: dict = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return None
    classes = {lookup[c.strip()] for c in codes if c.strip() in lookup}
    # Single unambiguous superclass only
    if len(classes) == 1:
        return classes.pop()
    return None


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------

def build_splits(config: dict, dataset_dir: str | None = None) -> tuple[list, list, list]:
    """Build train / val / test record lists from PTB-XL metadata.

    Reads ptbxl_database.csv. Expects .npy files at <dataset_dir>/npy/<stem>.npy
    where stem = Path(filename_hr).stem  (e.g. "00001_hr").

    Returns:
        train_records: list of (npy_path, "NORM") — NORM only, folds 1-9
        val_records:   list of (npy_path, "NORM") — NORM only, held-out fraction
        test_records:  list of (npy_path, superclass) — all classes, fold 10
    """
    if dataset_dir is None:
        dataset_dir = config["data"]["dataset_dir"]
    val_split = config["data"]["val_split"]
    npy_dir = Path(dataset_dir) / "npy"

    df = pd.read_csv(Path(dataset_dir) / "ptbxl_database.csv", index_col="ecg_id")
    lookup = _load_superclass_lookup(dataset_dir)
    df["superclass"] = df["scp_codes"].apply(lambda x: _aggregate_superclass(x, lookup))
    df = df[df["superclass"].notna()].copy()

    def _npy_path(filename_hr: str) -> str:
        stem = Path(filename_hr).stem  # e.g. "00001_hr"
        return str(npy_dir / f"{stem}.npy")

    # Test set: fold 10, all superclasses
    test_df = df[df["strat_fold"] == 10]
    test_records = [(_npy_path(row["filename_hr"]), row["superclass"]) for _, row in test_df.iterrows()]

    # Train/val: NORM only, folds 1-9
    norm_pool = df[(df["strat_fold"] != 10) & (df["superclass"] == "NORM")]
    norm_pool = norm_pool.sample(frac=1, random_state=42)
    n_val = int(len(norm_pool) * val_split)
    val_df = norm_pool.iloc[:n_val]
    train_df = norm_pool.iloc[n_val:]

    train_records = [(_npy_path(row["filename_hr"]), "NORM") for _, row in train_df.iterrows()]
    val_records = [(_npy_path(row["filename_hr"]), "NORM") for _, row in val_df.iterrows()]

    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def fit_norm_stats(
    records: list[tuple[str, str]],
    dataset_dir: str,
) -> dict:
    """Compute per-lead mean and std over NORM train .npy files."""
    signals = [np.load(npy_path) for npy_path, _ in records]
    stack = np.stack(signals, axis=0)  # (N, 12, 5000)
    return {"mean": stack.mean(axis=(0, 2)).tolist(), "std": stack.std(axis=(0, 2)).tolist()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PTBXLDataset(Dataset):
    """PyTorch Dataset for PTB-XL. Loads preprocessed .npy files into RAM."""

    def __init__(
        self,
        records: list[tuple[str, str]],
        dataset_dir: str,
        norm_stats: dict | None = None,
        num_workers: int | None = None,
    ):
        self.labels = [label for _, label in records]
        args = [(npy_path, norm_stats) for npy_path, _ in records]
        n_workers = num_workers if num_workers is not None else min(cpu_count(), 8)
        print(f"  Loading {len(args)} records using {n_workers} CPU cores...", flush=True)
        with Pool(processes=n_workers) as pool:
            signals = pool.map(_load_one, args)
        self.signals = torch.from_numpy(np.stack(signals, axis=0))  # (N, 12, 5000)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        return self.signals[idx], self.labels[idx]
