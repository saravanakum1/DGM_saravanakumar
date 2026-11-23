"""MIMIC-IV-ECG dataset implementation.

Supports two storage formats (auto-detected from dataset_dir):
  1. Individual .npy files: <dataset_dir>/npy/<study_id>.npy   (preferred)
  2. Batched .npz files:    <dataset_dir>/npy_batches/ecg_batch_N.npz
                             keys: "study_<study_id>" → (12, 5000) float32

Normal records: report_0 == "Sinus rhythm" AND report_2 == "Normal ECG"
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.preprocess.pipeline import _load_one

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

MIMIC_NORMAL_LABEL = "NORMAL"

MIMIC_CLASS_COLORS = {
    "NORMAL": "#22C55E",
    "AF":     "#EF4444",
    "ST":     "#F97316",
    "OTHER":  "#A855F7",
}

MIMIC_SEQ_LEN = 5000
MIMIC_SAMPLING_RATE = 500
MIMIC_KAGGLE_INPUT_PATH = "/kaggle/input/datasets/sharav95/ecg-npy-batches"
MIMIC_REQUIRED_FILES = ["machine_measurements.csv"]

# Ordered rules applied to concatenated report_* text — first match wins
_LABEL_RULES: list[tuple[str, str]] = [
    ("atrial fibrillation", "AF"),
    ("atrial flutter", "AF"),
    ("st elevation", "ST"),
    ("st depression", "ST"),
]


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def _extract_label(row: pd.Series, report_cols: list[str]) -> str:
    texts = " ".join(str(row[c] if c in row.index else "").lower() for c in report_cols)
    for keyword, cls in _LABEL_RULES:
        if keyword in texts:
            return cls
    return "OTHER"


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------

def _detect_format(dataset_dir: Path) -> str:
    """Return 'npy' if individual .npy files exist, 'npz' if batch files exist, else raise."""
    if (dataset_dir / "npy").is_dir() and any((dataset_dir / "npy").glob("*.npy")):
        return "npy"
    if (dataset_dir / "npy_batches").is_dir() and any((dataset_dir / "npy_batches").glob("*.npz")):
        return "npz"
    # npz files at dataset root (no subdirectory)
    if any(dataset_dir.glob("ecg_batch_*.npz")):
        return "npz-root"
    raise FileNotFoundError(
        f"No preprocessed data found in {dataset_dir}. "
        "Expected npy/, npy_batches/, or ecg_batch_*.npz files at root."
    )


def _build_npz_index(dataset_dir: Path, fmt: str) -> dict[str, tuple[Path, str]]:
    """Scan all .npz batch files and return {study_id: (npz_path, key)} index."""
    index = {}
    npz_dir = dataset_dir / "npy_batches" if fmt == "npz" else dataset_dir
    npz_files = sorted(npz_dir.glob("ecg_batch_*.npz"))
    print(f"  Indexing {len(npz_files)} npz batch files...", flush=True)
    for npz_path in npz_files:
        if npz_path.stat().st_size < 1_000_000:
            continue  # skip corrupted/empty files
        try:
            with np.load(npz_path) as data:
                for key in data.files:
                    study_id = key.replace("study_", "")
                    index[study_id] = (npz_path, key)
        except Exception:
            continue
    print(f"  Indexed {len(index)} records from npz batches.", flush=True)
    return index


def build_splits(config: dict, dataset_dir: str | None = None) -> tuple[list, list, list]:
    """Build train / val / test record lists from MIMIC-IV-ECG metadata.

    Auto-detects storage format: individual .npy files or batched .npz files.
    Splits by subject_id to prevent patient leakage.

    Returns:
        train_records: list of (path_or_key, "NORMAL") — NORMAL only
        val_records:   list of (path_or_key, "NORMAL") — NORMAL only
        test_records:  list of (path_or_key, label)    — all classes

    path_or_key is:
      - npy format:  str path to .npy file
      - npz format:  "npz:<npz_path>:<key>" — parsed by MIMICDataset
    """
    if dataset_dir is None:
        dataset_dir = config["data"]["dataset_dir"]
    val_split = config["data"]["val_split"]
    base = Path(dataset_dir)

    fmt = _detect_format(base)
    print(f"  Detected storage format: {fmt}", flush=True)

    meas_df = pd.read_csv(base / "machine_measurements.csv", low_memory=False,
                          dtype={"study_id": str, "subject_id": str})
    report_cols = [c for c in meas_df.columns if c.startswith("report_")]

    normal_mask = (
        (meas_df["report_0"] == "Sinus rhythm") &
        (meas_df["report_2"] == "Normal ECG")
    )
    normal_df = meas_df[normal_mask].copy()
    normal_df["superclass"] = MIMIC_NORMAL_LABEL
    anomaly_df = meas_df[~normal_mask].copy()
    anomaly_df["superclass"] = anomaly_df.apply(lambda row: _extract_label(row, report_cols), axis=1)
    df = pd.concat([normal_df, anomaly_df], ignore_index=True)

    if fmt == "npy":
        npy_dir = base / "npy"
        available = {p.stem for p in npy_dir.glob("*.npy")}
        df = df[df["study_id"].isin(available)]
        def _record_key(study_id: str) -> str:
            return str(npy_dir / f"{study_id}.npy")
    else:  # "npz" or "npz-root"
        npz_index = _build_npz_index(base, fmt)
        df = df[df["study_id"].isin(npz_index)]
        def _record_key(study_id: str) -> str:
            npz_path, key = npz_index[study_id]
            return f"npz:{npz_path}:{key}"

    subjects = sorted(df["subject_id"].unique())
    n_test = max(1, int(len(subjects) * 0.1))
    test_subjects = set(subjects[-n_test:])
    trainval_subjects = set(subjects[:-n_test])

    test_df = df[df["subject_id"].isin(test_subjects)]
    trainval_df = df[df["subject_id"].isin(trainval_subjects)]

    test_records = [(_record_key(row["study_id"]), row["superclass"])
                    for _, row in test_df.iterrows()]

    norm_pool = trainval_df[trainval_df["superclass"] == MIMIC_NORMAL_LABEL]
    norm_pool = norm_pool.sample(frac=1, random_state=42)
    n_val = int(len(norm_pool) * val_split)
    val_records = [(_record_key(row["study_id"]), MIMIC_NORMAL_LABEL)
                   for _, row in norm_pool.iloc[:n_val].iterrows()]
    train_records = [(_record_key(row["study_id"]), MIMIC_NORMAL_LABEL)
                     for _, row in norm_pool.iloc[n_val:].iterrows()]

    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def _load_signal(record_key: str) -> np.ndarray:
    """Load a signal from either npy path or npz:<path>:<key> reference."""
    if record_key.startswith("npz:"):
        _, npz_path, key = record_key.split(":", 2)
        with np.load(npz_path) as data:
            return data[key].astype(np.float32)
    return np.load(record_key).astype(np.float32)


def fit_norm_stats(
    records: list[tuple[str, str]],
    dataset_dir: str,
) -> dict:
    """Compute per-lead mean and std over NORMAL train records.

    Groups npz records by file so each batch file is opened only once
    (~801 opens instead of 74,706). Uses Welford online algorithm so
    no full stack is held in RAM.
    """
    npz_groups: dict[str, list[str]] = {}
    npy_keys: list[str] = []
    for key, _ in records:
        if key.startswith("npz:"):
            _, npz_path, npz_key = key.split(":", 2)
            npz_groups.setdefault(npz_path, []).append(npz_key)
        else:
            npy_keys.append(key)

    lead_sum = np.zeros(12, dtype=np.float64)
    lead_sum_sq = np.zeros(12, dtype=np.float64)
    count = 0
    total_files = len(npz_groups)

    for file_idx, (npz_path, keys) in enumerate(npz_groups.items(), 1):
        if file_idx % 100 == 0 or file_idx == total_files:
            print(f"  norm stats: {file_idx}/{total_files} files, {count} records", flush=True)
        with np.load(npz_path) as data:
            for npz_key in keys:
                sig = data[npz_key].astype(np.float64)  # (12, 5000)
                lead_sum += sig.mean(axis=1)
                lead_sum_sq += (sig ** 2).mean(axis=1)
                count += 1

    for key in npy_keys:
        sig = np.load(key).astype(np.float64)
        lead_sum += sig.mean(axis=1)
        lead_sum_sq += (sig ** 2).mean(axis=1)
        count += 1

    mean = lead_sum / count
    std = np.sqrt(np.maximum(lead_sum_sq / count - mean ** 2, 0.0))
    print(f"  norm stats done: {count} records.", flush=True)
    return {"mean": mean.tolist(), "std": std.tolist()}


# ---------------------------------------------------------------------------
# Record sorting helper
# ---------------------------------------------------------------------------

def sort_records_by_npz(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Sort records by npz file path so each file is read sequentially.

    With lazy loading, random shuffle causes ~1 file open per sample (worst case).
    Sorting by npz path means each file is opened once and all its records are
    read consecutively — reducing 74,706 opens to ~801 opens per epoch.
    Has no effect on npy-format records (sort is stable, npy keys sort after npz keys).
    """
    return sorted(records, key=lambda r: r[0])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIMICDataset(Dataset):
    """PyTorch Dataset for MIMIC-IV-ECG.

    Two modes:
      - Unprefetched (default): lazy per-sample npz reads. Low RAM, slow GPU utilisation.
      - Prefetched: call prefetch() once to load all signals into a pre-allocated tensor.
        GPU trains from RAM at full speed. Safe on Kaggle (30 GB RAM) because signals are
        written directly into a pre-allocated float32 tensor — no intermediate list/stack.
    """

    def __init__(
        self,
        records: list[tuple[str, str]],
        dataset_dir: str,
        norm_stats: dict | None = None,
        num_workers: int | None = None,
    ):
        self.record_keys = [key for key, _ in records]
        self.labels = [label for _, label in records]
        self.norm_stats = norm_stats
        self.signals: torch.Tensor | None = None  # populated by prefetch()
        print(f"  MIMICDataset: {len(records)} records (lazy loading)", flush=True)

    def prefetch(self, num_workers: int = 4) -> None:
        """Load all records into a pre-allocated RAM tensor using parallel workers.

        Each worker loads one npz file at a time — num_workers files load in
        parallel, cutting prefetch time from ~5 min to ~1-2 min.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from backend.preprocess.pipeline import normalize

        N = len(self.record_keys)
        print(f"  Prefetching {N} signals into RAM (workers={num_workers})...", flush=True)

        # Pre-allocate output tensor
        self.signals = torch.empty(N, 12, 5000, dtype=torch.float32)

        # Group by npz file
        npz_groups: dict[str, list[tuple[int, str]]] = {}
        npy_items: list[tuple[int, str]] = []
        for idx, key in enumerate(self.record_keys):
            if key.startswith("npz:"):
                _, npz_path, npz_key = key.split(":", 2)
                npz_groups.setdefault(npz_path, []).append((idx, npz_key))
            else:
                npy_items.append((idx, key))

        total = len(npz_groups)
        completed = [0]  # mutable counter shared across threads

        def _load_npz(npz_path: str, items: list[tuple[int, str]]) -> None:
            with np.load(npz_path) as data:
                for idx, npz_key in items:
                    sig = data[npz_key].astype(np.float32)
                    if self.norm_stats is not None:
                        sig = normalize(sig, self.norm_stats)
                    self.signals[idx] = torch.from_numpy(sig)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_load_npz, path, items): path
                       for path, items in npz_groups.items()}
            for future in as_completed(futures):
                future.result()  # re-raises any exception from the worker
                completed[0] += 1
                if completed[0] % 100 == 0 or completed[0] == total:
                    print(f"  Prefetch: {completed[0]}/{total} files", flush=True)

        for idx, npy_path in npy_items:
            sig = np.load(npy_path).astype(np.float32)
            if self.norm_stats is not None:
                sig = normalize(sig, self.norm_stats)
            self.signals[idx] = torch.from_numpy(sig)

        print(f"  Prefetch done: {N} signals ({self.signals.nbytes / 1e9:.1f} GB)", flush=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        if self.signals is not None:
            return self.signals[idx], self.labels[idx]

        # Fallback: lazy load from disk
        key = self.record_keys[idx]
        if key.startswith("npz:"):
            _, npz_path, npz_key = key.split(":", 2)
            with np.load(npz_path) as data:
                sig = data[npz_key].astype(np.float32)
        else:
            sig = np.load(key).astype(np.float32)

        if self.norm_stats is not None:
            from backend.preprocess.pipeline import normalize
            sig = normalize(sig, self.norm_stats)

        return torch.from_numpy(sig), self.labels[idx]
