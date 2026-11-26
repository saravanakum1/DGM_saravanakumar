"""
Offline preprocessing for PTB-XL 500 Hz records.

Reads ptbxl_database.csv, loads each record from records500/, applies:
  1. Bandpass filter  0.5–40 Hz  (Butterworth order 4, zero-phase)
  2. Notch filter     50 Hz      (powerline noise for European recordings)
  3. Z-score normalization per lead

Saves each record as <dataset_dir>/npy/<stem>.npy  e.g. 00001_hr.npy
Shape: (12, 5000)  float32

Usage:
    uv run python scripts/process_ptbxl.py
    uv run python scripts/process_ptbxl.py --dataset-dir /path/to/ptbxl --workers 8

Only records with a valid filename_hr are processed. Records that fail
(corrupt file, wrong shape) are skipped with a warning.
"""

import argparse
import ast
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import tomllib
import wfdb
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def butter_bandpass(signal: np.ndarray, lowcut: float, highcut: float, fs: int) -> np.ndarray:
    sos = butter(4, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def notch_filter(signal: np.ndarray, freq: float, q: float, fs: int) -> np.ndarray:
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, signal)


def zscore(signal: np.ndarray) -> np.ndarray:
    std = signal.std()
    if std < 1e-8:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


# ---------------------------------------------------------------------------
# Per-record worker
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> tuple[str, bool, str]:
    """Load, filter, z-score, save one record. Returns (stem, success, message)."""
    rec_path, out_path, fs = args
    stem = Path(out_path).stem
    try:
        record = wfdb.rdrecord(str(rec_path))
        sig = record.p_signal  # (T, 12)
        expected_len = fs * 10  # 500Hz→5000, 100Hz→1000
        if sig is None or sig.shape != (expected_len, 12):
            return stem, False, f"unexpected shape {sig.shape if sig is not None else None}"

        sig = sig.T.astype(np.float64)  # (12, 5000)
        sig = np.nan_to_num(sig, nan=0.0)

        for i in range(sig.shape[0]):
            lead = sig[i]
            lead = butter_bandpass(lead, 0.5, 40.0, fs)
            lead = notch_filter(lead, freq=50.0, q=30, fs=fs)
            sig[i] = zscore(lead)

        np.save(out_path, sig.astype(np.float32))
        return stem, True, ""
    except Exception as e:
        return stem, False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL 500 Hz records to .npy")
    parser.add_argument(
        "--dataset-dir", default=None,
        help="Path to PTB-XL dataset root. Defaults to config.toml [data] dataset_dir."
    )
    parser.add_argument(
        "--workers", type=int, default=min(cpu_count(), 8),
        help="Parallel worker processes (default: min(cpu_count, 8))"
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=None,
        help="Sampling rate: 500 (records500/filename_lr) or 100 (records100/filename_hr). Defaults to config."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for .npy files. Defaults to <dataset_dir>/npy/"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process records even if .npy already exists"
    )
    args = parser.parse_args()

    # Resolve dataset dir
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        config_path = Path(__file__).resolve().parents[1] / "config.toml"
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        dataset_dir = Path(config["data"]["dataset_dir"])

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    # Resolve sampling rate and filename column
    if args.sampling_rate is not None:
        fs = args.sampling_rate
    else:
        config_path = Path(__file__).resolve().parents[1] / "config.toml"
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        fs = config.get("data", {}).get("sampling_rate", 100)

    # PTB-XL naming: filename_hr = 500Hz (high resolution), filename_lr = 100Hz (low resolution)
    filename_col = "filename_hr" if fs == 500 else "filename_lr"
    print(f"Sampling rate: {fs} Hz  (using column: {filename_col})")

    out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {out_dir}")
    print(f"Workers : {args.workers}")

    # Load CSV
    df = pd.read_csv(dataset_dir / "ptbxl_database.csv", index_col="ecg_id")
    df = df[df[filename_col].notna()].copy()
    print(f"Records in CSV with {filename_col}: {len(df)}")

    # Build work list — skip existing unless --overwrite
    work = []
    skipped = 0
    for _, row in df.iterrows():
        rec_path = dataset_dir / row[filename_col]
        stem = Path(row[filename_col]).stem   # e.g. "00001_lr" or "00001_hr"
        out_path = out_dir / f"{stem}.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        work.append((rec_path, out_path, fs))

    if skipped:
        print(f"Skipping {skipped} already-processed records (use --overwrite to redo)")
    print(f"Processing {len(work)} records...")

    if not work:
        print("Nothing to do.")
        return

    succeeded, failed = 0, 0
    failures = []

    with Pool(processes=args.workers) as pool:
        for stem, ok, msg in tqdm(
            pool.imap_unordered(_process_one, work),
            total=len(work),
            desc="PTB-XL",
        ):
            if ok:
                succeeded += 1
            else:
                failed += 1
                failures.append((stem, msg))

    print(f"\nDone. {succeeded} saved, {failed} failed.")
    if failures:
        print("Failed records:")
        for stem, msg in failures:
            print(f"  {stem}: {msg}")

    print(f"Output: {out_dir}  ({succeeded} .npy files)")


if __name__ == "__main__":
    main()
