"""
Offline preprocessing for MIMIC-IV-ECG 500 Hz records.

Reads machine_measurements.csv, loads each record from the MIMIC waveform
directory structure, applies:
  1. Bandpass filter  0.5–40 Hz  (Butterworth order 4, zero-phase)
  2. Notch filter     60 Hz      (powerline noise for US recordings)
  3. Z-score normalization per lead

Saves each record as <dataset_dir>/npy/<study_id>.npy
Shape: (12, 5000)  float32

MIMIC waveform path pattern:
  files/p<subj[:4]>/p<subj>/s<study_id>/<study_id>

Usage:
    uv run python scripts/process_mimic.py
    uv run python scripts/process_mimic.py --dataset-dir /path/to/mimic-iv-ecg --workers 8

    # Kaggle (input is read-only — write npy to working dir):
    python scripts/process_mimic.py \
        --dataset-dir /kaggle/input/mimic-iv-ecg-... \
        --output-dir /kaggle/working/npy \
        --workers 4

Records that fail (corrupt file, wrong shape, missing leads) are skipped with a warning.
"""

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import tomllib
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Signal processing helpers (same as process_ptbxl.py but 60 Hz notch)
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

# Standard 12-lead order expected in output
_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _process_one(args: tuple) -> tuple[str, bool, str]:
    """Load, filter, z-score, save one MIMIC record. Returns (study_id, success, message)."""
    rec_path, out_path, fs = args
    study_id = Path(out_path).stem

    try:
        record = wfdb.rdrecord(str(rec_path))
        sig = record.p_signal  # (T, n_leads)

        if sig is None:
            return study_id, False, "p_signal is None"

        # MIMIC records are 500 Hz × 10 s = 5000 samples
        if sig.shape[0] != 5000:
            return study_id, False, f"unexpected length {sig.shape[0]} (expected 5000)"

        # Map leads to canonical order; fill missing leads with zeros
        available = {name.upper(): idx for idx, name in enumerate(record.sig_name)}
        out = np.zeros((12, 5000), dtype=np.float64)
        for i, lead in enumerate(_LEAD_NAMES):
            if lead in available:
                out[i] = sig[:, available[lead]]

        out = np.nan_to_num(out, nan=0.0)

        for i in range(out.shape[0]):
            lead_sig = out[i]
            lead_sig = butter_bandpass(lead_sig, 0.5, 40.0, fs)
            lead_sig = notch_filter(lead_sig, freq=60.0, q=30, fs=fs)
            out[i] = zscore(lead_sig)

        np.save(out_path, out.astype(np.float32))
        return study_id, True, ""

    except Exception as e:
        return study_id, False, str(e)


# ---------------------------------------------------------------------------
# MIMIC path helper
# ---------------------------------------------------------------------------

def _record_path(dataset_dir: Path, subject_id: str, study_id: str) -> Path:
    """Build MIMIC-IV-ECG waveform path from subject/study IDs."""
    subj = str(subject_id).zfill(8)  # p00000000
    folder = f"p{subj[:4]}"          # p0000
    return dataset_dir / "files" / folder / f"p{subj}" / f"s{study_id}" / str(study_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV-ECG 500 Hz records to .npy")
    parser.add_argument(
        "--dataset-dir", default=None,
        help="Path to MIMIC-IV-ECG dataset root. Defaults to config.toml [data] dataset_dir."
    )
    parser.add_argument(
        "--workers", type=int, default=min(cpu_count(), 8),
        help="Parallel worker processes (default: min(cpu_count, 8))"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write .npy files into. Defaults to <dataset-dir>/npy/. "
             "Use this on Kaggle where dataset_dir is read-only."
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

    csv_path = dataset_dir / "machine_measurements.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"machine_measurements.csv not found at {csv_path}")

    out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {out_dir}")
    print(f"Workers : {args.workers}")

    # Load CSV — need subject_id and study_id to build paths
    df = pd.read_csv(csv_path, dtype={"subject_id": str, "study_id": str})
    df = df.dropna(subset=["subject_id", "study_id"])
    print(f"Records in CSV: {len(df)}")

    # Build work list — skip existing unless --overwrite
    work = []
    skipped = 0
    missing = 0

    for _, row in df.iterrows():
        study_id = str(row["study_id"])
        subject_id = str(row["subject_id"])
        out_path = out_dir / f"{study_id}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        rec_path = _record_path(dataset_dir, subject_id, study_id)
        # wfdb.rdrecord appends .hea automatically; check that file exists
        if not rec_path.with_suffix(".hea").exists():
            missing += 1
            continue

        work.append((rec_path, out_path, 500))

    if skipped:
        print(f"Skipping {skipped} already-processed records (use --overwrite to redo)")
    if missing:
        print(f"Skipping {missing} records with no matching waveform file on disk")
    print(f"Processing {len(work)} records...")

    if not work:
        print("Nothing to do.")
        return

    succeeded, failed = 0, 0
    failures = []

    with Pool(processes=args.workers) as pool:
        for study_id, ok, msg in tqdm(
            pool.imap_unordered(_process_one, work),
            total=len(work),
            desc="MIMIC-IV-ECG",
        ):
            if ok:
                succeeded += 1
            else:
                failed += 1
                failures.append((study_id, msg))

    print(f"\nDone. {succeeded} saved, {failed} failed.")
    if failures:
        print(f"Failed records ({len(failures)}):")
        for study_id, msg in failures[:20]:
            print(f"  {study_id}: {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    print(f"Output: {out_dir}  ({succeeded} .npy files)")


if __name__ == "__main__":
    main()
