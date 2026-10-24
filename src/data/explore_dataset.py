"""
Quick dataset explorer for PTB-XL.

Usage:
    uv run python src/data/explore_dataset.py
"""

import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.download_dataset import get_dataset_path

dataset_dir = Path(get_dataset_path())

# ── 1. Load metadata ──────────────────────────────────────────────────────────
df = pd.read_csv(dataset_dir / "ptbxl_database.csv", index_col="ecg_id")
print(f"Total records: {len(df)}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# ── 2. Superclass distribution ────────────────────────────────────────────────
scp_df = pd.read_csv(dataset_dir / "scp_statements.csv", index_col=0)
lookup = {
    str(code).strip(): row["diagnostic_class"].strip()
    for code, row in scp_df.iterrows()
    if isinstance(row.get("diagnostic_class"), str)
    and row["diagnostic_class"].strip() in {"NORM", "MI", "STTC", "CD", "HYP"}
}

def dominant_class(scp_str):
    try:
        codes = ast.literal_eval(scp_str)
        best, score = None, -1
        for c, s in codes.items():
            sc = lookup.get(c.strip())
            if sc and s > score:
                best, score = sc, s
        return best
    except Exception:
        return None

df["superclass"] = df["scp_codes"].apply(dominant_class)

print(f"\nSuperclass distribution:")
print(df["superclass"].value_counts())

print(f"\nFold distribution:")
print(df["strat_fold"].value_counts().sort_index())

# ── 3. Signal shape check ─────────────────────────────────────────────────────
sample_path = str(dataset_dir / df["filename_lr"].iloc[0])
signal, meta = wfdb.rdsamp(sample_path)
print(f"\nSample signal shape (timesteps, leads): {signal.shape}")
print(f"Sampling frequency: {meta['fs']} Hz")
print(f"Lead names: {meta['sig_name']}")
print(f"Signal value range: [{signal.min():.4f}, {signal.max():.4f}] mV")

# ── 4. Plot one NORM and one MI ECG ──────────────────────────────────────────
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
time = np.arange(signal.shape[0]) / 100  # seconds

def plot_ecg(record_path, title, ax_row, axes):
    sig, _ = wfdb.rdsamp(record_path)
    for i, ax in enumerate(axes):
        ax.plot(time, sig[:, i], lw=0.8, color="#1E40AF")
        ax.set_ylabel(LEADS[i], fontsize=7, rotation=0, labelpad=20)
        ax.set_ylim(sig.min() - 0.1, sig.max() + 0.1)
        ax.tick_params(axis="both", labelsize=6)
        if i == 0:
            ax.set_title(title, fontsize=9, fontweight="bold")
        if i == 11:
            ax.set_xlabel("Time (s)", fontsize=7)

norm_row = df[df["superclass"] == "NORM"].iloc[0]
mi_row = df[df["superclass"] == "MI"].iloc[0]

fig, axes = plt.subplots(12, 2, figsize=(14, 16), sharex=True)
fig.suptitle("PTB-XL Sample ECGs", fontsize=12, fontweight="bold")

plot_ecg(str(dataset_dir / norm_row["filename_lr"]), "NORM (Normal)", 0, axes[:, 0])
plot_ecg(str(dataset_dir / mi_row["filename_lr"]),   "MI (Myocardial Infarction)", 1, axes[:, 1])

plt.tight_layout()
out = Path("artifacts/ecg_samples.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nECG plot saved to {out}")

# ── 5. Basic signal stats per lead ────────────────────────────────────────────
print("\nSignal stats for the sample NORM record (mV):")
sig_norm, _ = wfdb.rdsamp(str(dataset_dir / norm_row["filename_lr"]))
stats = pd.DataFrame({
    "lead": LEADS,
    "mean": sig_norm.mean(axis=0).round(4),
    "std":  sig_norm.std(axis=0).round(4),
    "min":  sig_norm.min(axis=0).round(4),
    "max":  sig_norm.max(axis=0).round(4),
})
print(stats.to_string(index=False))
