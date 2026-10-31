import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backend.preprocess.pipeline import load_record
from src.data.download_dataset import get_dataset_path

dataset_dir = get_dataset_path()

with open("artifacts/splits_cache.json") as f:
    splits = json.load(f)

classes = ["MI", "STTC", "CD", "HYP"]
found = {}
for record_path, label in splits["test"]:
    if label in classes and label not in found:
        found[label] = record_path
    if len(found) == len(classes):
        break

for label, record_path in found.items():
    signal = load_record(f"{dataset_dir}/{record_path}")
    rid = record_path.split("/")[-1]
    fname = f"sample_{label}_{rid}.csv"
    pd.DataFrame(signal.T).to_csv(fname, index=False, header=False)
    print(f"Saved {fname}")
