"""
Local-only utility to download PTB-XL via kagglehub for local development.

NOTE: For training, use a Kaggle Notebook where PTB-XL is pre-attached at
/kaggle/input/ptbxl-electrocardiography-database/. Set config.toml
[data] dataset_dir to that path when running inside Kaggle.

Local usage (optional — for local dataset inspection only):
    uv run python src/data/download_dataset.py

Prerequisites for local use:
    1. Create a Kaggle account at https://www.kaggle.com
    2. Account → API → Create New Token → downloads kaggle.json
    3. Place kaggle.json at ~/.config/kaggle/kaggle.json
       or set env vars: KAGGLE_USERNAME and KAGGLE_KEY

After running, copy the printed path into config.toml [data] dataset_dir.
kagglehub caches the download — re-running does not re-download.
"""

from dotenv import load_dotenv

load_dotenv()  # loads KAGGLE_USERNAME and KAGGLE_KEY from .env at repo root

import kagglehub


def main():
    print("Downloading PTB-XL dataset via kagglehub (local cache)...")
    path = kagglehub.dataset_download("physionet/ptbxl-electrocardiography-database")
    print(f"\nDataset cached at:\n  {path}")
    print("\nUpdate config.toml [data] section:")
    print(f'  dataset_dir = "{path}"')


if __name__ == "__main__":
    main()
