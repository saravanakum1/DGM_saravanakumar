"""
Resolves the PTB-XL dataset path for both local and Kaggle environments.

- On Kaggle Notebooks: returns /kaggle/input/ecg-ptbxl-data (already mounted, no download)
- Local: returns dataset_dir from config.toml

Usage:
    uv run python src/data/download_dataset.py   # prints resolved path

Import in other scripts:
    from src.data.download_dataset import get_dataset_path
    dataset_dir = get_dataset_path()
"""

from pathlib import Path


KAGGLE_DATA_PATH = "/kaggle/input/datasets/sharav95/ecg-ptbxl-data"


def get_dataset_path() -> str:
    """Return the PTB-XL dataset path for the current environment.

    On Kaggle Notebooks: returns the mounted ecg-ptbxl-data input path.
    Locally: reads dataset_dir from config.toml.
    """
    if Path(KAGGLE_DATA_PATH).exists():
        return KAGGLE_DATA_PATH

    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.toml not found at {config_path}")

    import tomllib
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    path = config["data"]["dataset_dir"]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"dataset_dir '{path}' does not exist. "
            "Update config.toml [data] dataset_dir to point to your local PTB-XL folder."
        )
    return path


def main():
    path = get_dataset_path()
    print(f"PTB-XL dataset path: {path}")
    required = ["ptbxl_database.csv", "scp_statements.csv", "records100"]
    missing = [f for f in required if not (Path(path) / f).exists()]
    if missing:
        print(f"WARNING: missing expected files/dirs: {missing}")
    else:
        print("OK: ptbxl_database.csv, scp_statements.csv, records100/ all present")


if __name__ == "__main__":
    main()
