"""Resolves the dataset path for both local and Kaggle environments.

- On Kaggle: returns the mounted Kaggle input path for the configured dataset.
- Local: returns dataset_dir from config.toml.

Usage:
    uv run python src/data/download_dataset.py   # prints resolved path

Import in other scripts:
    from src.data.download_dataset import get_dataset_path
    dataset_dir = get_dataset_path(config)
"""

from pathlib import Path


def get_dataset_path(config: dict | None = None) -> str:
    """Return the dataset directory path for the current environment.

    Args:
        config: pre-loaded config dict. If None, loads config.toml from repo root.
    """
    if config is None:
        import tomllib
        config_path = Path(__file__).resolve().parents[2] / "config.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.toml not found at {config_path}")
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

    from backend.preprocess.registry import get_spec
    spec = get_spec(config)

    if Path(spec.kaggle_input_path).exists():
        return spec.kaggle_input_path

    path = config["data"]["dataset_dir"]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"dataset_dir '{path}' does not exist. "
            f"Update config.toml [data] dataset_dir to point to your local "
            f"{config['data'].get('name', 'ptbxl')} folder."
        )
    return path


def main():
    path = get_dataset_path()
    print(f"Dataset path: {path}")
    import tomllib
    config_path = Path(__file__).resolve().parents[2] / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    from backend.preprocess.registry import get_spec
    spec = get_spec(config)
    missing = [f for f in spec.required_local_files if not (Path(path) / f).exists()]
    if missing:
        print(f"WARNING: missing expected files/dirs: {missing}")
    else:
        print(f"OK: {', '.join(spec.required_local_files)} all present")


if __name__ == "__main__":
    main()
