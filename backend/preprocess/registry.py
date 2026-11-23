"""Dataset registry.

Train/eval scripts call get_spec(config) once after loading config, then use spec.* throughout.
To add a new dataset: create backend/preprocess/<name>.py, add a factory below, register it.

Usage:
    from backend.preprocess.registry import get_spec
    spec = get_spec(config)

    train_records, val_records, test_records = spec.build_splits_fn(config, dataset_dir=dataset_dir)
    norm_stats = spec.fit_norm_stats_fn(train_records, dataset_dir)
    train_ds = spec.dataset_class(train_records, dataset_dir, norm_stats=norm_stats)
    binary_labels = (all_labels != spec.normal_label).astype(int)
    for cls, color in spec.class_colors.items(): ...
"""

from dataclasses import dataclass
from typing import Callable, Type

from torch.utils.data import Dataset


@dataclass
class DatasetSpec:
    name: str
    dataset_class: Type[Dataset]
    build_splits_fn: Callable        # (config, dataset_dir=None) -> (train, val, test)
    fit_norm_stats_fn: Callable      # (records, dataset_dir) -> dict
    normal_label: str                # label string for the normal class
    class_colors: dict[str, str]     # per-class hex colors for eval plots
    seq_len: int                     # samples per record (5000 for 500 Hz)
    sampling_rate: int               # Hz
    kaggle_input_path: str           # mounted path on Kaggle
    required_local_files: list[str]  # probed to validate dataset_dir


def _make_ptbxl_spec() -> DatasetSpec:
    from backend.preprocess.ptbxl import (
        PTBXLDataset, build_splits, fit_norm_stats,
        PTBXL_NORMAL_LABEL, PTBXL_CLASS_COLORS,
        PTBXL_SEQ_LEN, PTBXL_SAMPLING_RATE,
        PTBXL_KAGGLE_INPUT_PATH, PTBXL_REQUIRED_FILES,
    )
    return DatasetSpec(
        name="ptbxl",
        dataset_class=PTBXLDataset,
        build_splits_fn=build_splits,
        fit_norm_stats_fn=fit_norm_stats,
        normal_label=PTBXL_NORMAL_LABEL,
        class_colors=PTBXL_CLASS_COLORS,
        seq_len=PTBXL_SEQ_LEN,
        sampling_rate=PTBXL_SAMPLING_RATE,
        kaggle_input_path=PTBXL_KAGGLE_INPUT_PATH,
        required_local_files=PTBXL_REQUIRED_FILES,
    )


def _make_mimic_spec() -> DatasetSpec:
    from backend.preprocess.mimic import (
        MIMICDataset, build_splits, fit_norm_stats,
        MIMIC_NORMAL_LABEL, MIMIC_CLASS_COLORS,
        MIMIC_SEQ_LEN, MIMIC_SAMPLING_RATE,
        MIMIC_KAGGLE_INPUT_PATH, MIMIC_REQUIRED_FILES,
    )
    return DatasetSpec(
        name="mimic-iv-ecg",
        dataset_class=MIMICDataset,
        build_splits_fn=build_splits,
        fit_norm_stats_fn=fit_norm_stats,
        normal_label=MIMIC_NORMAL_LABEL,
        class_colors=MIMIC_CLASS_COLORS,
        seq_len=MIMIC_SEQ_LEN,
        sampling_rate=MIMIC_SAMPLING_RATE,
        kaggle_input_path=MIMIC_KAGGLE_INPUT_PATH,
        required_local_files=MIMIC_REQUIRED_FILES,
    )


_REGISTRY: dict[str, Callable[[], DatasetSpec]] = {
    "ptbxl":        _make_ptbxl_spec,
    "mimic-iv-ecg": _make_mimic_spec,
}

_cache: dict[str, DatasetSpec] = {}


def get_spec(config: dict) -> DatasetSpec:
    """Return DatasetSpec for the dataset named in config['data']['name'].

    Defaults to 'ptbxl' when the key is absent (backward-compatible with old configs).
    """
    name = config["data"].get("name", "ptbxl")
    if name not in _REGISTRY:
        known = ", ".join(f'"{k}"' for k in _REGISTRY)
        raise KeyError(f"Unknown dataset '{name}' in config.toml [data] name. Known: {known}")
    if name not in _cache:
        _cache[name] = _REGISTRY[name]()
    return _cache[name]
