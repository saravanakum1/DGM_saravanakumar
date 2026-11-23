"""
Shared pytest configuration.

Marks tests in test_fm_train_eval_smoke.py as requiring the real PTB-XL
dataset. Run the full suite with:
    uv run pytest tests/ -v
"""

import os
from pathlib import Path

import pytest

DATASET_DIR = "/Users/rsaravanakumar/Applications/personal/mtech/DGM_saravanakumar/dataset"
_PROBE_RECORD = os.path.join(DATASET_DIR, "records500/00000/00009_hr.dat")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_dataset: test requires the local PTB-XL dataset to be present",
    )


def pytest_collection_modifyitems(config, items):
    dataset_available = Path(_PROBE_RECORD).exists()
    skip_marker = pytest.mark.skip(
        reason=f"PTB-XL dataset not found at {DATASET_DIR}"
    )
    for item in items:
        if "test_fm_train_eval_smoke" in str(item.fspath):
            if not dataset_available:
                item.add_marker(skip_marker)
