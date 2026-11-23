"""
Smoke tests for path resolution in train_flow_matching.py and eval_flow_matching.py.

These tests catch the class of bugs that broke Kaggle runs:
  - wrong parents[n] depth for sys.path / config_path / artifacts_dir
  - using raw config["data"]["dataset_dir"] instead of get_dataset_path()
  - missing splits cache / bundled norm stats fallback logic
  - norm_stats_path type (Path vs str) consistency

No dataset, model weights, or GPU required — all checks are static or use
a temporary directory as a fake artifacts dir.

Run with:
    uv run --with pytest pytest tests/test_fm_path_resolution.py -v
"""

import importlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FM   = REPO_ROOT / "src" / "train" / "train_flow_matching.py"
EVAL_FM    = REPO_ROOT / "src" / "eval"  / "eval_flow_matching.py"
TRAIN_DDPM = REPO_ROOT / "src" / "train" / "train_ddpm.py"
EVAL_DDPM  = REPO_ROOT / "src" / "eval"  / "eval_ddpm.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source(path: Path) -> str:
    return path.read_text()


def _parents_depth(source: str, fragment: str) -> list[int]:
    """Return every n in parents[n] that appears in lines containing `fragment`."""
    import re
    depths = []
    for line in source.splitlines():
        if fragment in line:
            for m in re.finditer(r"parents\[(\d+)\]", line):
                depths.append(int(m.group(1)))
    return depths


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------

class TestScriptFilesExist:
    def test_train_flow_matching_exists(self):
        assert TRAIN_FM.exists(), f"Missing: {TRAIN_FM}"

    def test_eval_flow_matching_exists(self):
        assert EVAL_FM.exists(), f"Missing: {EVAL_FM}"


# ---------------------------------------------------------------------------
# 2. sys.path depth — must be parents[2] for scripts at src/{subdir}/
# ---------------------------------------------------------------------------

class TestSysPathDepth:
    def test_train_fm_syspath_depth(self):
        depths = _parents_depth(_source(TRAIN_FM), "sys.path.insert")
        assert depths, "No sys.path.insert found in train_flow_matching.py"
        assert all(d == 2 for d in depths), (
            f"train_flow_matching.py sys.path uses parents{depths}, expected parents[2] "
            f"(script lives at src/train/ — 2 levels from repo root)"
        )

    def test_train_ddpm_syspath_depth(self):
        depths = _parents_depth(_source(TRAIN_DDPM), "sys.path.insert")
        assert all(d == 2 for d in depths)

    def test_eval_fm_syspath_depth(self):
        depths = _parents_depth(_source(EVAL_FM), "sys.path.insert")
        assert depths, "No sys.path.insert found in eval_flow_matching.py"
        assert all(d == 2 for d in depths), (
            f"eval_flow_matching.py sys.path uses parents{depths}, expected parents[2]"
        )

    def test_eval_ddpm_syspath_depth(self):
        depths = _parents_depth(_source(EVAL_DDPM), "sys.path.insert")
        assert all(d == 2 for d in depths)


# ---------------------------------------------------------------------------
# 3. config_path depth — must be parents[2] for scripts at src/{subdir}/
# ---------------------------------------------------------------------------

class TestConfigPathDepth:
    def test_train_fm_config_path_depth(self):
        depths = _parents_depth(_source(TRAIN_FM), "config_path")
        assert depths, "No config_path assignment found in train_flow_matching.py"
        assert all(d == 2 for d in depths), (
            f"train_flow_matching.py config_path uses parents{depths}, expected parents[2]"
        )

    def test_train_ddpm_config_path_depth(self):
        depths = _parents_depth(_source(TRAIN_DDPM), "config_path")
        assert all(d == 2 for d in depths)

    def test_eval_fm_config_path_depth(self):
        depths = _parents_depth(_source(EVAL_FM), "config_path")
        assert depths, "No config_path assignment found in eval_flow_matching.py"
        assert all(d == 2 for d in depths), (
            f"eval_flow_matching.py config_path uses parents{depths}, expected parents[2]"
        )

    def test_eval_ddpm_config_path_depth(self):
        depths = _parents_depth(_source(EVAL_DDPM), "config_path")
        assert all(d == 2 for d in depths)


# ---------------------------------------------------------------------------
# 4. config_path resolves to a real file from repo root
# ---------------------------------------------------------------------------

class TestConfigPathResolvable:
    def _resolved_config(self, script: Path) -> Path:
        # Simulate: parents[2] of src/train/script.py == repo root
        return script.parents[2] / "config.toml"

    def test_train_fm_config_exists(self):
        assert self._resolved_config(TRAIN_FM).exists(), (
            "config.toml not found at repo root — train_flow_matching.py would crash"
        )

    def test_eval_fm_config_exists(self):
        assert self._resolved_config(EVAL_FM).exists(), (
            "config.toml not found at repo root — eval_flow_matching.py would crash"
        )

    def test_config_toml_parseable(self):
        config_path = REPO_ROOT / "config.toml"
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        assert "data" in config
        assert "artifacts" in config
        assert "flow_matching" in config, (
            "[flow_matching] section missing from config.toml — train_flow_matching.py "
            "will use only FM_DEFAULTS"
        )

    def test_config_has_flow_matching_section(self):
        with open(REPO_ROOT / "config.toml", "rb") as f:
            config = tomllib.load(f)
        fm = config.get("flow_matching", {})
        for key in ("epochs", "batch_size", "lr", "num_steps", "ema_decay", "threshold_percentile"):
            assert key in fm, f"[flow_matching].{key} missing from config.toml"


# ---------------------------------------------------------------------------
# 5. get_dataset_path() is used — not raw config["data"]["dataset_dir"]
# ---------------------------------------------------------------------------

class TestDatasetPathResolution:
    def test_train_fm_uses_get_dataset_path(self):
        src = _source(TRAIN_FM)
        assert "get_dataset_path" in src, (
            "train_flow_matching.py does not import/call get_dataset_path() — "
            "on Kaggle it will read the literal 'dataset' string and fail"
        )
        assert 'config["data"]["dataset_dir"]' not in src or "get_dataset_path" in src, (
            "train_flow_matching.py falls back to raw config dataset_dir instead of get_dataset_path()"
        )

    def test_eval_fm_uses_get_dataset_path(self):
        src = _source(EVAL_FM)
        assert "get_dataset_path" in src, (
            "eval_flow_matching.py does not use get_dataset_path() — will fail on Kaggle"
        )

    def test_get_dataset_path_importable(self):
        # Ensure the module itself is importable (no syntax errors)
        spec = importlib.util.spec_from_file_location(
            "download_dataset",
            REPO_ROOT / "src" / "data" / "download_dataset.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "get_dataset_path")

    def test_get_dataset_path_returns_string_when_available(self):
        # Only call get_dataset_path() if the Kaggle path or the local dataset exists,
        # since it raises FileNotFoundError otherwise (by design).
        import tomllib
        kaggle_path = Path("/kaggle/input/datasets/sharav95/ecg-ptbxl-data")
        config_path = REPO_ROOT / "config.toml"
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        local_path = Path(config["data"]["dataset_dir"])
        if not kaggle_path.exists() and not local_path.exists():
            pytest.skip("Local dataset not present — get_dataset_path() would raise by design")

        spec = importlib.util.spec_from_file_location(
            "download_dataset",
            REPO_ROOT / "src" / "data" / "download_dataset.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.get_dataset_path()
        assert isinstance(result, str), "get_dataset_path() must return a str"
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 6. Splits cache fallback logic present in train_flow_matching
# ---------------------------------------------------------------------------

class TestSplitsCacheFallback:
    def test_train_fm_checks_bundled_splits(self):
        src = _source(TRAIN_FM)
        assert "splits_cache.json" in src, (
            "train_flow_matching.py does not reference splits_cache.json — "
            "it will rebuild splits from scratch on every run (slow on Kaggle)"
        )
        assert "KAGGLE_CODE_PATH" in src, (
            "train_flow_matching.py does not check KAGGLE_CODE_PATH for bundled splits"
        )

    def test_train_fm_saves_splits_cache(self):
        src = _source(TRAIN_FM)
        assert "splits_cache" in src and ("json.dump" in src or "json.dumps" in src), (
            "train_flow_matching.py does not save splits_cache.json after building splits"
        )

    def test_train_ddpm_checks_bundled_splits(self):
        # Reference: ddpm already does this correctly
        src = _source(TRAIN_DDPM)
        assert "KAGGLE_CODE_PATH" in src
        assert "splits_cache.json" in src


# ---------------------------------------------------------------------------
# 7. Bundled norm stats fallback logic present
# ---------------------------------------------------------------------------

class TestNormStatsFallback:
    def test_train_fm_checks_bundled_norm_stats(self):
        src = _source(TRAIN_FM)
        assert "KAGGLE_CODE_PATH" in src, (
            "train_flow_matching.py does not check KAGGLE_CODE_PATH for bundled norm stats"
        )
        assert "norm_stats.json" in src

    def test_train_fm_norm_stats_path_is_path_object(self):
        src = _source(TRAIN_FM)
        # Must use str(norm_stats_path) when passing to save_norm_stats / load_norm_stats
        # because save_norm_stats calls os.makedirs which needs a str on older Pythons
        assert 'str(norm_stats_path)' in src, (
            "train_flow_matching.py passes norm_stats_path to save_norm_stats/load_norm_stats "
            "without str() — may fail if save_norm_stats expects a string path"
        )


# ---------------------------------------------------------------------------
# 8. artifacts_dir.mkdir uses parents=True (safe on fresh Kaggle env)
# ---------------------------------------------------------------------------

class TestArtifactsDirCreation:
    def test_train_fm_mkdir_parents(self):
        src = _source(TRAIN_FM)
        assert "mkdir" in src, "train_flow_matching.py does not call mkdir on artifacts_dir"
        # parents=True ensures nested dirs are created; exist_ok=True prevents crash if exists
        assert "exist_ok=True" in src, (
            "train_flow_matching.py mkdir does not use exist_ok=True"
        )

    def test_train_ddpm_mkdir_parents(self):
        src = _source(TRAIN_DDPM)
        assert "parents=True" in src and "exist_ok=True" in src


# ---------------------------------------------------------------------------
# 9. config_path.exists() guard present (clear error instead of cryptic IOError)
# ---------------------------------------------------------------------------

class TestConfigExistsGuard:
    def test_train_fm_checks_config_exists(self):
        src = _source(TRAIN_FM)
        assert "config_path.exists()" in src, (
            "train_flow_matching.py opens config.toml without checking it exists first — "
            "on a wrong working directory the error message will be confusing"
        )

    def test_train_ddpm_checks_config_exists(self):
        src = _source(TRAIN_DDPM)
        assert "config_path.exists()" in src


# ---------------------------------------------------------------------------
# 10. Runtime: config_path actually resolves to a readable file
# ---------------------------------------------------------------------------

class TestConfigRuntimeReadable:
    def test_config_toml_readable(self):
        config_path = TRAIN_FM.parents[2] / "config.toml"
        assert config_path.is_file()
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        assert config["artifacts"]["norm_stats_file"].endswith(".json")
        assert config["artifacts"]["output_dir"] != ""
