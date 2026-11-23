# Multi-Dataset Plugin Architecture

## Problem

All preprocessing, label logic, and dataset path resolution is hardcoded for PTB-XL inside `backend/preprocess/pipeline.py` and `src/data/download_dataset.py`. Switching to MIMIC-IV-ECG (or any future dataset) requires editing multiple files across train, eval, and preprocessing layers.

**Goal:** Change one line in `config.toml` to switch datasets. Everything else — preprocessing, label mapping, normal-class definition, Kaggle path resolution — adapts automatically.

---

## Current State (PTB-XL only)

```
config.toml  →  [data] dataset_dir, sampling_rate, val_split

backend/preprocess/pipeline.py
    SUPERCLASS_MAP, TARGET_TIMESTEPS=1000
    load_record()        ← wfdb.rdsamp on every training run
    bandpass_filter()    ← applied at training time, every run
    _load_superclass_lookup(), _extract_superclass()  ← PTB-XL SCP codes
    build_splits()       ← reads ptbxl_database.csv + strat_fold column
    fit_norm_stats()     ← loads WFDB + filters to compute stats
    PTBXLDataset         ← loads WFDB → filter → normalize on __init__

src/data/download_dataset.py
    get_dataset_path()  ← hardcodes sharav95/ecg-ptbxl-data Kaggle path

src/train/train_ddpm.py         ← imports PTBXLDataset, build_splits directly
src/train/train_flow_matching.py
src/train/train_vae.py
src/eval/eval_ddpm.py           ← hardcodes CLASS_COLORS for PTB-XL
src/eval/eval_flow_matching.py
src/eval/eval_vae.py
```

**Problem with current approach:** Raw WFDB loading + bandpass filtering runs on every training start. For large datasets this adds significant startup time and couples the training pipeline to dataset-specific signal processing code.

---

## Target State

### Two-Phase Pipeline

```
PHASE 1 — Offline preprocessing (run once per dataset)
────────────────────────────────────────────────────────
Raw WFDB files
    ↓
process_ptbxl()        bandpass (0.5–40 Hz) + notch (50 Hz) + z-score per lead
process_mimic_direct() bandpass (0.5–40 Hz) + notch (60 Hz) + z-score per lead
    ↓
.npy files  (12, 5000) float32   ← identical format for both datasets
stored at:  dataset/ptbxl/npy/<record_name>.npy
            dataset/mimic-iv-ecg/npy/<study_id>.npy

PHASE 2 — Training (reads .npy, no WFDB, no filtering)
────────────────────────────────────────────────────────
config.toml  [data] name = "ptbxl"   ← only line that changes per dataset
    ↓
get_spec(config)  →  DatasetSpec
    ↓
build_splits_fn()    reads CSV metadata → list of (.npy path, label)
fit_norm_stats_fn()  np.load() .npy files → per-lead mean/std
dataset_class()      np.load() + optional normalize → torch.Tensor
    ↓
DataLoader → model training
```

### Code layout

```
backend/preprocess/
    pipeline.py      ← shared utils: load_npy, normalize, save/load_norm_stats
    ptbxl.py         ← PTB-XL-specific: build_splits, fit_norm_stats, PTBXLDataset, constants
    mimic.py         ← MIMIC-specific: build_splits, fit_norm_stats, MIMICDataset, constants
    registry.py      ← DatasetSpec dataclass + get_spec(config) factory

src/data/download_dataset.py
    get_dataset_path(config)  ← dataset-aware, reads kaggle_input_path from spec

src/train/train_*.py  }  import get_spec(config), use spec.* everywhere
src/eval/eval_*.py    }  no more direct pipeline imports or hardcoded CLASS_COLORS
```

---

## Key Design Decision: No Resampling — Both Datasets at 500 Hz

Both datasets are preprocessed offline at **500 Hz / 5000 samples per lead** before training:

- **PTB-XL**: uses `filename_hr` (`records500/`) — the 500 Hz high-resolution version. Preprocessing applies bandpass (0.5–40 Hz) + notch (50 Hz) + z-score, saves as `(12, 5000)` `.npy`.
- **MIMIC-IV-ECG**: native 500 Hz from `wfdb.rdrecord`. Same preprocessing pipeline, saves as `(12, 5000)` `.npy`.

Both datasets are saved as `(12, 5000)` float32 arrays — **no resampling required**.

### Model Architecture Impact

All three models (DDPM `ECGUNet1d`, Flow Matching `FlowMatchingUNet1D`, VAE) use `Conv1d` layers that are **fully shape-agnostic** — the forward pass works on any sequence length. Feeding `(batch, 12, 5000)` instead of `(batch, 12, 1000)` requires **zero model code changes**.

What changes proportionally at 5000 samples vs 1000:

| Concern | 1000 samples (100 Hz) | 5000 samples (500 Hz) |
|---|---|---|
| Bottleneck seq len (3 downsamples of ÷2) | 125 | 625 |
| Self-attention cost (DDPM bottleneck) | O(125²) | O(625²) — 25× more |
| Memory per batch | baseline | ~5× larger |
| Training time per epoch | baseline | significantly slower |

Config changes: `sampling_rate = 500`, `seq_len = 5000` (used only in the one visualization line in `train_flow_matching.py`).

---

## New Files

### `backend/preprocess/pipeline.py` (simplified shared utils)

Removes all WFDB loading and bandpass filtering — those now live in the offline preprocessing scripts. Keeps only what training needs:

| Function | Purpose |
|---|---|
| `load_npy(path)` | `np.load(path)` → `(12, 5000)` float32 |
| `normalize(signal, stats)` | per-lead mean/std normalization |
| `save_norm_stats(stats, path)` | write JSON |
| `load_norm_stats(path)` | read JSON |
| `_load_one(args)` | `load_npy` + optional `normalize` — picklable for `Pool.map` |

`bandpass_filter`, `load_record` (WFDB), `SUPERCLASS_MAP`, `TARGET_TIMESTEPS` are all removed.

---

### `backend/preprocess/registry.py`

```python
@dataclass
class DatasetSpec:
    name: str                      # "ptbxl" or "mimic-iv-ecg"
    dataset_class: Type[Dataset]   # PTBXLDataset or MIMICDataset
    build_splits_fn: Callable      # build_splits(config, dataset_dir) → (train, val, test)
    fit_norm_stats_fn: Callable    # fit_norm_stats(records, dataset_dir, fs) → dict
    normal_label: str              # "NORM" (ptbxl) | "NORMAL" (mimic)
    class_colors: dict[str, str]   # per-class hex colors for eval plots
    seq_len: int                   # 1000 for both (after MIMIC resample)
    sampling_rate: int             # 100 for both
    kaggle_input_path: str         # mounted Kaggle path for this dataset
    required_local_files: list[str]

def get_spec(config: dict) -> DatasetSpec:
    name = config["data"].get("name", "ptbxl")  # defaults to ptbxl if key absent
    ...
```

`get_spec()` uses lazy factory functions — imports only the needed dataset module, so `import registry` does not pull in `wfdb`, `scipy`, etc. until a spec is actually requested.

### `backend/preprocess/ptbxl.py`

Dataset-level constants:

| Constant | Value |
|---|---|
| `PTBXL_NORMAL_LABEL` | `"NORM"` |
| `PTBXL_CLASS_COLORS` | `{NORM, MI, STTC, CD, HYP}` |
| `PTBXL_SEQ_LEN` | `5000` |
| `PTBXL_SAMPLING_RATE` | `500` |
| `PTBXL_KAGGLE_INPUT_PATH` | `/kaggle/input/datasets/sharav95/ecg-ptbxl-data` |
| `PTBXL_REQUIRED_FILES` | `[ptbxl_database.csv, scp_statements.csv, npy/]` |

`build_splits` reads `ptbxl_database.csv`:
- filters `diagnostic_superclass` contains `"NORM"` for train/val
- uses `strat_fold == 10` for test set (all superclasses)
- returns list of `(npy_path, label)` tuples pointing to pre-saved `.npy` files

`PTBXLDataset.__init__` calls `_load_one(npy_path, norm_stats)` — just `np.load` + normalize, no WFDB.

### `backend/preprocess/mimic.py`

| Concern | PTB-XL | MIMIC-IV-ECG |
|---|---|---|
| Offline preprocessing | bandpass + notch (50 Hz) + z-score | bandpass + notch (60 Hz) + z-score |
| .npy shape | `(12, 5000)` float32 | `(12, 5000)` float32 |
| Label source | `scp_codes` + `scp_statements.csv` | `report_0` + `report_2` columns |
| Normal filter | `diagnostic_superclass` contains `"NORM"` | `report_0 == "Sinus rhythm"` AND `report_2 == "Normal ECG"` |
| Normal label | `"NORM"` | `"NORMAL"` |
| Test split strategy | `strat_fold == 10` (built-in) | last 10% of subjects by `subject_id` |
| Anomaly classes | MI, STTC, CD, HYP | AF, ST, OTHER |

`build_splits` reads `machine_measurements.csv`, filters by `report_0`/`report_2` for normal class, splits by `subject_id` to avoid patient leakage, returns `(npy_path, label)` tuples.

`MIMICDataset.__init__` is identical in structure to `PTBXLDataset` — both just call `np.load` + normalize.

---

## Modified Files

### `backend/preprocess/pipeline.py`

**Removes entirely:** `load_record` (WFDB), `bandpass_filter`, `SUPERCLASS_MAP`, `TARGET_TIMESTEPS`, `_load_superclass_lookup`, `_extract_superclass`, `build_splits`, `fit_norm_stats`, `PTBXLDataset`

**Keeps / replaces with:**
- `load_npy(path)` → `np.load(path)` returning `(12, 5000)` float32
- `normalize`, `save_norm_stats`, `load_norm_stats` — unchanged
- `_load_one(npy_path, norm_stats)` — simplified: no WFDB, no filtering

**Adds at bottom** (backward-compat re-exports so existing tests don't break):
```python
from backend.preprocess.ptbxl import PTBXLDataset, build_splits, SUPERCLASS_MAP  # noqa: F401
```

### `config.toml`

Add one line to `[data]`, and update `sampling_rate`:
```toml
[data]
name = "ptbxl"          # ← NEW. Switch to "mimic-iv-ecg" to use MIMIC
dataset_dir = "dataset"
sampling_rate = 500     # ← updated from 100 (both datasets are 500 Hz)
val_split = 0.1
```

### `src/data/download_dataset.py`

`get_dataset_path()` → `get_dataset_path(config=None)` (backward-compatible: if `config=None`, loads config internally).

Before checking local `dataset_dir`, checks `spec.kaggle_input_path` — automatically uses the right Kaggle dataset mount for whichever dataset is configured.

### Train scripts (`train_ddpm.py`, `train_flow_matching.py`, `train_vae.py`)

**Import change:**
```python
# Before
from backend.preprocess.pipeline import PTBXLDataset, build_splits, fit_norm_stats, ...

# After
from backend.preprocess.registry import get_spec
from backend.preprocess.pipeline import load_norm_stats, save_norm_stats
```

**After config load, add:**
```python
spec = get_spec(config)
dataset_dir = get_dataset_path(config)
```

**Call-site replacements:**

| Before | After |
|---|---|
| `build_splits(config, ...)` | `spec.build_splits_fn(config, ...)` |
| `fit_norm_stats(records, dir, fs=fs)` | `spec.fit_norm_stats_fn(records, dir, fs=fs)` |
| `PTBXLDataset(records, dir, ...)` | `spec.dataset_class(records, dir, ...)` |

`train_flow_matching.py` line 245 also changes:
```python
# Before
z = torch.randn(fm_cfg["val_vis_samples"], 12, 1000, device=device)
# After
z = torch.randn(fm_cfg["val_vis_samples"], 12, spec.seq_len, device=device)
# spec.seq_len = 5000 for both datasets
```

### Eval scripts (`eval_ddpm.py`, `eval_flow_matching.py`, `eval_vae.py`)

Same import swap as train scripts, plus:

- Remove the local `CLASS_COLORS` dict (defined identically in all three files)
- Replace all uses of `CLASS_COLORS` with `spec.class_colors`
- Replace `(all_labels != "NORM")` with `(all_labels != spec.normal_label)`

`eval_flow_matching.py` has a `_load_splits()` helper that calls `build_splits` directly — change signature to accept `spec` and use `spec.build_splits_fn`.

---

## File Change Summary

| File | Change type |
|---|---|
| `config.toml` | Add `name = "ptbxl"`, update `sampling_rate = 500` |
| `backend/preprocess/registry.py` | **NEW** — `DatasetSpec` + `get_spec()` |
| `backend/preprocess/ptbxl.py` | **NEW** — PTB-XL splits/dataset/constants |
| `backend/preprocess/mimic.py` | **NEW** — MIMIC splits/dataset/constants |
| `backend/preprocess/pipeline.py` | **REWRITTEN** — remove WFDB/filter code; keep `load_npy`, `normalize`, `save/load_norm_stats`, `_load_one`; add backward-compat re-exports |
| `src/data/download_dataset.py` | `config=None` param; dataset-aware Kaggle path via `spec` |
| `src/train/train_ddpm.py` | Import swap + `spec.*` call-sites |
| `src/train/train_flow_matching.py` | Import swap + `spec.*` + fix `seq_len=1000` → `spec.seq_len` |
| `src/train/train_vae.py` | Import swap + `spec.*` call-sites |
| `src/eval/eval_ddpm.py` | Import swap + `spec.*` + remove hardcoded `CLASS_COLORS` |
| `src/eval/eval_flow_matching.py` | Import swap + `spec.*` + remove hardcoded `CLASS_COLORS` |
| `src/eval/eval_vae.py` | Import swap + `spec.*` + remove hardcoded `CLASS_COLORS` |

**No changes to:** `backend/models/ddpm.py`, `backend/models/flow_matching.py`, `backend/models/vae.py`, `tests/` — Conv1d layers are shape-agnostic; backward-compat re-exports keep tests passing

---

## Adding a Third Dataset Later

1. Write an offline preprocessing script (like `process_ptbxl` / `process_mimic_direct`) — bandpass + notch + z-score → saves `(12, 5000)` float32 `.npy` files
2. Create `backend/preprocess/<name>.py` with `build_splits`, `fit_norm_stats`, a `Dataset` subclass, and the constant block (`*_NORMAL_LABEL`, `*_CLASS_COLORS`, `*_SEQ_LEN`, `*_SAMPLING_RATE`, `*_KAGGLE_INPUT_PATH`, `*_REQUIRED_FILES`)
3. Add a `_make_<name>_spec()` factory in `registry.py`
4. Register it in `_REGISTRY_FACTORIES`
5. Set `name = "<name>"` in `config.toml`

No train/eval script changes required. No model architecture changes required.
