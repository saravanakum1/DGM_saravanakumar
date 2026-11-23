# Evaluation

Eval scripts score the test split and produce metrics and plots. They load the split cache written by the corresponding train script, so train must run first.

## Run

=== "DDPM"

    ```bash
    uv run python src/eval/eval_ddpm.py
    ```

    Requires:
    - `artifacts/ddpm_ema_weights.pt`
    - `artifacts/norm_stats.json` (must contain `ddpm_threshold`)

=== "Flow Matching"

    ```bash
    uv run python src/eval/eval_flow_matching.py
    ```

    Requires:
    - `artifacts/fm_ema_weights.pt`
    - `artifacts/norm_stats.json` (must contain `fm_threshold`)

=== "VAE"

    ```bash
    uv run python src/eval/eval_vae.py
    ```

    Requires:
    - `artifacts/vae_weights.pt`
    - `artifacts/norm_stats.json`

## Outputs

Each eval script writes to `artifacts/`:

| File | Description |
|---|---|
| `metrics.json` | AUROC, PR-AUC, threshold, per-class AUROC (merged across models) |
| `<model>_pr_curve.png` | Precision-Recall curve |
| `<model>_score_distribution.png` | Per-class anomaly score histogram |
| `<model>_per_class_auroc.png` | One-vs-NORM AUROC bar chart per class |

Multiple eval scripts can be run in sequence — `metrics.json` is merged, not overwritten.

## Metrics

**AUROC** — area under the ROC curve (binary: NORMAL vs anomaly).

**PR-AUC** — area under the Precision-Recall curve. More informative than AUROC on imbalanced test sets.

**Per-class AUROC** — one-vs-NORM AUROC for each anomaly class. Shows which pathologies each model detects best.

**Anomaly threshold** — set at the `threshold_percentile`-th percentile of NORMAL validation scores. Used for binary classification in the dashboard.

## PTB-XL class colors

| Class | Color |
|---|---|
| NORM | Green |
| MI | Red |
| STTC | Orange |
| CD | Yellow |
| HYP | Purple |

## MIMIC-IV-ECG class colors

| Class | Color |
|---|---|
| NORMAL | Green |
| AF | Red |
| ST | Orange |
| OTHER | Purple |
