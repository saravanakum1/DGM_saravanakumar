## ADDED Requirements

### Requirement: Evaluation entry point is src/eval/eval_vae.py
The system SHALL provide `src/eval/eval_vae.py` as the sole evaluation entry point, runnable via `uv run python src/eval/eval_vae.py`. It SHALL load `config.toml`, load trained weights from `artifacts/vae_weights.pt` and `norm_stats.json`, run inference on the full test set (all classes), and produce evaluation outputs.

#### Scenario: Evaluation script runs without error after training
- **WHEN** `uv run python src/eval/eval_vae.py` is executed after `train_vae.py` has completed
- **THEN** the script exits with code 0 and prints AUROC to stdout

#### Scenario: Evaluation fails with clear message if weights file is missing
- **WHEN** `eval_vae.py` is run before `train_vae.py` (no `vae_weights.pt` present)
- **THEN** a `FileNotFoundError` with a message indicating the missing weights is raised

---

### Requirement: AUROC is computed and printed for binary anomaly detection
The evaluation SHALL compute AUROC treating NORM as negative (0) and all other classes (MI, STTC, CD, HYP) as positive (1). Reconstruction MSE per sample is used as the anomaly score. AUROC SHALL be printed to stdout and saved in `artifacts/metrics.json`.

#### Scenario: AUROC is a float in [0, 1]
- **WHEN** AUROC is computed over the test set
- **THEN** the value is a float between 0.0 and 1.0 inclusive

#### Scenario: AUROC exceeds random baseline
- **WHEN** the VAE has been trained to convergence on NORM data
- **THEN** the computed AUROC exceeds 0.6 on the PTB-XL test set

---

### Requirement: Precision-Recall curve is saved as a plot
The evaluation SHALL generate a Precision-Recall curve using `sklearn.metrics.precision_recall_curve` and save it as `artifacts/vae_pr_curve.png` using matplotlib. The plot SHALL include the PR-AUC value in the title or legend.

#### Scenario: PR curve PNG file is created after evaluation
- **WHEN** `eval_vae.py` completes successfully
- **THEN** `artifacts/vae_pr_curve.png` exists and has non-zero file size

---

### Requirement: Per-class anomaly score distribution is saved as a plot
The evaluation SHALL generate a histogram of reconstruction MSE scores grouped by superclass (NORM, MI, STTC, CD, HYP) and save it as `artifacts/vae_score_distribution.png`. Each class SHALL be plotted as a separate overlapping histogram with a legend.

#### Scenario: Score distribution PNG contains all five classes
- **WHEN** `eval_vae.py` completes and `vae_score_distribution.png` is generated
- **THEN** the plot legend contains entries for NORM, MI, STTC, CD, and HYP

#### Scenario: NORM scores are visually lower than anomaly class scores
- **WHEN** the VAE is trained to reasonable convergence
- **THEN** the mean reconstruction MSE for NORM is lower than the mean for at least three of the four anomaly classes (MI, STTC, CD, HYP)

---

### Requirement: Evaluation metrics are saved to artifacts/metrics.json
The evaluation SHALL save a JSON file at `artifacts/metrics.json` with the structure:
```json
{
  "VAE": {
    "auroc": <float>,
    "pr_auc": <float>,
    "threshold": <float>,
    "threshold_percentile": <int>
  }
}
```

#### Scenario: metrics.json is created with correct keys after evaluation
- **WHEN** `eval_vae.py` completes successfully
- **THEN** `artifacts/metrics.json` exists and contains the `VAE` key with `auroc` and `pr_auc` sub-keys
