# ECG Anomaly Dashboard — Architecture Proposal

## 1. Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| Frontend + Backend | **Streamlit** | Python-native, no serialization overhead, HF Spaces native support |
| Plotting | **Plotly** | Interactive ECG waveforms, heatmaps, radar charts, histograms |
| Models | **PyTorch** | VAE, LSTM, DDPM loaded directly in Python |
| Deployment | **Hugging Face Spaces** | Free tier, public shareable URL, Streamlit native |

---

## 2. Project Structure

```
config.toml                         ← all hyperparameters, paths, training settings (TOML)
app.py                              ← landing page / entry point
pages/
  01_Upload_Analyze.py              ← main inference page
  02_Model_Comparison.py            ← metrics and model comparison
  03_Training_Diagnostics.py        ← loss curves, model info, latent space
components/
  ecg_viewer.py                     ← ECG waveform + heatmap overlay renderer
  model_card.py                     ← per-model result card (score, class, confidence)
  metrics_table.py                  ← AUROC / F1 / Precision / Recall table
  disagreement_banner.py            ← warning banner when models conflict
backend/
  preprocess/
    __init__.py
    pipeline.py                     ← shared preprocessing: load, filter, normalize, split
  inference.py                      ← model loading and inference runner
  models/
    vae.py                          ← VAE model definition
    lstm.py                         ← LSTM model definition
    ddpm.py                         ← DDPM model definition
  utils/
    heatmap.py                      ← per-timestep anomaly contribution
    scoring.py                      ← score normalization, confidence calculation
src/
  data/
    download_dataset.py             ← kagglehub download + prints resolved path
  train/
    train_vae.py                    ← VAE training entry point (uv run)
    train_lstm.py                   ← LSTM training entry point (future)
    train_ddpm.py                   ← DDPM training entry point (future)
  eval/
    eval_vae.py                     ← VAE evaluation: AUROC, PR curve, score dist
    eval_lstm.py                    ← LSTM evaluation (future)
    eval_ddpm.py                    ← DDPM evaluation (future)
dataset/                            ← PTB-XL downloaded via kagglehub (gitignored)
artifacts/
  vae_weights.pt                    ← trained VAE weights
  lstm_weights.pt                   ← trained LSTM weights
  ddpm_weights.pt                   ← trained DDPM weights
  vae_loss.json                     ← train/val loss history
  lstm_loss.json                    ← train/val loss history
  ddpm_loss.json                    ← train/val loss history
  metrics.json                      ← pre-computed AUROC, F1, Precision, Recall
  norm_stats.json                   ← global mean/std + anomaly threshold from NORM train split
  vae_pr_curve.png                  ← VAE precision-recall curve plot
  vae_score_distribution.png        ← VAE per-class anomaly score histogram
```

---

## 3. Page Architecture

### 3.1 Page 1 — Upload & Analyze

**Purpose:** Primary inference page. User uploads an ECG, selects models, runs analysis.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  ECG Anomaly Detection Dashboard                                │
├──────────────┬──────────────────────────────────────────────────┤
│   SIDEBAR    │                 MAIN AREA                        │
│              │                                                  │
│ ── Input ──  │  [1] ECG WAVEFORM VIEWER                        │
│              │  ┌────────────────────────────────────────────┐  │
│ ○ Upload     │  │ Lead I   ∿∿∿∿∿██████∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿    │  │
│   ECG file   │  │ Lead II  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿    │  │
│ ○ PTB-XL     │  │ Lead III ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿    │  │
│   record ID  │  │  ... 12 leads ...                          │  │
│              │  │  ████ = high anomaly  ░░ = low suspicion   │  │
│ ── Models ── │  └────────────────────────────────────────────┘  │
│ ☑ VAE        │                                                  │
│ ☑ LSTM       │  [2] MODEL RESULT CARDS                         │
│ ☑ DDPM       │  ┌───────────┐ ┌───────────┐ ┌───────────┐     │
│              │  │   VAE     │ │   LSTM    │ │   DDPM    │     │
│ ── Leads ──  │  │ ━━━━━━━━━ │ │ ━━━━━━━━━ │ │ ━━━━━━━━━ │     │
│ ○ All 12     │  │Score: .82 │ │Score: .45 │ │Score: .79 │     │
│ ○ Lead II    │  │[████████] │ │[████░░░░] │ │[███████░] │     │
│ ○ Custom     │  │Class: MI  │ │Class: NORM│ │Class: MI  │     │
│              │  │Conf: 78%  │ │Conf: 61%  │ │Conf: 74%  │     │
│ ── Options ──│  │Time: 42ms │ │Time: 18ms │ │Time:310ms │     │
│ Threshold:   │  └───────────┘ └───────────┘ └───────────┘     │
│ [──0.5──]    │                                                  │
│              │  ⚠ LSTM disagrees — NORM vs MI (VAE + DDPM)    │
│ Preprocess:  │                                                  │
│ ☑ Bandpass   │  [3] ANOMALY HEATMAP (per model, per timestep)  │
│ ☑ Normalize  │  ┌────────────────────────────────────────────┐  │
│              │  │ VAE  heatmap [green → yellow → red]        │  │
│ [ANALYZE]    │  │ LSTM heatmap [green → yellow → red]        │  │
│              │  │ DDPM heatmap [green → yellow → red]        │  │
│              │  └────────────────────────────────────────────┘  │
└──────────────┴──────────────────────────────────────────────────┘
```

#### Sidebar Controls

| Control | Type | Default | Description |
|---|---|---|---|
| Input mode | Radio | Upload file | Upload ECG file or enter PTB-XL record ID |
| File upload | `st.file_uploader` | — | Accepts `.csv`, `.dat`, `.hea` |
| Model selection | `st.multiselect` | All 3 | VAE, LSTM, DDPM |
| Lead selection | Radio + multiselect | All 12 | All 12 / Lead II / Custom |
| Threshold | `st.slider` | 0.5 | Anomaly decision boundary (0.0 – 1.0) |
| Bandpass filter | `st.checkbox` | On | 0.5 – 40 Hz |
| Normalize | `st.checkbox` | On | Global stats from NORM train split |
| Analyze button | `st.button` | — | Triggers inference |

#### Main Area Components

1. **ECG Waveform Viewer** — Plotly multi-line chart, one trace per lead, with colored background regions overlaid from heatmap
2. **Model Result Cards** — `st.columns(3)`, one card per selected model
3. **Disagreement Banner** — `st.warning()` when models disagree on NORMAL vs ANOMALY verdict
4. **Anomaly Heatmap** — Plotly heatmap (timestep × lead), one per model, stacked vertically

---

### 3.2 Page 2 — Model Comparison

**Purpose:** Aggregate view of model performance across the PTB-XL test set. Pre-computed metrics.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Model Comparison                                               │
├─────────────────────────────────────────────────────────────────┤
│  [TAB: Metrics] [TAB: Radar Chart] [TAB: Score Distribution]   │
│                                                                 │
│  ── Metrics Table ──                                            │
│  ┌────────┬───────┬──────┬───────┬────────┬──────┬────────┐   │
│  │ Model  │ AUROC │  F1  │ Prec  │ Recall │  ms  │ Params │   │
│  ├────────┼───────┼──────┼───────┼────────┼──────┼────────┤   │
│  │ VAE    │ 0.91  │ 0.84 │ 0.87  │  0.81  │  42  │  2.1M  │   │
│  │ LSTM   │ 0.76  │ 0.71 │ 0.69  │  0.73  │  18  │  1.4M  │   │
│  │ DDPM   │ 0.89  │ 0.82 │ 0.85  │  0.79  │  310 │  8.3M  │   │
│  └────────┴───────┴──────┴───────┴────────┴──────┴────────┘   │
│                                                                 │
│  ── Radar Chart ──              ── Score Distribution ──       │
│       AUROC                     VAE  [histogram]               │
│      /     \                    LSTM [histogram]               │
│   Speed ── F1                   DDPM [histogram]               │
│      \     /                    (anomaly score spread          │
│    Recall-Prec                   across test set)              │
│                                                                 │
│  ── Per-Class F1 Breakdown ──                                  │
│  Class │ VAE  │ LSTM │ DDPM                                    │
│  MI    │ 0.88 │ 0.72 │ 0.86                                    │
│  STTC  │ 0.81 │ 0.68 │ 0.79                                    │
│  CD    │ 0.84 │ 0.76 │ 0.80                                    │
│  HYP   │ 0.79 │ 0.65 │ 0.77                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### Tabs

| Tab | Content |
|---|---|
| Metrics | AUROC, F1, Precision, Recall, Inference Time, Param Count table |
| Radar Chart | Plotly radar — one polygon per model across 5 axes |
| Score Distribution | Plotly histogram of anomaly scores per model on test set |

#### Guard

```python
if "results" not in st.session_state:
    st.warning("Run analysis on Upload & Analyze page first.")
    st.stop()
```

---

### 3.3 Page 3 — Training Diagnostics

**Purpose:** Show model training evidence. Loss curves, architecture info, latent space visualization.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Training Diagnostics                                           │
├─────────────────────────────────────────────────────────────────┤
│  Select Model: [VAE ▼]                                         │
│                                                                 │
│  ── Loss Curve ──                                              │
│  Loss │╲                                                        │
│       │ ╲__                                                     │
│       │    ╲____                                                │
│       │        ╲_________ train                                 │
│       │          ╲_______ val                                   │
│       └──────────────────────── epoch                          │
│                                                                 │
│  ── Model Info ──                                              │
│  Architecture:  1D CNN Encoder → Latent(128) → Decoder         │
│  Training data: PTB-XL NORM only (9,514 records)               │
│  Epochs:        100   Batch size: 64   LR: 1e-3                │
│  Threshold:     95th percentile of NORM reconstruction error   │
│                                                                 │
│  ── Latent Space (VAE only) ──                                 │
│  t-SNE 2D projection of NORM vs anomaly classes                │
│  [scatter: NORM=blue MI=red STTC=orange CD=yellow HYP=purple]  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Components

| Component | Models | Description |
|---|---|---|
| Loss curve | VAE, LSTM, DDPM | Train vs val loss per epoch — loaded from `artifacts/*_loss.json` |
| Model info card | All | Architecture summary, training hyperparameters, threshold value |
| Latent space | VAE only | t-SNE 2D scatter of test set latent vectors, colored by class |

**Note:** Page 3 has no session state dependency — loss curves and model info are always available from saved artifacts.

---

## 4. Page Navigation

### 4.1 Sidebar Navigation (Auto-generated)

Streamlit auto-generates navigation from the `pages/` folder. No routing code required.

```
┌─────────────────────┐
│  ECG Dashboard      │
│  ───────────────    │
│  📊 Upload & Analyze│  ← highlighted when active
│  📈 Model Comparison│
│  🔬 Training Diag.. │
│                     │
│  ── Page Controls ──│
│  ...                │
└─────────────────────┘
```

### 4.2 Guided Navigation Flow

```
User opens app
      │
      ▼
Page 1: Upload & Analyze
  → upload ECG file
  → select models + options
  → click [ANALYZE]
  → results stored in st.session_state
  → st.success("Analysis complete!")
  → [View Model Comparison →] button appears
      │
      ├──→ Sidebar click: Model Comparison
      │     reads st.session_state["results"]
      │     renders metrics + radar + histograms
      │
      └──→ Sidebar click: Training Diagnostics
            loads pre-saved artifacts (always available)
            no session state dependency
```

### 4.3 Programmatic Navigation (Post-Analysis)

```python
# Page 1 — after ANALYZE completes
if st.session_state.get("results"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Model Comparison →"):
            st.switch_page("pages/02_Model_Comparison.py")
    with col2:
        if st.button("View Training Diagnostics →"):
            st.switch_page("pages/03_Training_Diagnostics.py")
```

### 4.4 Session State Schema

```python
st.session_state = {
    "ecg_signal":       np.ndarray,      # shape (1000, 12) preprocessed signal
    "ecg_raw":          np.ndarray,      # shape (1000, 12) raw signal for display
    "results": {
        "VAE": {
            "anomaly_score":  float,     # 0.0 – 1.0
            "predicted_class": str,      # NORM / MI / STTC / CD / HYP
            "confidence":     float,     # 0.0 – 1.0
            "heatmap":        np.ndarray,# shape (1000, 12) per-timestep scores
            "inference_ms":   int,
        },
        "LSTM": { ... },
        "DDPM": { ... },
    },
    "threshold":        float,           # current slider value
    "selected_models":  list[str],       # e.g. ["VAE", "LSTM", "DDPM"]
    "selected_leads":   list[int],       # e.g. [0, 1, ..., 11]
}
```

---

## 5. Color Coding Scheme

### Waveform Heatmap

| Anomaly Score | Color | Hex |
|---|---|---|
| < 0.3 | White (normal) | `#FFFFFF` |
| 0.3 – 0.5 | Yellow | `#FDE68A` |
| 0.5 – 0.7 | Orange | `#FB923C` |
| > 0.7 | Red | `#EF4444` |

### Score Bar

| Verdict | Color | Hex |
|---|---|---|
| Below threshold (NORMAL) | Green | `#22C55E` |
| Above threshold (ANOMALY) | Red | `#EF4444` |

### Class Badge

| Class | Color | Hex |
|---|---|---|
| NORM | Green | `#22C55E` |
| MI | Red | `#EF4444` |
| STTC | Orange | `#F97316` |
| CD | Yellow | `#EAB308` |
| HYP | Purple | `#A855F7` |

---

## 6. Confidence Score

Confidence is derived from how far the anomaly score is from the decision threshold:

```
confidence = min(abs(score - threshold) / threshold, 1.0) × 100%
```

| score | threshold | confidence | interpretation |
|---|---|---|---|
| 0.82 | 0.5 | 78% | strongly anomalous |
| 0.55 | 0.5 | 10% | borderline |
| 0.10 | 0.5 | 80% | strongly normal |
| 0.45 | 0.5 | 10% | borderline normal |

Displayed as percentage in the model card. Color-coded green for borderline, red for high confidence anomaly.

---

## 7. What to Show — Signal Display Decisions

| Signal | Shown Where | Format |
|---|---|---|
| Anomaly score | Page 1 model card | 0–1 progress bar, colored |
| Confidence | Page 1 model card | Percentage |
| Inference time | Page 1 model card | Milliseconds |
| Predicted class | Page 1 model card | Colored badge |
| Disagreement | Page 1 banner | `st.warning()` |
| AUROC / F1 / Precision / Recall | Page 2 metrics tab | Table + radar chart |
| Score distribution | Page 2 score dist tab | Histogram per model |
| Per-class F1 | Page 2 metrics tab | Table |
| Training loss | Page 3 only | Train vs val line chart |
| Model architecture | Page 3 only | Text summary card |
| Latent space | Page 3, VAE only | t-SNE scatter plot |
| Raw loss at inference | Never | Not interpretable without context |

---

## 8. Deployment — Hugging Face Spaces

| Property | Value |
|---|---|
| Platform | Hugging Face Spaces |
| Runtime | Streamlit |
| Free tier | 2 vCPU, 16 GB RAM |
| GPU | Not required (CPU inference sufficient) |
| DDPM inference | Reduced to 20–50 diffusion steps for demo speed |
| Config file | `README.md` with `sdk: streamlit` header |
| Entry point | `app.py` |
| Model weights | Stored in `artifacts/` or HF Model Hub |

### HF Spaces README Header

```yaml
---
title: ECG Anomaly Detection Dashboard
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---
```
