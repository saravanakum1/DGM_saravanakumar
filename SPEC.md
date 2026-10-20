# ECG-ADComparison — Project Specification

## 1. Problem Statement

Electrocardiogram (ECG) signals are used to monitor heart activity. Detecting anomalies in ECG is
hard because of:

1. ECG signals vary between patients (inter-patient variability)
2. Abnormal samples are less common than normal samples (class imbalance)
3. ECG recordings may contain noise and missing values
4. Different models capture different signal properties

Different models detect anomalies under different conditions. Users need to implement and test each
model separately, making it difficult to understand which model is most suitable for a particular
ECG anomaly-detection task.

---

## 2. Solution Overview

A Unified Dashboard that:
- Shows model-specific anomaly scores
- Classifies ECG signals into diagnostic categories
- Visualizes suspicious regions in the signal
- Provides evaluation metrics for comparison across models

---

## 3. Dataset

| Property | Value |
|---|---|
| **Dataset** | PTB-XL |
| **Size** | 21,799 records, ~18,869 patients |
| **Signal length** | 10 seconds |
| **Sampling rate** | 100 Hz (downsampled version) |
| **Input shape** | 1000 timesteps × 12 leads |
| **Setup** | Unsupervised — train on NORM only, test on all classes |

### 3.1 Label Schema (5 Superclasses)

| Label | Description | Anomaly? |
|---|---|---|
| NORM | Normal ECG | No (training only) |
| MI | Myocardial Infarction | Yes |
| STTC | ST/T Change | Yes |
| CD | Conduction Disturbance | Yes |
| HYP | Hypertrophy | Yes |

### 3.2 Why PTB-XL

- Largest openly licensed clinical ECG dataset
- 18,869 diverse patients → addresses inter-patient variability
- Real clinical recordings → real noise and missing values
- 12-lead signals → richer signal for generative models
- Predefined train/val/test splits (use fold 10 as test)

---

## 4. Architecture Overview

```
PTB-XL Raw Signal (1000 × 12)
         │
         ▼
┌─────────────────────────────┐
│     SHARED PREPROCESSING    │
│  • Normalize (zero mean,    │
│    unit variance)           │
│  • Bandpass filter          │
│    (0.5 – 40 Hz)            │
│  • Missing value handling   │
│    (interpolation)          │
└─────────────────────────────┘
         │
    ┌────┴──────────────┐
    │         │         │
    ▼         ▼         ▼
  VAE       LSTM      DDPM
    │         │         │
    ▼         ▼         ▼
 recon     predict   denoise
 error     error     error
    │         │         │
    └────┬────┴─────────┘
         │
         ▼
┌─────────────────────────────┐
│         DASHBOARD           │
│  • Anomaly scores (3 models)│
│  • Classification output    │
│  • Suspicious region viz    │
│  • Evaluation metrics       │
└─────────────────────────────┘
```

---

## 5. Shared Preprocessing

| Step | Method | Reason |
|---|---|---|
| Normalization | Zero mean, unit variance per lead | Inter-patient variability |
| Noise removal | Bandpass filter 0.5–40 Hz | Remove baseline wander, muscle artifact |
| Missing values | Linear interpolation or zero-fill | Some leads missing in PTB-XL |
| Train/test split | PTB-XL predefined folds (fold 10 = test) | Reproducibility |

---

## 6. Models

### 6.1 VAE — Variational Autoencoder

- **Type**: Reconstruction-based anomaly detection
- **Training**: NORM signals only
- **Anomaly Score**: Reconstruction error (input vs decoded output) + KL divergence
- **Input**: (1000 × 12)
- **Architecture**: Encoder → Latent space (z) → Decoder
- **Key idea**: Model learns normal ECG distribution. Anomalous signals reconstruct poorly → high error = anomaly

### 6.2 LSTM — Autoregressive Model

- **Type**: Prediction-based anomaly detection
- **Training**: NORM signals only
- **Anomaly Score**: Next-timestep prediction error
- **Input**: Sequence (1000 × 12)
- **Architecture**: Stacked LSTM → Linear output layer
- **Key idea**: Model learns to predict next timestep from past. Unexpected signal patterns produce high prediction error → anomaly

### 6.3 DDPM — Denoising Diffusion Probabilistic Model

- **Type**: Diffusion-based anomaly detection
- **Training**: NORM signals only
- **Anomaly Score**: Reconstruction deviation after denoising
- **Input**: (1000 × 12)
- **Architecture**: Forward diffusion (add noise) → Reverse diffusion (denoise)
- **Key idea**: Model learns to denoise normal ECG. Anomalous signals deviate from learned denoising path → high deviation = anomaly
- **Reference**: "Diffusion-Based Electrocardiography Noise Quantification via Anomaly Detection" — Han et al. (arXiv:2506.11815), adapted for cardiac anomaly detection

### 6.4 Out of Scope (Future Work)

| Model | Reason Parked |
|---|---|
| Score-Based Models | Complexity, time constraints |
| S4 / SSM (Mamba) | Complexity, time constraints |

---

## 7. Dashboard

### 7.1 Features

| Feature | Description |
|---|---|
| Anomaly Score View | Side-by-side scores from VAE, LSTM, DDPM for a given ECG |
| Classification Output | Predicted class (NORM / MI / STTC / CD / HYP) per model |
| Suspicious Region Visualization | Highlight time regions with highest anomaly contribution |
| Evaluation Metrics | Compare models on standard metrics |
| Model Disagreement View | Highlight when models produce conflicting scores |

### 7.2 Evaluation Metrics

| Metric | Purpose |
|---|---|
| AUROC | Overall anomaly detection performance |
| F1 Score | Per-class classification performance |
| Precision / Recall | Anomaly detection trade-off |
| Inference Time | Practical model comparison |

---

## 8. Out of Scope

- Real-time ECG streaming
- Wearable device integration
- Model retraining via dashboard
- Score-Based and SSM models (parked for future)

---

## 9. References

1. PTB-XL Dataset — Wagner et al. 2020, PhysioNet
2. DDPM — Ho et al. 2020, "Denoising Diffusion Probabilistic Models"
3. Diffusion-Based ECG Anomaly Detection — Han et al. 2025, arXiv:2506.11815
