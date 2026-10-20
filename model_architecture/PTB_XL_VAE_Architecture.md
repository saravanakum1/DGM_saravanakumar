# PTB-XL VAE Architecture Spec

## Dataset
| Property | Value |
|---|---|
| Dataset | PTB-XL |
| Size | 21,799 records, ~18,869 patients |
| Signal length | 10 seconds |
| Sampling rate | 100 Hz |
| Input shape | 1000 timesteps × 12 leads |
| Setup | Unsupervised — train on NORM only, test on all classes |

---

## Step 0: Data Loading & Preprocessing (Shared / Reusable)

> This section is model-agnostic. Any future model (GAN, diffusion, etc.) should reuse this.

- Load PTB-XL records using `wfdb` or `ptb-xl` loader
- Downsample to 100 Hz (1000 timesteps)
- Normalize each lead to zero mean, unit variance (per-sample or global stats from NORM train split)
- Filter NORM records for training
- Split NORM into train / val (e.g., 90/10)
- Keep all classes (NORM + non-NORM) for test set
- Output shape: `(batch, 12, 1000)` — 12 leads as channels, 1000 timesteps

---

## Model: 1D CNN Variational Autoencoder

### Encoder
```
Input:  (batch, 12, 1000)

Conv1d(in=12,  out=32,  kernel=7, stride=2) + ReLU  → (batch, 32,  497)
Conv1d(in=32,  out=64,  kernel=5, stride=2) + ReLU  → (batch, 64,  247)
Conv1d(in=64,  out=128, kernel=3, stride=2) + ReLU  → (batch, 128, 123)

Flatten → (batch, 15744)

fc_mu    : Linear(15744, 128)
fc_logvar: Linear(15744, 128)
```

### Reparameterization
```
z = mu + epsilon * exp(0.5 * logvar)
where epsilon ~ N(0, I)
```

### Decoder
```
Input: z  (batch, 128)

Linear(128, 15744) + ReLU → (batch, 15744)
Reshape             → (batch, 128, 123)

ConvTranspose1d(in=128, out=64, kernel=3, stride=2) + ReLU
ConvTranspose1d(in=64,  out=32, kernel=5, stride=2) + ReLU
ConvTranspose1d(in=32,  out=12, kernel=7, stride=2) + Tanh

Interpolate (linear) → (batch, 12, 1000)   # fix off-by-one from strided convs
```

### Forward Pass
```
encode(x)            → mu, logvar
reparameterize(...)  → z
decode(z)            → x_recon
return x_recon, mu, logvar
```

---

## VAE Loss

```
recon_loss = MSE(x_recon, x)                            # mean reduced, continuous signal
kl_loss    = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
total_loss = recon_loss + beta * kl_loss
```

| Parameter | Value |
|---|---|
| Loss type | MSE (not BCE — ECG is continuous) |
| beta | 1 |
| Latent dim | 128 |

---

## Optimizer & Training

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | 50 |
| Batch size | 64 |
| Train data | NORM only |
| Val data | NORM held-out (10%) |

- Print train loss every epoch
- Plot loss curve (train loss vs epoch) after training

---

## Anomaly Detection

- Compute **reconstruction MSE** per sample on the full test set (all classes)
- Threshold: 95th or 99th percentile of NORM val reconstruction errors
- Classify: sample is anomaly if recon error > threshold

### Evaluation Metrics
- AUROC
- Precision-Recall curve
- Score distribution plot by class (NORM vs MI vs STTC vs CD vs HYP)

---

## Implementation Order

```
1. Data loading & preprocessing     (shared, reusable)
2. VAE class definition             (encoder, decoder, forward, reparameterize)
3. Loss function                    (MSE recon + KL)
4. Training loop                    (50 epochs, print loss per epoch)
5. Plot loss curve
6. Anomaly detection                (recon error → threshold → classify)
7. Evaluation                       (AUROC, PR curve, score distribution)
```
