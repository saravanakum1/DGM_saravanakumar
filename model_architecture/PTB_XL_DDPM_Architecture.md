# PTB-XL DDPM Architecture Spec

## Dataset
| Property | Value |
|---|---|
| Dataset | PTB-XL |
| Size | 21,799 records, ~18,869 patients |
| Signal length | 10 seconds |
| Sampling rate | 100 Hz |
| Input shape | 1000 timesteps × 12 leads |
| Setup | Unsupervised — train on NORM only, test on all classes |

> **Data Loading & Preprocessing**: Reuse shared pipeline from VAE spec (Step 0).  
> Output shape into model: `(batch, 12, 1000)` — 12 leads as channels, 1000 timesteps.

---

## Model: DDPM with Noise Parameterization (ε → x₀)

### End-to-End Flow

```
Train
─────
ε ~ N(0, I),  t ~ Uniform(1, T)
x_t  = q_sample(x_0, t, ε)         forward process
ε̂   = UNet(x_t, t)                 noise prediction
loss = noise_loss(ε̂, ε, t)         L_simple or L_vlb

Inference (one reverse step)
─────────────────────────────
ε̂       = UNet_EMA(x_t, t)l
x̂_0     = epsilon_to_x0(x_t, ε̂, t)          ← ε → x̂_0 conversion
μ_θ     = epsilon_to_mean(x_t, ε̂, t)
x_{t−1} = p_sample(μ_θ, t)

Anomaly score
─────────────
score = MSE(x_0, x̂_0)  after partial denoising at t*
```

---

## Noise Schedule

| Parameter | Value |
|---|---|
| Diffusion steps T | 1000 |
| Schedule type | Linear |
| β_start | 1e-4 |
| β_end | 0.02 |
| α_t | 1 − β_t |
| ᾱ_t | ∏ α_s for s=1..t (cumulative product) |
| ᾱ_{t−1} | prepend 1.0 to ᾱ[:-1]  so ᾱ_0 = 1 |
| σ̃²_t | β_t · (1 − ᾱ_{t−1}) / (1 − ᾱ_t)  — posterior variance |

```
q_sample:
    x_t = √ᾱ_t · x_0  +  √(1 − ᾱ_t) · ε,    ε ~ N(0, I)
```

All schedule tensors are precomputed once and registered as model buffers (move with `.to(device)`).

---

## Noise Estimator: 1D UNet

The only learned component. Takes `(x_t, t)` and predicts `ε̂` with the same shape as the input.

### Sinusoidal Time Embedding

```
t  →  SinusoidalPosEmb(dim=128)          half sin, half cos
   →  Linear(128, 512) + SiLU
   →  Linear(512, 512)                   final t_emb shape: (batch, 512)
```

Injected into every ResBlock via `t_proj: Linear(512, out_ch)` — added (not concatenated) after the first conv.

### ResBlock1d Definition

```
ResBlock1d(in_ch, out_ch, t_emb_dim=512):

    h = Conv1d(in_ch→out_ch, k=3, pad=1)( SiLU(GroupNorm(8, in_ch)(x)) )
    h = h + t_proj(temb)[:, :, None]    ← broadcast over L dimension
    h = Conv1d(out_ch→out_ch, k=3, pad=1)( SiLU(GroupNorm(8, out_ch)(h)) )

    residual = Conv1d(in_ch→out_ch, k=1)(x)   if in_ch ≠ out_ch
               Identity(x)                     otherwise

    output = h + residual
```

### SelfAttention1d (bottleneck only)

```
SelfAttention1d(dim=512, heads=8):
    h = GroupNorm(8, dim)(x)
    h = MultiheadAttention(h, h, h)    along L dimension  (L=125 at bottleneck)
    output = x + h                     residual
```

### Encoder (Downsampling Path)

```
Input: x_t  (batch, 12, 1000)

input_conv: Conv1d(12→64, k=3, pad=1)              → (batch, 64, 1000)

enc1a: ResBlock1d(64,  64,  512)                    → (batch,  64, 1000)  ┐ skip s1
enc1b: ResBlock1d(64,  64,  512)                    → (batch,  64, 1000)  ┘
down1: Conv1d(64→128,  k=4, stride=2, pad=1)       → (batch, 128,  500)

enc2a: ResBlock1d(128, 128, 512)                    → (batch, 128,  500)  ┐ skip s2
enc2b: ResBlock1d(128, 128, 512)                    → (batch, 128,  500)  ┘
down2: Conv1d(128→256, k=4, stride=2, pad=1)       → (batch, 256,  250)

enc3a: ResBlock1d(256, 256, 512)                    → (batch, 256,  250)  ┐ skip s3
enc3b: ResBlock1d(256, 256, 512)                    → (batch, 256,  250)  ┘
down3: Conv1d(256→512, k=4, stride=2, pad=1)       → (batch, 512,  125)

enc4a: ResBlock1d(512, 512, 512)                    → (batch, 512,  125)
enc4b: ResBlock1d(512, 512, 512)                    → (batch, 512,  125)
```

### Bottleneck

```
mid1:     ResBlock1d(512, 512, 512)                 → (batch, 512, 125)
mid_attn: SelfAttention1d(512, heads=8)             → (batch, 512, 125)
mid2:     ResBlock1d(512, 512, 512)                 → (batch, 512, 125)
```

### Decoder (Upsampling Path)

Skip connections are concatenated along the channel axis before each ResBlock pair.

```
up3:   ConvTranspose1d(512→512, k=4, stride=2, pad=1)   → (batch, 512, 250)
cat(up3, s3)                                             → (batch, 512+256=768, 250)
dec3a: ResBlock1d(768, 256, 512)                         → (batch, 256, 250)
dec3b: ResBlock1d(256, 256, 512)                         → (batch, 256, 250)

up2:   ConvTranspose1d(256→256, k=4, stride=2, pad=1)   → (batch, 256, 500)
cat(up2, s2)                                             → (batch, 256+128=384, 500)
dec2a: ResBlock1d(384, 128, 512)                         → (batch, 128, 500)
dec2b: ResBlock1d(128, 128, 512)                         → (batch, 128, 500)

up1:   ConvTranspose1d(128→128, k=4, stride=2, pad=1)   → (batch, 128, 1000)
cat(up1, s1)                                             → (batch, 128+64=192, 1000)
dec1a: ResBlock1d(192, 64, 512)                          → (batch,  64, 1000)
dec1b: ResBlock1d( 64, 64, 512)                          → (batch,  64, 1000)
```

> If ConvTranspose1d output length ≠ skip length (off-by-one from stride), align with `F.interpolate` before cat.

### Output Head

```
GroupNorm(8, 64) + SiLU
Conv1d(64→12, k=1)                                       → (batch, 12, 1000)   ε̂
```

No activation — raw unbounded noise prediction.

---

## Noise Parameterization: ε → x̂_0 → μ_θ

The UNet predicts `ε̂`; all downstream quantities are analytical — no extra learned modules.

### Step 1 — epsilon_to_x0

Invert the forward process to recover x̂_0:

```
          x_t  −  √(1 − ᾱ_t) · ε̂_θ(x_t, t)
x̂_0  =  ─────────────────────────────────────
                       √ᾱ_t
```

- With perfect `ε̂ = ε` this exactly recovers `x_0`.
- During inference it produces a per-step estimate used directly as the anomaly score target.

### Step 2 — epsilon_to_mean

Convert noise prediction to the reverse-process mean analytically:

```
                   x_t              (1 − α_t)
μ_θ(x_t, t)  =  ────────  −  ─────────────────────────  · ε̂_θ(x_t, t)
                  √α_t          √α_t · √(1 − ᾱ_t)
```

`α_t = 1 − β_t` — single-step signal retention (NOT cumulative).

### Step 3 — p_sample reverse step

Posterior variance is fixed (not learned):

```
σ̃²_t = β_t · (1 − ᾱ_{t−1}) / (1 − ᾱ_t)

x_{t−1} = μ_θ + σ̃_t · z,    z ~ N(0, I)
         (z = 0 at t = 0 — no noise on the final step)
```

### Precomputed schedule tensors

| Tensor | Formula | Used in |
|---|---|---|
| `betas` | linear β_start..β_end | σ̃²_t numerator |
| `alphas` | 1 − betas | epsilon_to_mean |
| `alpha_bars` | cumprod(alphas) | q_sample, epsilon_to_x0, epsilon_to_mean |
| `alpha_bars_prev` | [1, ᾱ_1, ..., ᾱ_{T-1}] | σ̃²_t computation |
| `posterior_variance` | betas·(1−ᾱ_{t−1})/(1−ᾱ_t) | p_sample reverse step |

---

## Training Objective

```
for each batch x_0:
    ε   ~ N(0, I)
    t   ~ Uniform(1, T)           # exclude t=0: σ̃²_0 = 0, weight undefined
    x_t  = q_sample(x_0, t, ε)
    ε̂   = UNet(x_t, t)
    loss = noise_loss(ε̂, ε, t)
```

### L_simple (default — unweighted MSE)

```
L_simple = mean( ‖ε − ε̂_θ‖² )
```

### L_vlb (full weighted)

```
           1              (1 − α_t)²
w_t  =  ────────  ·  ─────────────────────
        2σ̃²_t          (1 − ᾱ_t) · α_t

L_vlb = mean( w_t · ‖ε − ε̂_θ‖² )
```

Default: `L_simple`. `L_vlb` available via flag (matches exact ELBO derivation; Ho et al. show simple loss works better empirically).

---

## Optimizer & Training

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 2e-4 |
| Epochs | 100 |
| Batch size | 64 |
| Train data | NORM only |
| Val data | NORM held-out (10%) |
| EMA decay | 0.9999 — EMA weights used for inference |
| Gradient clip | max_norm = 1.0 |

- Print train loss every epoch
- Plot loss curve (train loss vs epoch) after training
- Save EMA weights to `artifacts/ddpm_ema_weights.pt`

---

## Anomaly Detection

### Inference: Partial Diffusion + Reconstruction

Full T=1000 reverse steps per sample is slow. Use **partial diffusion** at fixed noise level `t*`:

```
1. Add noise to x_0 at t*:
       t_vec  = [t*−1, ..., t*−1]   shape (B,)
       x_{t*} = q_sample(x_0, t_vec, ε)

2. Reverse-denoise from t* → 0:
       for t in [t*−1, t*−2, ..., 0]:
           ε̂       = UNet_EMA(x_t, t)
           x̂_0     = epsilon_to_x0(x_t, ε̂, t)
           μ_θ     = epsilon_to_mean(x_t, ε̂, t)
           x_{t−1} = p_sample(μ_θ, t)

3. Anomaly score = MSE(x_0, x_{t=0})  per sample  shape (B,)
```

| Parameter | Value |
|---|---|
| Noise level t* | 200 (20% of T=1000) |
| Reverse steps | 200 DDPM steps, t* → 0 |

- NORM signals reconstruct with low MSE — they are on the learned denoising manifold
- Anomalous signals deviate from the denoising trajectory → higher MSE

### Threshold

```
Threshold = 95th percentile of NORM val reconstruction errors
Classify: anomaly if score > threshold
```

### Evaluation Metrics

- AUROC
- Precision-Recall curve (PR-AUC)
- Score distribution plot by class (NORM / MI / STTC / CD / HYP)

---

## Implementation Order

```
1.  Data loading & preprocessing     reuse shared pipeline from VAE spec
2.  make_noise_schedule()            β linear; precompute α_t, ᾱ_t, ᾱ_{t−1}, σ̃²_t
3.  extract() helper                 index schedule tensors at batch t, broadcast to (B, C, L)
4.  q_sample()                       forward: x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε
5.  SinusoidalTimeEmbedding          sin/cos pos-emb → MLP → dim 512
6.  ResBlock1d                       GroupNorm+Conv1d, time-emb addition, residual skip
7.  SelfAttention1d                  MultiHeadAttention along L, GroupNorm + residual
8.  ECGUNet1d                        encoder → bottleneck → decoder, skip cat, output head
9.  epsilon_to_x0()                  x̂_0 = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t
10. epsilon_to_mean()                μ_θ from x_t and ε̂
11. p_sample()                       x_{t−1} = μ_θ + σ̃_t·z
12. noise_loss()                     L_simple default; L_vlb weighted flag
13. DDPM wrapper class               schedule buffers + UNet; exposes loss(), reconstruct(), anomaly_score()
14. Training loop                    100 epochs, EMA update, gradient clip 1.0
15. Plot loss curve
16. Anomaly detection                partial diffusion at t*=200, full reverse, MSE score
17. Evaluation                       AUROC, PR curve, score distribution
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Noise parameterization (predict ε, derive x̂_0 and μ_θ) |  ε̂ → x̂_0 is a free analytical step — no extra network head needed |
| `epsilon_to_x0` as explicit step | x̂_0 available at every reverse step for anomaly scoring and optional clamping `[−1, 1]` to prevent drift |
| μ_θ derived analytically from ε̂ | Closed-form rearrangement — no extra parameters needed |
| σ̃²_t fixed (not learned) | learning variance adds instability for marginal gain on reconstruction tasks |
| L_simple by default | show simple MSE outperforms full ELBO weight empirically; L_vlb kept as a flag |
| t ~ Uniform(1, T), excludes t=0 | σ̃²_0 = 0 → weighted loss undefined at t=0; excluded to avoid NaN |
| 1D UNet (not 2D) | ECG is a 1D temporal signal; 2D conv is the wrong inductive bias |
| GroupNorm (not BatchNorm) | Stable at any batch size; standard in diffusion literature |
| Self-attention at bottleneck only (L=125) | Global context where it is cheapest; full-scale attention at L=1000 would be 64× more expensive |
| EMA weights for inference | Reduces denoising variance; standard DDPM practice |
| Gradient clip max_norm=1.0 | Numerical stabilization during training |
| Partial diffusion at t*=200 | 5× faster than full T=1000 reverse; moderate noise maximises anomaly signal visibility |
