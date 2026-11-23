# PTB-XL Flow Matching Architecture Spec

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

## Background: Flow Matching vs DDPM

| Property | DDPM | Flow Matching |
|---|---|---|
| Probability path | Gaussian noise schedule (1000 steps) | Straight line: `x_t = (1-t)·z + t·x` |
| Training target | Predict noise ε | Predict velocity field `v = x - z` |
| Inference steps | 200–1000 (slow) | 50 (RK4) — fewer NFE, same quality |
| Integration error | Cumulative (many steps needed) | 4th-order accurate with RK4 |
| Likelihood | Intractable | **Exact via ODE + instantaneous change of variables** |
| Anomaly score | Partial diffusion MSE (t*=200) | **−log p(x) via ODE likelihood** |

---

## Model: Conditional Flow Matching with 1D U-Net

### Core Idea

Train a velocity field network `v_θ(x_t, t)` on NORM ECGs only.  
At inference, run the **reverse ODE** (x_data → z) while integrating the log-determinant of the flow Jacobian.  
This gives an **exact log-likelihood** `log p(x)` — anomalous ECGs have low likelihood (high `-log p(x)`).

### Probability Path (Straight / Linear)

```
x_t = (1 - t) · z  +  t · x_data        t ∈ [0, 1]

where:
  z       ~ N(0, I)     source distribution (pure noise at t=0)
  x_data  ~ p_data      NORM ECG (target at t=1)
  x_t                   interpolated point along the path

Target velocity (constant along path):
  u_t = dx_t/dt = x_data - z
```

This is the **Conditional Flow Matching (CFM)** objective. The straight path is optimal transport between Gaussian and data — no curved trajectories needed.

---

## Architecture: FlowMatchingUNet1D

### Input / Output
| | Shape | Description |
|---|---|---|
| Input `x_t` | `(B, 12, 1000)` | Noisy ECG at flow time t |
| Input `t` | `(B,)` | Flow time scalar ∈ [0, 1] |
| Output `v` | `(B, 12, 1000)` | Predicted velocity field |

### Time Embedding (identical to notebook)

```
t: (B,) → SinusoidalTimeEmbedding(dim=128) → (B, 128)
         → Linear(128, 512) → SiLU → Linear(512, 128)
         → temb: (B, 128)
```

Sinusoidal embedding uses log-spaced frequencies (same formula as notebook):
```
freq_i = exp(-log(10000) · i / (half_dim - 1))
emb    = [sin(t · freq), cos(t · freq)]     dim=128
```

### ResBlock1D

Each ResBlock conditions on the time embedding via a linear projection:

```
ResBlock1D(in_ch, out_ch, time_dim=128):

  norm1    = GroupNorm(8, in_ch)
  conv1    = Conv1d(in_ch, out_ch, k=3, pad=1)
  time_proj= Linear(128, out_ch)          ← injects t into feature map
  norm2    = GroupNorm(8, out_ch)
  conv2    = Conv1d(out_ch, out_ch, k=3, pad=1)
  residual = Conv1d(in_ch, out_ch, k=1)  ← if in_ch ≠ out_ch, else Identity

forward(x, temb):
  h = SiLU(norm1(x))
  h = conv1(h)
  h = h + time_proj(temb)[:, :, None]    ← broadcast over time axis
  h = SiLU(norm2(h))
  h = conv2(h)
  return h + residual(x)
```

### Downsample1D / Upsample1D

```
Downsample1D(ch):
  Conv1d(ch, ch, k=4, stride=2, pad=1)   → length / 2

Upsample1D(ch):
  interpolate(scale_factor=2, mode='nearest')
  Conv1d(ch, ch, k=3, pad=1)             → length × 2
```

### Full Network Layout

```
Input: (B, 12, 1000)
  │
  ├── input_conv: Conv1d(12, 64, k=3, pad=1)             → (B,  64, 1000)
  │
  ├── ENCODER
  │   ├── enc1: ResBlock1D(64→64) × 2    [skip: e1]      → (B,  64, 1000)
  │   ├── down1: Downsample1D(64)                         → (B,  64,  500)
  │   ├── enc2: ResBlock1D(64→128) × 2   [skip: e2]      → (B, 128,  500)
  │   ├── down2: Downsample1D(128)                        → (B, 128,  250)
  │   ├── enc3: ResBlock1D(128→256) × 2  [skip: e3]      → (B, 256,  250)
  │   └── down3: Downsample1D(256)                        → (B, 256,  125)
  │
  ├── BOTTLENECK
  │   └── mid: ResBlock1D(256→256) × 2                   → (B, 256,  125)
  │
  └── DECODER
      ├── up3:  Upsample1D(256)                           → (B, 256,  250)
      │         cat([up3, e3])                            → (B, 512,  250)
      │         ResBlock1D(512→256) × 2                   → (B, 256,  250)
      │
      ├── up2:  Upsample1D(256)                           → (B, 256,  500)
      │         cat([up2, e2])                            → (B, 384,  500)
      │         ResBlock1D(384→128) × 2                   → (B, 128,  500)
      │
      ├── up1:  Upsample1D(128)                           → (B, 128, 1000)
      │         cat([up1, e1])                            → (B, 192, 1000)
      │         ResBlock1D(192→64) × 2                    → (B,  64, 1000)
      │
      └── output:
              GroupNorm(8, 64) → SiLU
              Conv1d(64, 12, k=3, pad=1)                  → (B,  12, 1000)
                                                          = velocity field v_θ(x_t, t)
```

### Parameter Count (estimate)

| Component | Params |
|---|---|
| Time MLP | ~66K |
| Input conv | ~7K |
| Encoder (3 levels, 2 ResBlocks each) | ~3.5M |
| Bottleneck (2 ResBlocks) | ~1.2M |
| Decoder (3 levels, 2 ResBlocks each) | ~4.8M |
| Output conv | ~7K |
| **Total** | **~9.6M** |

---

## Training

### Loss Function

```
L(θ) = E_{z~N(0,I), x~p_data, t~U[0,1]} [ ‖ v_θ(x_t, t) - (x_data - z) ‖² ]

where:
  x_t = (1-t)·z + t·x_data     (linear interpolation)
  u_t = x_data - z              (constant target velocity)
```

### Training Loop (per batch)

```
1. Sample x_data from NORM train set        shape: (B, 12, 1000)
2. Sample z ~ N(0,I)                        shape: (B, 12, 1000)
3. Sample t ~ Uniform(0, 1)                 shape: (B,)
4. Compute x_t = (1-t)·z + t·x_data
5. Predict v = v_θ(x_t, t)
6. loss = MSE(v, x_data - z)
7. Backprop + AdamW + grad clip (max_norm=1.0)
8. Update EMA model (decay=0.9999)
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Epochs | 100 |
| Grad clip max norm | 1.0 |
| EMA decay | 0.9999 |
| Base channels | 64 |
| Time embedding dim | 128 |

---

## Samplers

### Euler Sampler (fast, training visualization only)

```
x ← z ~ N(0,I)
dt = 1.0 / num_steps

for step in range(num_steps):           # num_steps = 100
    t = step / num_steps
    v = v_θ(x, t)
    x = x + dt · v

return x                                 # generated ECG
```

Used during training to visualize sample quality. **Not used for anomaly scoring.**

### RK4 Sampler (used for generation — forward ODE: z → x)

```
x ← z ~ N(0,I)
dt = 1.0 / num_steps

for step in range(num_steps):           # num_steps = 50
    t = step · dt

    k1 = v_θ(x,           t        )
    k2 = v_θ(x + dt/2·k1, t + dt/2 )
    k3 = v_θ(x + dt/2·k2, t + dt/2 )
    k4 = v_θ(x + dt·k3,   t + dt   )

    x = x + (dt/6) · (k1 + 2·k2 + 2·k3 + k4)

return x                                 # generated ECG
```

**Why RK4 over Euler:** 4th-order accuracy with half the steps (50 vs 100). Less integration error is critical when the trajectory must be numerically reversed for likelihood computation.

---

## Anomaly Scoring via ODE Likelihood

### Mathematical Foundation

A continuous normalizing flow (CNF) defined by the ODE `dx/dt = v_θ(x, t)` gives an **exact log-likelihood** via the instantaneous change of variables formula:

```
log p_data(x) = log p_z(z)  +  ∫₁⁰ div(v_θ(x_t, t)) dt

where:
  x_1 = x_data    (the input ECG, at t=1)
  x_0 = z         (the encoded noise, at t=0)
  div(v_θ)         divergence of the velocity field
  p_z(z)           N(0,I) — easy to evaluate
  p_data(x)        learned data density
```

The **anomaly score** is the **negative log-likelihood**:

```
anomaly_score(x) = -log p_data(x)
                 = -log p_z(z)  -  ∫₁⁰ div(v_θ(x_t, t)) dt
```

High score → low likelihood under the NORM distribution → anomaly.

### Divergence via Hutchinson Trace Estimator

Exact divergence requires `d·d` Jacobian entries (d = 12×1000 = 12000 — intractable).  
Instead, use the **Hutchinson stochastic estimator** (unbiased, O(d) cost):

```
div(v_θ(x, t)) ≈ εᵀ · (∂v_θ/∂x) · ε       ε ~ N(0, I) or Rademacher

Computed via a single vjp (vector-Jacobian product):
  _, vjp_fn = torch.func.vjp(lambda x_: v_θ(x_, t), x)
  div_est   = (vjp_fn(ε)[0] * ε).sum()
```

### Inference Algorithm (Reverse ODE + Log-Likelihood)

```
Given: x_data (input ECG, shape: (1, 12, 1000))

Step 1 — Run reverse ODE  (x_data at t=1  →  z at t=0)
         simultaneously integrate log |det J| via Hutchinson:

  x    ← x_data
  logp ← 0.0
  dt   = -1.0 / num_steps          # negative: integrating t=1 → t=0

  for step in range(num_steps):    # num_steps = 50, using RK4
      t = 1.0 - step / num_steps   # t decrements from 1 to 0

      Sample ε ~ N(0,I)  (same shape as x)

      # RK4 step for x
      k1, div1 = v_and_div(x,           t        , ε)
      k2, div2 = v_and_div(x + dt/2·k1, t + dt/2 , ε)
      k3, div3 = v_and_div(x + dt/2·k2, t + dt/2 , ε)
      k4, div4 = v_and_div(x + dt·k3,   t + dt   , ε)

      x    = x    + (dt/6)·(k1 + 2·k2 + 2·k3 + k4)
      logp = logp + (dt/6)·(div1 + 2·div2 + 2·div3 + div4)

  z = x   # encoded noise at t=0

Step 2 — Evaluate prior log-likelihood:
  log_pz = -0.5 · (z² + log(2π)).sum(dim=[1,2])

Step 3 — Total log-likelihood:
  log_px = log_pz + logp              # logp accumulates -div along reverse path

Step 4 — Anomaly score:
  anomaly_score = -log_px             # higher = more anomalous
```

Where `v_and_div(x, t, ε)` returns the velocity and Hutchinson divergence estimate in one forward + vjp pass.

### Heatmap

The per-timestep anomaly signal comes from the **squared velocity magnitude** at `t=1` (the data boundary), averaged over leads:

```python
with torch.no_grad():
    t1 = torch.ones(1, device=device)
    v_at_data = v_θ(x_data, t1)                       # (1, 12, 1000)
heatmap = v_at_data.pow(2).mean(dim=1)[0].cpu().numpy()  # (1000,)
```

High velocity at a timestep means the model is "pushing hard" to correct that region — consistent with anomalous morphology.

### Threshold

- Fit on NORM validation set: 95th percentile of `-log p(x)` scores
- Stored in `artifacts/norm_stats.json` as `"fm_threshold"`
- Verdict: `anomaly_score > fm_threshold` → **ANOMALY**, else **NORMAL**

---

## EMA (Exponential Moving Average)

Same pattern as notebook — EMA model is used for all inference and sampling:

```
ema_param = decay · ema_param + (1 - decay) · model_param
decay = 0.9999
```

Checkpoint saves both `model` and `ema_model` state dicts.

---

## Artifacts

| File | Description |
|---|---|
| `artifacts/fm_ema_weights.pt` | EMA model weights (used for inference) |
| `artifacts/fm_loss.json` | `{"train": [...], "val": [...]}` per epoch |
| `artifacts/norm_stats.json` | Add key `"fm_threshold"` alongside VAE/DDPM thresholds |

---

## Integration with Dashboard

`backend/inference.py` loads the flow matching model alongside VAE and DDPM:

```python
fm_path = artifacts_dir / "fm_ema_weights.pt"
if fm_path.exists():
    fm = FlowMatchingUNet1D(config)
    fm.load_state_dict(torch.load(fm_path, map_location=device))
    fm.eval()
    models["FM"] = fm
```

`run_inference()` calls `model.anomaly_score(x)` which runs the reverse ODE + Hutchinson likelihood:

```python
if "FM" in models:
    score, heatmap = models["FM"].anomaly_score(x)    # -log p(x), (1000,) heatmap
    results["FM"] = {
        "anomaly_score": float(score),
        "heatmap":       heatmap.cpu().numpy(),
        "inference_ms":  elapsed_ms,
    }
```

Threshold lookup: `norm_stats.get("fm_threshold", 10.0)` — same `.get()` pattern used for DDPM.

---
