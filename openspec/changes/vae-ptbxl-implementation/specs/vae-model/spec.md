## ADDED Requirements

### Requirement: Encoder compresses 12-lead ECG to latent distribution
The encoder SHALL accept input of shape `(batch, 12, 1000)` and produce `mu` and `logvar` tensors of shape `(batch, latent_dim)` via three Conv1d layers followed by a Flatten and two parallel Linear projections.

Conv1d layer sequence (configurable via `config.toml` `[model]` section):
- Conv1d(12 → 32, kernel=7, stride=2) + ReLU → (batch, 32, 497)
- Conv1d(32 → 64, kernel=5, stride=2) + ReLU → (batch, 64, 247)
- Conv1d(64 → 128, kernel=3, stride=2) + ReLU → (batch, 128, 123)
- Flatten → (batch, 15744)
- fc_mu: Linear(15744, latent_dim)
- fc_logvar: Linear(15744, latent_dim)

#### Scenario: Encoder produces correct output shapes
- **WHEN** a batch of shape `(8, 12, 1000)` is passed through the encoder
- **THEN** `mu` has shape `(8, 128)` and `logvar` has shape `(8, 128)`

#### Scenario: Encoder output is deterministic given same input and weights
- **WHEN** the same input tensor is passed through the encoder twice with `torch.no_grad()`
- **THEN** both `mu` and `logvar` outputs are identical

---

### Requirement: Reparameterization enables differentiable sampling
The VAE SHALL implement the reparameterization trick: `z = mu + epsilon * exp(0.5 * logvar)` where `epsilon ~ N(0, I)`. During inference (eval mode), the system SHALL use `mu` directly without sampling.

#### Scenario: Reparameterization produces stochastic latent during training
- **WHEN** the model is in training mode and `reparameterize(mu, logvar)` is called twice with the same inputs
- **THEN** the two `z` tensors are not identical (due to random epsilon)

#### Scenario: Reparameterization uses mu directly during eval
- **WHEN** the model is in eval mode and `reparameterize(mu, logvar)` is called
- **THEN** the returned `z` equals `mu` exactly

---

### Requirement: Decoder reconstructs signal of shape (batch, 12, 1000)
The decoder SHALL accept latent vector `z` of shape `(batch, latent_dim)` and produce reconstructed signal of shape `(batch, 12, 1000)` via a Linear layer, reshape, three ConvTranspose1d layers, and a final `F.interpolate` to fix off-by-one from strided convolutions.

Layer sequence:
- Linear(latent_dim, 15744) + ReLU → reshape to (batch, 128, 123)
- ConvTranspose1d(128 → 64, kernel=3, stride=2) + ReLU
- ConvTranspose1d(64 → 32, kernel=5, stride=2) + ReLU
- ConvTranspose1d(32 → 12, kernel=7, stride=2) + Tanh
- F.interpolate(size=1000, mode='linear', align_corners=False)

#### Scenario: Decoder output matches input shape exactly
- **WHEN** a latent tensor of shape `(8, 128)` is passed through the decoder
- **THEN** output shape is `(8, 12, 1000)`

#### Scenario: Decoder output values are in [-1, 1] before interpolation activation
- **WHEN** decoder produces output with Tanh as final activation
- **THEN** all values in the pre-interpolation output are in the range `[-1.0, 1.0]`

---

### Requirement: Forward pass returns reconstruction, mu, and logvar
The VAE `forward(x)` method SHALL return a tuple `(x_recon, mu, logvar)` where `x_recon` has the same shape as `x`.

#### Scenario: Forward pass returns three tensors with correct shapes
- **WHEN** input `x` of shape `(4, 12, 1000)` is passed through `model.forward(x)`
- **THEN** returns tuple `(x_recon, mu, logvar)` with shapes `(4, 12, 1000)`, `(4, 128)`, `(4, 128)`

---

### Requirement: Anomaly score is per-sample reconstruction MSE
The VAE SHALL expose an `anomaly_score(x)` method that returns the mean squared error between `x` and `x_recon`, reduced over all timesteps and leads, producing a scalar per sample — shape `(batch,)`.

#### Scenario: Anomaly score is higher for out-of-distribution input
- **WHEN** the model is trained on NORM signals and anomaly scores are computed for NORM vs MI samples
- **THEN** the mean anomaly score for MI samples exceeds the mean for NORM samples
