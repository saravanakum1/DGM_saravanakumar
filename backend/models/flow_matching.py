import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbeddingMLP(nn.Module):
    def __init__(self, sin_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.sin = SinusoidalTimeEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin(t))


# ---------------------------------------------------------------------------
# ResBlock1D
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = 128, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(temb)[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.residual(x)


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------

class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if target_len is not None and x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# FlowMatchingUNet1D
# ---------------------------------------------------------------------------

class FlowMatchingUNet1D(nn.Module):
    """1D U-Net velocity field for Conditional Flow Matching on ECG (12 leads × seq_len).

    Input:  x_t (B, 12, seq_len) noisy ECG at flow time t
            t   (B,)             flow time ∈ [0, 1]
    Output: v   (B, 12, seq_len) predicted velocity field
    """

    def __init__(self, time_dim: int = 128):
        super().__init__()
        self.time_emb = TimeEmbeddingMLP(sin_dim=time_dim, out_dim=time_dim)

        # Encoder
        self.input_conv = nn.Conv1d(12, 64, kernel_size=3, padding=1)

        self.enc1a = ResBlock1D(64, 64, time_dim)
        self.enc1b = ResBlock1D(64, 64, time_dim)
        self.down1 = Downsample1D(64)

        self.enc2a = ResBlock1D(64, 128, time_dim)
        self.enc2b = ResBlock1D(128, 128, time_dim)
        self.down2 = Downsample1D(128)

        self.enc3a = ResBlock1D(128, 256, time_dim)
        self.enc3b = ResBlock1D(256, 256, time_dim)
        self.down3 = Downsample1D(256)

        # Bottleneck
        self.mid1 = ResBlock1D(256, 256, time_dim)
        self.mid2 = ResBlock1D(256, 256, time_dim)

        # Decoder
        self.up3 = Upsample1D(256)
        self.dec3a = ResBlock1D(512, 256, time_dim)
        self.dec3b = ResBlock1D(256, 256, time_dim)

        self.up2 = Upsample1D(256)
        self.dec2a = ResBlock1D(384, 128, time_dim)
        self.dec2b = ResBlock1D(128, 128, time_dim)

        self.up1 = Upsample1D(128)
        self.dec1a = ResBlock1D(192, 64, time_dim)
        self.dec1b = ResBlock1D(64, 64, time_dim)

        # Output head
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv1d(64, 12, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_emb(t)

        x = self.input_conv(x_t)

        e1 = self.enc1b(self.enc1a(x, temb), temb)       # (B, 64, 1000)
        x = self.down1(e1)                                 # (B, 64, 500)

        e2 = self.enc2b(self.enc2a(x, temb), temb)        # (B, 128, 500)
        x = self.down2(e2)                                 # (B, 128, 250)

        e3 = self.enc3b(self.enc3a(x, temb), temb)        # (B, 256, 250)
        x = self.down3(e3)                                 # (B, 256, 125)

        x = self.mid2(self.mid1(x, temb), temb)           # (B, 256, 125)

        x = self.up3(x, e3.shape[-1])
        x = torch.cat([x, e3], dim=1)                     # (B, 512, 250)
        x = self.dec3b(self.dec3a(x, temb), temb)         # (B, 256, 250)

        x = self.up2(x, e2.shape[-1])
        x = torch.cat([x, e2], dim=1)                     # (B, 384, 500)
        x = self.dec2b(self.dec2a(x, temb), temb)         # (B, 128, 500)

        x = self.up1(x, e1.shape[-1])
        x = torch.cat([x, e1], dim=1)                     # (B, 192, 1000)
        x = self.dec1b(self.dec1a(x, temb), temb)         # (B, 64, 1000)

        return self.out_conv(F.silu(self.out_norm(x)))    # (B, 12, 1000)


# ---------------------------------------------------------------------------
# Flow Matching training helpers
# ---------------------------------------------------------------------------

def cfm_loss(model: FlowMatchingUNet1D, x_data: torch.Tensor) -> torch.Tensor:
    """Conditional Flow Matching training loss (MSE on velocity field)."""
    B, device = x_data.shape[0], x_data.device
    z = torch.randn_like(x_data)
    t = torch.rand(B, device=device)
    t_expand = t[:, None, None]
    x_t = (1 - t_expand) * z + t_expand * x_data
    target = x_data - z
    pred = model(x_t, t)
    return F.mse_loss(pred, target)


@torch.no_grad()
def cfm_score(
    model: FlowMatchingUNet1D,
    x_data: torch.Tensor,
    n_samples: int = 20,
) -> torch.Tensor:
    """Per-sample CFM reconstruction score for anomaly detection.

    Averages MSE(v_θ(x_t, t), x - z) over n_samples draws of (t, z).
    NORM signals score low (model knows them), anomalies score high.

    Returns shape (B,) — higher = more anomalous.
    """
    B, device = x_data.shape[0], x_data.device
    scores = torch.zeros(B, device=device)
    for _ in range(n_samples):
        z = torch.randn_like(x_data)
        t = torch.rand(B, device=device)
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * z + t_expand * x_data
        target = x_data - z
        pred = model(x_t, t)
        # MSE per sample: mean over leads and timesteps
        mse = (pred - target).pow(2).mean(dim=[1, 2])
        scores = scores + mse
    return scores / n_samples


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_euler(
    model: FlowMatchingUNet1D,
    z: torch.Tensor,
    num_steps: int = 100,
) -> torch.Tensor:
    """Euler sampler for training visualization (z → x_data). Not used for scoring."""
    x = z.clone()
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((x.shape[0],), step * dt, device=x.device)
        v = model(x, t)
        x = x + dt * v
    return x


@torch.no_grad()
def sample_rk4(
    model: FlowMatchingUNet1D,
    z: torch.Tensor,
    num_steps: int = 50,
) -> torch.Tensor:
    """RK4 sampler for generation (z → x_data)."""
    x = z.clone()
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_val = step * dt
        t = torch.full((x.shape[0],), t_val, device=x.device)
        k1 = model(x, t)
        k2 = model(x + dt / 2 * k1, torch.full_like(t, t_val + dt / 2))
        k3 = model(x + dt / 2 * k2, torch.full_like(t, t_val + dt / 2))
        k4 = model(x + dt * k3,     torch.full_like(t, t_val + dt))
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ---------------------------------------------------------------------------
# Anomaly scoring via ODE likelihood
# ---------------------------------------------------------------------------

def _velocity_and_div(
    model: FlowMatchingUNet1D,
    x: torch.Tensor,
    t_val: float,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (velocity, Hutchinson divergence estimate) with grad enabled."""
    x_ = x.detach().requires_grad_(True)
    t_vec = torch.full((x_.shape[0],), t_val, device=x_.device)
    v = model(x_, t_vec)
    # vjp: computes εᵀ · (∂v/∂x) in one backward pass
    vjp_val = torch.autograd.grad(
        outputs=v,
        inputs=x_,
        grad_outputs=eps,
        create_graph=False,
        retain_graph=False,
    )[0]
    div_est = (vjp_val * eps).sum(dim=[1, 2])   # (B,) — per-sample trace estimate
    return v.detach(), div_est.detach()


def ode_log_likelihood(
    model: FlowMatchingUNet1D,
    x_data: torch.Tensor,
    num_steps: int = 50,
    n_hutchinson: int = 1,
) -> torch.Tensor:
    """Compute log p(x_data) via reverse ODE + Hutchinson trace estimator.

    Runs the reverse ODE x_data (t=1) → z (t=0) using RK4 while accumulating
    ∫ div(v_θ) dt via Hutchinson's stochastic estimator.

    Returns log_px per sample, shape (B,).
    """
    B, device = x_data.shape[0], x_data.device
    x = x_data.clone()
    logp = torch.zeros(B, device=device)
    dt = -1.0 / num_steps

    for step in range(num_steps):
        t_val = 1.0 - step / num_steps

        # Average over multiple Hutchinson probes for lower variance
        div_accum = torch.zeros(B, device=device)
        v_accum = None
        for _ in range(n_hutchinson):
            eps = torch.randn_like(x)

            with torch.enable_grad():
                v1, d1 = _velocity_and_div(model, x, t_val, eps)
                v2, d2 = _velocity_and_div(model, x + dt / 2 * v1, t_val + dt / 2, eps)
                v3, d3 = _velocity_and_div(model, x + dt / 2 * v2, t_val + dt / 2, eps)
                v4, d4 = _velocity_and_div(model, x + dt * v3, t_val + dt, eps)

            div_accum = div_accum - (dt / 6) * (d1 + 2 * d2 + 2 * d3 + d4)  # -div(v): instantaneous log-density change
            if v_accum is None:
                v_accum = (dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)

        x = x + v_accum
        logp = logp + div_accum / n_hutchinson

    z = x
    # log N(0,I): sum over leads and timesteps, then mean handled at call site
    log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=[1, 2])
    return log_pz + logp


# ---------------------------------------------------------------------------
# FlowMatching wrapper (inference API matching DDPM)
# ---------------------------------------------------------------------------

class FlowMatching(nn.Module):
    """Flow Matching wrapper for ECG anomaly detection.

    Exposes anomaly_score() and generate() to match the DDPM interface
    used in backend/inference.py.
    """

    def __init__(self, time_dim: int = 128, num_steps: int = 50, n_hutchinson: int = 1, seq_len: int = 5000):
        super().__init__()
        self.num_steps = num_steps
        self.n_hutchinson = n_hutchinson
        self.seq_len = seq_len
        self.unet = FlowMatchingUNet1D(time_dim=time_dim)

    def loss(self, x_data: torch.Tensor) -> torch.Tensor:
        return cfm_loss(self.unet, x_data)

    def anomaly_score(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (score_per_sample, heatmap).

        score  : CFM reconstruction error per sample, shape (B,). Higher = more anomalous.
        heatmap: squared velocity magnitude at t=0.5 averaged over leads, shape (1000,).
        """
        self.unet.eval()

        # Heatmap: velocity magnitude at t=0.5 (midpoint of flow)
        with torch.no_grad():
            t_mid = torch.full((x.shape[0],), 0.5, device=x.device)
            v_mid = self.unet(x, t_mid)
        heatmap = v_mid.pow(2).mean(dim=1)[0].cpu()   # (1000,)

        # CFM reconstruction score
        score = cfm_score(self.unet, x, n_samples=self.n_hutchinson * 10)

        return score, heatmap

    @torch.no_grad()
    def generate(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Generate ECG samples via RK4 ODE (z → x_data)."""
        z = torch.randn(n_samples, 12, self.seq_len, device=device)
        return sample_rk4(self.unet, z, num_steps=self.num_steps)
