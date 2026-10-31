import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def make_noise_schedule(T: int, beta_start: float, beta_end: float) -> dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "alpha_bars_prev": alpha_bars_prev,
        "posterior_variance": posterior_variance,
    }


def extract(tensor: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Index schedule tensor at batch timesteps and broadcast to signal shape."""
    out = tensor[t]                               # (B,)
    while out.ndim < len(shape):
        out = out.unsqueeze(-1)
    return out.expand(shape)


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------

def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    sqrt_ab = extract(alpha_bars.sqrt(), t, x_0.shape)
    sqrt_one_minus_ab = extract((1.0 - alpha_bars).sqrt(), t, x_0.shape)
    return sqrt_ab * x_0 + sqrt_one_minus_ab * eps


# ---------------------------------------------------------------------------
# Reverse-process helpers (all analytical — no learned parameters)
# ---------------------------------------------------------------------------

def epsilon_to_x0(
    x_t: torch.Tensor,
    eps_hat: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    sqrt_ab = extract(alpha_bars.sqrt(), t, x_t.shape)
    sqrt_one_minus_ab = extract((1.0 - alpha_bars).sqrt(), t, x_t.shape)
    return (x_t - sqrt_one_minus_ab * eps_hat) / sqrt_ab


def epsilon_to_mean(
    x_t: torch.Tensor,
    eps_hat: torch.Tensor,
    t: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    sqrt_alpha = extract(alphas.sqrt(), t, x_t.shape)
    one_minus_alpha = extract(1.0 - alphas, t, x_t.shape)
    sqrt_one_minus_ab = extract((1.0 - alpha_bars).sqrt(), t, x_t.shape)
    return x_t / sqrt_alpha - (one_minus_alpha / (sqrt_alpha * sqrt_one_minus_ab)) * eps_hat


def p_sample(
    mu_theta: torch.Tensor,
    t: torch.Tensor,
    posterior_variance: torch.Tensor,
) -> torch.Tensor:
    z = torch.randn_like(mu_theta)
    sigma = extract(posterior_variance.sqrt(), t, mu_theta.shape)
    # No noise on the final step (t == 0)
    mask = (t > 0).float().view(-1, *([1] * (mu_theta.ndim - 1)))
    return mu_theta + mask * sigma * z


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def noise_loss(
    eps: torch.Tensor,
    eps_hat: torch.Tensor,
    t: torch.Tensor,
    posterior_variance: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    vlb: bool = False,
) -> torch.Tensor:
    if not vlb:
        return F.mse_loss(eps_hat, eps)

    one_minus_alpha = extract(1.0 - alphas, t, eps.shape)
    one_minus_ab = extract(1.0 - alpha_bars, t, eps.shape)
    pv = extract(posterior_variance, t, eps.shape)
    w_t = (1.0 / (2.0 * pv.clamp(min=1e-8))) * (one_minus_alpha ** 2) / (one_minus_ab * extract(alphas, t, eps.shape))
    return (w_t * (eps - eps_hat) ** 2).mean()


# ---------------------------------------------------------------------------
# UNet building blocks
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
    def __init__(self, sin_dim: int = 128, out_dim: int = 512):
        super().__init__()
        self.sin = SinusoidalTimeEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin(t))


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int = 512, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.residual(x)


class SelfAttention1d(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1)        # (B, L, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# 1D UNet
# ---------------------------------------------------------------------------

class ECGUNet1d(nn.Module):
    def __init__(self, t_emb_dim: int = 512):
        super().__init__()
        self.time_emb = TimeEmbeddingMLP(sin_dim=128, out_dim=t_emb_dim)

        # Encoder
        self.input_conv = nn.Conv1d(12, 64, kernel_size=3, padding=1)

        self.enc1a = ResBlock1d(64, 64, t_emb_dim)
        self.enc1b = ResBlock1d(64, 64, t_emb_dim)
        self.down1 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)

        self.enc2a = ResBlock1d(128, 128, t_emb_dim)
        self.enc2b = ResBlock1d(128, 128, t_emb_dim)
        self.down2 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1)

        self.enc3a = ResBlock1d(256, 256, t_emb_dim)
        self.enc3b = ResBlock1d(256, 256, t_emb_dim)
        self.down3 = nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1)

        self.enc4a = ResBlock1d(512, 512, t_emb_dim)
        self.enc4b = ResBlock1d(512, 512, t_emb_dim)

        # Bottleneck
        self.mid1 = ResBlock1d(512, 512, t_emb_dim)
        self.mid_attn = SelfAttention1d(512, heads=8)
        self.mid2 = ResBlock1d(512, 512, t_emb_dim)

        # Decoder (skip cat doubles channels before each pair)
        self.up3 = nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1)
        self.dec3a = ResBlock1d(512 + 256, 256, t_emb_dim)
        self.dec3b = ResBlock1d(256, 256, t_emb_dim)

        self.up2 = nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1)
        self.dec2a = ResBlock1d(256 + 128, 128, t_emb_dim)
        self.dec2b = ResBlock1d(128, 128, t_emb_dim)

        self.up1 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.dec1a = ResBlock1d(128 + 64, 64, t_emb_dim)
        self.dec1b = ResBlock1d(64, 64, t_emb_dim)

        # Output head — raw noise prediction, no activation
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv1d(64, 12, kernel_size=1)

    @staticmethod
    def _align(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.shape[-1] != target_len:
            x = F.interpolate(x, size=target_len, mode="nearest")
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        # Encoder
        x = self.input_conv(x_t)

        s1 = self.enc1b(self.enc1a(x, t_emb), t_emb)
        x = self.down1(s1)

        s2 = self.enc2b(self.enc2a(x, t_emb), t_emb)
        x = self.down2(s2)

        s3 = self.enc3b(self.enc3a(x, t_emb), t_emb)
        x = self.down3(s3)

        x = self.enc4b(self.enc4a(x, t_emb), t_emb)

        # Bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        # Decoder
        x = self._align(self.up3(x), s3.shape[-1])
        x = torch.cat([x, s3], dim=1)
        x = self.dec3b(self.dec3a(x, t_emb), t_emb)

        x = self._align(self.up2(x), s2.shape[-1])
        x = torch.cat([x, s2], dim=1)
        x = self.dec2b(self.dec2a(x, t_emb), t_emb)

        x = self._align(self.up1(x), s1.shape[-1])
        x = torch.cat([x, s1], dim=1)
        x = self.dec1b(self.dec1a(x, t_emb), t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


# ---------------------------------------------------------------------------
# DDPM wrapper
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """DDPM with noise parameterization (ε → x̂_0) for ECG anomaly detection.

    Args:
        config: dict loaded from config.toml — uses config["ddpm"] section
    """

    def __init__(self, config: dict):
        super().__init__()
        d = config["ddpm"]
        self.T: int = d.get("T", 1000)
        self.t_star: int = d.get("t_star", 200)
        self.vlb: bool = d.get("vlb", False)

        sched = make_noise_schedule(self.T, d.get("beta_start", 1e-4), d.get("beta_end", 0.02))
        for name, tensor in sched.items():
            self.register_buffer(name, tensor)

        self.unet = ECGUNet1d(t_emb_dim=512)

    # ------------------------------------------------------------------
    # Core forward (used during training)
    # ------------------------------------------------------------------

    def loss(self, x_0: torch.Tensor) -> torch.Tensor:
        B = x_0.shape[0]
        # t ~ Uniform(1, T) — exclude t=0 (posterior variance is 0)
        t = torch.randint(1, self.T, (B,), device=x_0.device)
        eps = torch.randn_like(x_0)
        x_t = q_sample(x_0, t, eps, self.alpha_bars)
        eps_hat = self.unet(x_t, t)
        return noise_loss(
            eps, eps_hat, t,
            self.posterior_variance, self.alphas, self.alpha_bars,
            vlb=self.vlb,
        )

    # ------------------------------------------------------------------
    # Single reverse step
    # ------------------------------------------------------------------

    def _reverse_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        unet: ECGUNet1d,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps_hat = unet(x_t, t)
        x0_hat = epsilon_to_x0(x_t, eps_hat, t, self.alpha_bars)
        mu = epsilon_to_mean(x_t, eps_hat, t, self.alphas, self.alpha_bars)
        x_prev = p_sample(mu, t, self.posterior_variance)
        return x_prev, x0_hat

    # ------------------------------------------------------------------
    # Anomaly score — partial diffusion at t*
    # ------------------------------------------------------------------

    @torch.no_grad()
    def anomaly_score(
        self,
        x_0: torch.Tensor,
        ema_unet: ECGUNet1d | None = None,
    ) -> torch.Tensor:
        """Return per-sample reconstruction MSE, shape (batch,).

        Args:
            x_0:      Clean ECG batch (batch, 12, 1000).
            ema_unet: EMA copy of the UNet. Falls back to self.unet if None.
        """
        unet = ema_unet if ema_unet is not None else self.unet
        unet.eval()

        B, device = x_0.shape[0], x_0.device

        # 1. Add noise at t*
        t_vec = torch.full((B,), self.t_star - 1, dtype=torch.long, device=device)
        eps = torch.randn_like(x_0)
        x_t = q_sample(x_0, t_vec, eps, self.alpha_bars)

        # 2. Reverse-denoise from t* → 0
        for step in range(self.t_star - 1, -1, -1):
            t_vec = torch.full((B,), step, dtype=torch.long, device=device)
            x_t, _ = self._reverse_step(x_t, t_vec, unet)

        # 3. MSE between original and fully denoised reconstruction
        return F.mse_loss(x_0, x_t, reduction="none").mean(dim=(1, 2))

    # ------------------------------------------------------------------
    # Convenience: full reconstruct (used for visualisation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(
        self,
        x_0: torch.Tensor,
        ema_unet: ECGUNet1d | None = None,
    ) -> torch.Tensor:
        """Reconstruct via partial diffusion. Returns x̂_0 shape (batch, 12, 1000)."""
        unet = ema_unet if ema_unet is not None else self.unet
        unet.eval()

        B, device = x_0.shape[0], x_0.device
        t_vec = torch.full((B,), self.t_star - 1, dtype=torch.long, device=device)
        eps = torch.randn_like(x_0)
        x_t = q_sample(x_0, t_vec, eps, self.alpha_bars)

        for step in range(self.t_star - 1, -1, -1):
            t_vec = torch.full((B,), step, dtype=torch.long, device=device)
            x_t, _ = self._reverse_step(x_t, t_vec, unet)

        return x_t
