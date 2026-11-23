import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """1D CNN encoder: (batch, 12, seq_len) -> mu, logvar of shape (batch, latent_dim)."""

    def __init__(self, latent_dim: int, channels: list[int], kernels: list[int], strides: list[int], seq_len: int = 5000):
        super().__init__()
        self._seq_len = seq_len
        in_ch = 12
        layers = []
        all_out = list(channels)
        for i, (out_ch, k, s) in enumerate(zip(all_out, kernels, strides)):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s))
            if i < len(all_out) - 1:
                layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Compute flattened size with a dummy forward pass using actual seq_len
        with torch.no_grad():
            dummy = torch.zeros(1, 12, seq_len)
            flat_size = self.conv(dummy).numel()

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        self._flat_size = flat_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-4.0, 4.0)
        return mu, logvar


class Decoder(nn.Module):
    """1D CNN decoder: (batch, latent_dim) -> (batch, 12, seq_len)."""

    def __init__(self, latent_dim: int, channels: list[int], kernels: list[int], strides: list[int], flat_size: int, seq_len: int = 5000):
        super().__init__()
        # channels/kernels/strides come in encoder order; reverse for decoder
        dec_channels = list(reversed(channels))   # [128, 64, 32]
        dec_kernels = list(reversed(kernels))
        dec_strides = list(reversed(strides))

        self.fc = nn.Linear(latent_dim, flat_size)
        self._reshape_ch = dec_channels[0]
        self._reshape_len = flat_size // dec_channels[0]
        self._seq_len = seq_len

        layers = []
        in_ch = dec_channels[0]
        all_out = dec_channels[1:] + [12]
        for i, (out_ch, k, s) in enumerate(zip(all_out, dec_kernels, dec_strides)):
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=s))
            if i < len(all_out) - 1:
                layers.append(nn.ReLU())
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self._reshape_ch, self._reshape_len)
        h = self.deconv(h)
        # Fix off-by-one from strided convolutions
        h = F.interpolate(h, size=self._seq_len, mode="linear", align_corners=False)
        return h


class VAE(nn.Module):
    """1D CNN Variational Autoencoder for 12-lead ECG anomaly detection.

    Args:
        config: dict loaded from config.toml — uses config["model"] section
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        latent_dim: int = m["latent_dim"]
        channels: list[int] = list(m["encoder_channels"])
        kernels: list[int] = list(m["encoder_kernels"])
        strides: list[int] = list(m["encoder_strides"])
        seq_len: int = config.get("data", {}).get("sampling_rate", 500) * 10  # 500Hz×10s=5000, 100Hz×10s=1000

        self.encoder = Encoder(latent_dim, channels, kernels, strides, seq_len=seq_len)
        self.decoder = Decoder(latent_dim, channels, kernels, strides, self.encoder._flat_size, seq_len=seq_len)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample reconstruction MSE, shape (batch,)."""
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
        # Mean over leads and timesteps per sample
        return F.mse_loss(x_recon, x, reduction="none").mean(dim=(1, 2))
