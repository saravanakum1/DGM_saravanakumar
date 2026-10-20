import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """1D CNN encoder: (batch, 12, 1000) -> mu, logvar of shape (batch, latent_dim)."""

    def __init__(self, latent_dim: int, channels: list[int], kernels: list[int], strides: list[int]):
        super().__init__()
        in_ch = 12
        layers = []
        for out_ch, k, s in zip(channels, kernels, strides):
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Compute flattened size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 12, 1000)
            flat_size = self.conv(dummy).numel()

        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)
        self._flat_size = flat_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """1D CNN decoder: (batch, latent_dim) -> (batch, 12, 1000)."""

    def __init__(self, latent_dim: int, channels: list[int], kernels: list[int], strides: list[int], flat_size: int):
        super().__init__()
        # channels/kernels/strides come in encoder order; reverse for decoder
        dec_channels = list(reversed(channels))   # [128, 64, 32]
        dec_kernels = list(reversed(kernels))
        dec_strides = list(reversed(strides))

        self.fc = nn.Sequential(nn.Linear(latent_dim, flat_size), nn.ReLU())
        self._reshape_ch = dec_channels[0]
        self._reshape_len = flat_size // dec_channels[0]

        layers = []
        in_ch = dec_channels[0]
        for i, (out_ch, k, s) in enumerate(zip(dec_channels[1:] + [12], dec_kernels, dec_strides)):
            activation = nn.ReLU() if i < len(dec_channels) - 1 else nn.Tanh()
            layers += [nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=s), activation]
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self._reshape_ch, self._reshape_len)
        h = self.deconv(h)
        # Fix off-by-one from strided convolutions
        h = F.interpolate(h, size=1000, mode="linear", align_corners=False)
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

        self.encoder = Encoder(latent_dim, channels, kernels, strides)
        self.decoder = Decoder(latent_dim, channels, kernels, strides, self.encoder._flat_size)
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
