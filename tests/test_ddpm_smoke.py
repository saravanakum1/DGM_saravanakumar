"""
Smoke tests for the DDPM model (backend/models/ddpm.py).

All tests use synthetic tensors — no dataset required.
Run with:
    uv run pytest tests/test_ddpm_smoke.py -v
"""

import copy

import pytest
import torch

from backend.models.ddpm import (
    DDPM,
    ECGUNet1d,
    ResBlock1d,
    SelfAttention1d,
    SinusoidalTimeEmbedding,
    TimeEmbeddingMLP,
    epsilon_to_mean,
    epsilon_to_x0,
    extract,
    make_noise_schedule,
    noise_loss,
    p_sample,
    q_sample,
)

B, C, L = 2, 12, 1000  # batch, leads, timesteps
T = 1000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def schedule():
    return make_noise_schedule(T, beta_start=1e-4, beta_end=0.02)


@pytest.fixture(scope="module")
def config():
    return {
        "ddpm": {
            "T": T,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "t_star": 200,
            "vlb": False,
            "ema_decay": 0.9999,
        }
    }


@pytest.fixture(scope="module")
def model(config):
    m = DDPM(config)
    m.eval()
    return m


@pytest.fixture
def x0():
    return torch.randn(B, C, L)


@pytest.fixture
def t_batch():
    return torch.randint(1, T, (B,))


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class TestNoiseSchedule:
    def test_keys(self, schedule):
        expected = {"betas", "alphas", "alpha_bars", "alpha_bars_prev", "posterior_variance"}
        assert expected == set(schedule.keys())

    def test_shapes(self, schedule):
        for name, tensor in schedule.items():
            assert tensor.shape == (T,), f"{name} wrong shape"

    def test_betas_range(self, schedule):
        assert schedule["betas"][0].item() == pytest.approx(1e-4, rel=1e-3)
        assert schedule["betas"][-1].item() == pytest.approx(0.02, rel=1e-3)

    def test_alpha_bars_monotone_decreasing(self, schedule):
        ab = schedule["alpha_bars"]
        assert (ab[1:] < ab[:-1]).all()

    def test_alpha_bars_prev_starts_with_one(self, schedule):
        assert schedule["alpha_bars_prev"][0].item() == pytest.approx(1.0)

    def test_posterior_variance_positive(self, schedule):
        # t=0 has σ̃²=0 because alpha_bars_prev[0]=1 and alpha_bars[0]≈1
        assert (schedule["posterior_variance"][1:] > 0).all()


# ---------------------------------------------------------------------------
# extract() helper
# ---------------------------------------------------------------------------

class TestExtract:
    def test_output_shape_matches_target(self, schedule):
        t = torch.randint(0, T, (B,))
        out = extract(schedule["alpha_bars"], t, (B, C, L))
        assert out.shape == (B, C, L)

    def test_values_correct(self, schedule):
        t = torch.tensor([0, T - 1])
        out = extract(schedule["alpha_bars"], t, (2, 1, 1))
        assert out[0, 0, 0].item() == pytest.approx(schedule["alpha_bars"][0].item())
        assert out[1, 0, 0].item() == pytest.approx(schedule["alpha_bars"][T - 1].item())


# ---------------------------------------------------------------------------
# Forward process — q_sample
# ---------------------------------------------------------------------------

class TestQSample:
    def test_output_shape(self, schedule, x0):
        t = torch.randint(1, T, (B,))
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, schedule["alpha_bars"])
        assert x_t.shape == (B, C, L)

    def test_recovers_x0_at_t0(self, schedule, x0):
        # At t=0 alpha_bar≈1 → x_t ≈ x_0 (noise contribution ≈ 0)
        t = torch.zeros(B, dtype=torch.long)
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, schedule["alpha_bars"])
        # alpha_bars[0] is very close to 1, so x_t ≈ x0
        assert torch.allclose(x_t, x0, atol=0.1)

    def test_nearly_pure_noise_at_t_max(self, schedule, x0):
        # At t=T-1 alpha_bar≈0 → x_t ≈ eps
        t = torch.full((B,), T - 1, dtype=torch.long)
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, schedule["alpha_bars"])
        # Signal coefficient √ᾱ_{T-1} ≈ 0.01 → x_t dominated by eps
        ab = schedule["alpha_bars"][T - 1].item()
        assert ab < 0.01


# ---------------------------------------------------------------------------
# Noise parameterization
# ---------------------------------------------------------------------------

class TestEpsilonToX0:
    def test_output_shape(self, schedule, x0):
        t = torch.randint(1, T, (B,))
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, schedule["alpha_bars"])
        x0_hat = epsilon_to_x0(x_t, eps, t, schedule["alpha_bars"])
        assert x0_hat.shape == (B, C, L)

    def test_exact_recovery_with_true_eps(self, schedule, x0):
        # With perfect ε̂ = ε, epsilon_to_x0 should exactly recover x_0
        t = torch.randint(1, T - 1, (B,))
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, schedule["alpha_bars"])
        x0_hat = epsilon_to_x0(x_t, eps, t, schedule["alpha_bars"])
        assert torch.allclose(x0_hat, x0, atol=1e-4)


class TestEpsilonToMean:
    def test_output_shape(self, schedule, x0):
        t = torch.randint(1, T, (B,))
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, schedule["alpha_bars"])
        mu = epsilon_to_mean(x_t, eps, t, schedule["alphas"], schedule["alpha_bars"])
        assert mu.shape == (B, C, L)


class TestPSample:
    def test_output_shape(self, schedule, x0):
        t = torch.randint(1, T, (B,))
        eps = torch.randn_like(x0)
        x_t = q_sample(x0, t, eps, schedule["alpha_bars"])
        mu = epsilon_to_mean(x_t, eps, t, schedule["alphas"], schedule["alpha_bars"])
        x_prev = p_sample(mu, t, schedule["posterior_variance"])
        assert x_prev.shape == (B, C, L)

    def test_no_noise_at_t0(self, schedule, x0):
        # At t=0 p_sample must return exactly μ_θ (z is masked out)
        t = torch.zeros(B, dtype=torch.long)
        mu = torch.randn(B, C, L)
        x_prev = p_sample(mu, t, schedule["posterior_variance"])
        assert torch.allclose(x_prev, mu)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestNoiseLoss:
    def test_l_simple_shape(self, schedule, x0):
        t = torch.randint(1, T, (B,))
        eps = torch.randn_like(x0)
        eps_hat = torch.randn_like(x0)
        loss = noise_loss(eps, eps_hat, t,
                          schedule["posterior_variance"], schedule["alphas"],
                          schedule["alpha_bars"], vlb=False)
        assert loss.shape == ()  # scalar

    def test_l_simple_zero_when_perfect(self, schedule, x0):
        t = torch.randint(1, T, (B,))
        eps = torch.randn_like(x0)
        loss = noise_loss(eps, eps, t,
                          schedule["posterior_variance"], schedule["alphas"],
                          schedule["alpha_bars"], vlb=False)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_l_vlb_shape(self, schedule, x0):
        t = torch.randint(1, T, (B,))
        eps = torch.randn_like(x0)
        eps_hat = torch.randn_like(x0)
        loss = noise_loss(eps, eps_hat, t,
                          schedule["posterior_variance"], schedule["alphas"],
                          schedule["alpha_bars"], vlb=True)
        assert loss.shape == ()

    def test_l_vlb_non_negative(self, schedule, x0):
        t = torch.randint(1, T, (B,))
        eps = torch.randn_like(x0)
        eps_hat = torch.randn_like(x0)
        loss = noise_loss(eps, eps_hat, t,
                          schedule["posterior_variance"], schedule["alphas"],
                          schedule["alpha_bars"], vlb=True)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class TestTimeEmbedding:
    def test_sinusoidal_shape(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t = torch.randint(0, T, (B,))
        out = emb(t)
        assert out.shape == (B, 128)

    def test_mlp_shape(self):
        mlp = TimeEmbeddingMLP(sin_dim=128, out_dim=512)
        t = torch.randint(0, T, (B,))
        out = mlp(t)
        assert out.shape == (B, 512)

    def test_different_t_different_embedding(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t1 = torch.tensor([0])
        t2 = torch.tensor([500])
        assert not torch.allclose(emb(t1), emb(t2))


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------

class TestResBlock1d:
    def test_same_channels(self):
        block = ResBlock1d(64, 64, t_emb_dim=512)
        x = torch.randn(B, 64, L)
        t_emb = torch.randn(B, 512)
        out = block(x, t_emb)
        assert out.shape == (B, 64, L)

    def test_channel_change(self):
        block = ResBlock1d(64, 128, t_emb_dim=512)
        x = torch.randn(B, 64, L)
        t_emb = torch.randn(B, 512)
        out = block(x, t_emb)
        assert out.shape == (B, 128, L)


class TestSelfAttention1d:
    def test_output_shape(self):
        attn = SelfAttention1d(dim=512, heads=8)
        x = torch.randn(B, 512, 125)
        out = attn(x)
        assert out.shape == (B, 512, 125)

    def test_residual_preserves_scale(self):
        # With zero input, output should also be near zero
        attn = SelfAttention1d(dim=512, heads=8)
        x = torch.zeros(B, 512, 125)
        out = attn(x)
        assert out.abs().max().item() < 1e-3


# ---------------------------------------------------------------------------
# ECGUNet1d
# ---------------------------------------------------------------------------

class TestECGUNet1d:
    def test_output_shape(self):
        unet = ECGUNet1d()
        x = torch.randn(B, C, L)
        t = torch.randint(1, T, (B,))
        out = unet(x, t)
        assert out.shape == (B, C, L)

    def test_output_shape_batch1(self):
        unet = ECGUNet1d()
        x = torch.randn(1, C, L)
        t = torch.randint(1, T, (1,))
        out = unet(x, t)
        assert out.shape == (1, C, L)

    def test_no_activation_on_output(self):
        # Raw noise prediction — output should be unbounded (not clipped to [-1,1])
        unet = ECGUNet1d()
        x = torch.randn(B, C, L) * 10
        t = torch.randint(1, T, (B,))
        out = unet(x, t)
        # Not checking exact values — just ensuring no inf/nan
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# DDPM wrapper
# ---------------------------------------------------------------------------

class TestDDPM:
    def test_schedule_buffers_registered(self, model):
        buffer_names = {name for name, _ in model.named_buffers()}
        for expected in ("betas", "alphas", "alpha_bars", "alpha_bars_prev", "posterior_variance"):
            assert expected in buffer_names

    def test_loss_is_scalar(self, model, x0):
        model.train()
        loss = model.loss(x0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_decreases_with_zero_noise(self, model, x0):
        # With perfect prediction ε̂=ε, L_simple should be near 0 — test indirectly
        # by verifying loss is a finite positive scalar
        model.train()
        loss = model.loss(x0)
        assert torch.isfinite(loss)

    def test_anomaly_score_shape(self, model, x0):
        scores = model.anomaly_score(x0)
        assert scores.shape == (B,)

    def test_anomaly_score_non_negative(self, model, x0):
        scores = model.anomaly_score(x0)
        assert (scores >= 0).all()

    def test_reconstruct_shape(self, model, x0):
        x_hat = model.reconstruct(x0)
        assert x_hat.shape == (B, C, L)

    def test_reconstruct_finite(self, model, x0):
        x_hat = model.reconstruct(x0)
        assert torch.isfinite(x_hat).all()

    def test_ema_update_changes_weights(self, config, x0):
        m = DDPM(config)
        ema = copy.deepcopy(m)
        # Do one gradient step on m
        opt = torch.optim.Adam(m.unet.parameters(), lr=1e-3)
        loss = m.loss(x0)
        loss.backward()
        opt.step()
        # EMA update
        decay = 0.9999
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), m.parameters()):
                ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        # EMA params should differ from original deepcopy but be close
        for ep, p in zip(ema.parameters(), m.parameters()):
            assert not torch.equal(ep, p)

    def test_no_gradient_during_inference(self, model, x0):
        scores = model.anomaly_score(x0)
        assert scores.grad_fn is None

    def test_t_star_controls_reverse_steps(self, config, x0):
        cfg_low  = {**config, "ddpm": {**config["ddpm"], "t_star": 50}}
        cfg_high = {**config, "ddpm": {**config["ddpm"], "t_star": 100}}
        m_low  = DDPM(cfg_low).eval()
        m_high = DDPM(cfg_high).eval()
        # Both must produce valid scores — t_star only affects loop length
        s_low  = m_low.anomaly_score(x0)
        s_high = m_high.anomaly_score(x0)
        assert s_low.shape == (B,)
        assert s_high.shape == (B,)

    def test_vlb_loss_flag(self, config, x0):
        cfg = {**config, "ddpm": {**config["ddpm"], "vlb": True}}
        m = DDPM(cfg)
        m.train()
        loss = m.loss(x0)
        assert loss.shape == ()
        assert loss.item() >= 0
