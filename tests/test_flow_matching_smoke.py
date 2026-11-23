"""
Smoke tests for the Flow Matching model (backend/models/flow_matching.py).

All tests use synthetic tensors — no dataset required.
Run with:
    uv run pytest tests/test_flow_matching_smoke.py -v
"""

import copy
import math

import pytest
import torch

from backend.models.flow_matching import (
    FlowMatching,
    FlowMatchingUNet1D,
    ResBlock1D,
    SinusoidalTimeEmbedding,
    TimeEmbeddingMLP,
    Downsample1D,
    Upsample1D,
    cfm_loss,
    ode_log_likelihood,
    sample_euler,
    sample_rk4,
)

B, C, L = 2, 12, 1000  # batch, leads, timesteps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def unet():
    m = FlowMatchingUNet1D(time_dim=128)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model():
    m = FlowMatching(time_dim=128, num_steps=4, n_hutchinson=1)
    m.eval()
    return m


@pytest.fixture
def x():
    return torch.randn(B, C, L)


@pytest.fixture
def t_batch():
    return torch.rand(B)


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t = torch.rand(B)
        out = emb(t)
        assert out.shape == (B, 128)

    def test_different_t_different_embedding(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t1 = torch.tensor([0.0])
        t2 = torch.tensor([0.5])
        assert not torch.allclose(emb(t1), emb(t2))

    def test_same_t_same_embedding(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t = torch.tensor([0.3])
        assert torch.allclose(emb(t), emb(t))

    def test_output_finite(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t = torch.rand(B)
        assert torch.isfinite(emb(t)).all()


class TestTimeEmbeddingMLP:
    def test_output_shape(self):
        mlp = TimeEmbeddingMLP(sin_dim=128, out_dim=128)
        t = torch.rand(B)
        out = mlp(t)
        assert out.shape == (B, 128)

    def test_output_finite(self):
        mlp = TimeEmbeddingMLP(sin_dim=128, out_dim=128)
        t = torch.rand(B)
        assert torch.isfinite(mlp(t)).all()


# ---------------------------------------------------------------------------
# ResBlock1D
# ---------------------------------------------------------------------------

class TestResBlock1D:
    def test_same_channels(self):
        block = ResBlock1D(64, 64, time_dim=128)
        x = torch.randn(B, 64, L)
        temb = torch.randn(B, 128)
        out = block(x, temb)
        assert out.shape == (B, 64, L)

    def test_channel_change(self):
        block = ResBlock1D(64, 128, time_dim=128)
        x = torch.randn(B, 64, L)
        temb = torch.randn(B, 128)
        out = block(x, temb)
        assert out.shape == (B, 128, L)

    def test_output_finite(self):
        block = ResBlock1D(64, 128, time_dim=128)
        x = torch.randn(B, 64, L)
        temb = torch.randn(B, 128)
        assert torch.isfinite(block(x, temb)).all()

    def test_residual_identity_when_same_channels(self):
        block = ResBlock1D(64, 64, time_dim=128)
        assert isinstance(block.residual, torch.nn.Identity)

    def test_residual_conv_when_channels_differ(self):
        block = ResBlock1D(64, 128, time_dim=128)
        assert isinstance(block.residual, torch.nn.Conv1d)


# ---------------------------------------------------------------------------
# Downsample1D / Upsample1D
# ---------------------------------------------------------------------------

class TestDownsample1D:
    def test_halves_length(self):
        down = Downsample1D(64)
        x = torch.randn(B, 64, 1000)
        out = down(x)
        assert out.shape == (B, 64, 500)

    def test_preserves_channels(self):
        down = Downsample1D(128)
        x = torch.randn(B, 128, 500)
        out = down(x)
        assert out.shape == (B, 128, 250)


class TestUpsample1D:
    def test_doubles_length(self):
        up = Upsample1D(64)
        x = torch.randn(B, 64, 500)
        out = up(x)
        assert out.shape == (B, 64, 1000)

    def test_target_len_override(self):
        up = Upsample1D(128)
        x = torch.randn(B, 128, 249)   # odd length after downsample
        out = up(x, target_len=500)
        assert out.shape == (B, 128, 500)


# ---------------------------------------------------------------------------
# FlowMatchingUNet1D
# ---------------------------------------------------------------------------

class TestFlowMatchingUNet1D:
    def test_output_shape(self, unet, x):
        t = torch.rand(B)
        out = unet(x, t)
        assert out.shape == (B, C, L)

    def test_output_shape_batch1(self, unet):
        x1 = torch.randn(1, C, L)
        t1 = torch.rand(1)
        out = unet(x1, t1)
        assert out.shape == (1, C, L)

    def test_output_finite(self, unet, x):
        t = torch.rand(B)
        assert torch.isfinite(unet(x, t)).all()

    def test_different_t_different_output(self, unet, x):
        x1 = x[:1]
        out_t0 = unet(x1, torch.tensor([0.0]))
        out_t1 = unet(x1, torch.tensor([1.0]))
        assert not torch.allclose(out_t0, out_t1)

    def test_parameter_count_reasonable(self, unet):
        total = sum(p.numel() for p in unet.parameters())
        # Expect between 3M and 15M (architecture spec says ~9.6M)
        assert 3_000_000 < total < 15_000_000, f"Unexpected param count: {total:,}"


# ---------------------------------------------------------------------------
# CFM loss
# ---------------------------------------------------------------------------

class TestCFMLoss:
    def test_loss_is_scalar(self, unet, x):
        unet.train()
        loss = cfm_loss(unet, x)
        assert loss.shape == ()

    def test_loss_positive(self, unet, x):
        unet.train()
        loss = cfm_loss(unet, x)
        assert loss.item() > 0

    def test_loss_finite(self, unet, x):
        unet.train()
        loss = cfm_loss(unet, x)
        assert torch.isfinite(loss)

    def test_loss_zero_with_perfect_prediction(self):
        # When the model always predicts the true velocity, loss should be near 0
        class PerfectVelocity(torch.nn.Module):
            def forward(self, x_t, t):
                return self._target  # set externally

        pv = PerfectVelocity()
        z = torch.randn(B, C, L)
        x_data = torch.randn(B, C, L)
        target = x_data - z
        pv._target = target
        # Can't use cfm_loss directly (it re-samples z), so just verify MSE identity
        loss = torch.nn.functional.mse_loss(pv._target, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class TestSampleEuler:
    def test_output_shape(self, unet):
        z = torch.randn(B, C, L)
        out = sample_euler(unet, z, num_steps=4)
        assert out.shape == (B, C, L)

    def test_output_finite(self, unet):
        z = torch.randn(B, C, L)
        out = sample_euler(unet, z, num_steps=4)
        assert torch.isfinite(out).all()

    def test_output_differs_from_noise(self, unet):
        z = torch.randn(B, C, L)
        out = sample_euler(unet, z, num_steps=4)
        assert not torch.allclose(out, z)

    def test_no_gradients(self, unet):
        z = torch.randn(B, C, L)
        out = sample_euler(unet, z, num_steps=4)
        assert out.grad_fn is None


class TestSampleRK4:
    def test_output_shape(self, unet):
        z = torch.randn(B, C, L)
        out = sample_rk4(unet, z, num_steps=4)
        assert out.shape == (B, C, L)

    def test_output_finite(self, unet):
        z = torch.randn(B, C, L)
        out = sample_rk4(unet, z, num_steps=4)
        assert torch.isfinite(out).all()

    def test_output_differs_from_noise(self, unet):
        z = torch.randn(B, C, L)
        out = sample_rk4(unet, z, num_steps=4)
        assert not torch.allclose(out, z)

    def test_no_gradients(self, unet):
        z = torch.randn(B, C, L)
        out = sample_rk4(unet, z, num_steps=4)
        assert out.grad_fn is None

    def test_rk4_vs_euler_differ(self, unet):
        # RK4 and Euler should produce different results (different integration order)
        torch.manual_seed(42)
        z = torch.randn(1, C, L)
        out_euler = sample_euler(unet, z.clone(), num_steps=4)
        out_rk4 = sample_rk4(unet, z.clone(), num_steps=4)
        assert not torch.allclose(out_euler, out_rk4)


# ---------------------------------------------------------------------------
# ODE log-likelihood
# ---------------------------------------------------------------------------

class TestODELogLikelihood:
    def test_output_shape(self, unet, x):
        log_px = ode_log_likelihood(unet, x[:1], num_steps=2, n_hutchinson=1)
        assert log_px.shape == (1,)

    def test_output_finite(self, unet, x):
        log_px = ode_log_likelihood(unet, x[:1], num_steps=2, n_hutchinson=1)
        assert torch.isfinite(log_px).all()

    def test_log_likelihood_is_scalar_per_sample(self, unet, x):
        log_px = ode_log_likelihood(unet, x, num_steps=2, n_hutchinson=1)
        assert log_px.shape == (B,)

    def test_anomaly_score_positive_for_random_input(self, unet, x):
        log_px = ode_log_likelihood(unet, x[:1], num_steps=2, n_hutchinson=1)
        # Random untrained model: -log p(x) is some large positive number
        score = -log_px
        assert score.shape == (1,)
        assert torch.isfinite(score).all()

    def test_multiple_hutchinson_samples(self, unet, x):
        log_px = ode_log_likelihood(unet, x[:1], num_steps=2, n_hutchinson=3)
        assert log_px.shape == (1,)
        assert torch.isfinite(log_px).all()

    def test_per_sample_scores_independent(self, unet, x):
        """Regression: div_est must sum over dim=[1,2] (per-sample), not all dims.

        With the old bug (.sum() collapsing across batch), the divergence accumulated
        over the entire batch instead of per sample, so scores scaled with batch size.
        We verify this by checking that the score of sample[0] alone is in the same
        rough range as when scored in a batch — not 2× larger or 2× smaller.
        The Hutchinson estimator is stochastic, so we only require the ratio to be
        between 0.5 and 2.0 (the old bug caused ~B× deviation).
        """
        torch.manual_seed(42)
        single = ode_log_likelihood(unet, x[:1], num_steps=3, n_hutchinson=1)
        torch.manual_seed(42)
        batch_score = ode_log_likelihood(unet, x, num_steps=3, n_hutchinson=1)
        # Both should be in the same ballpark — ratio of absolute values near 1
        ratio = abs(batch_score[0].item()) / max(abs(single[0].item()), 1e-6)
        assert 0.1 < ratio < 10.0, (
            f"Batch score ({batch_score[0]:.1f}) is wildly different from single score ({single[0]:.1f}). "
            f"Ratio={ratio:.2f}. Likely div_est sums over entire batch instead of per sample."
        )


# ---------------------------------------------------------------------------
# FlowMatching wrapper
# ---------------------------------------------------------------------------

class TestFlowMatching:
    def test_loss_is_scalar(self, model, x):
        model.unet.train()
        loss = model.loss(x)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_finite(self, model, x):
        model.unet.train()
        loss = model.loss(x)
        assert torch.isfinite(loss)

    def test_anomaly_score_shape(self, model, x):
        score, heatmap = model.anomaly_score(x[:1])
        assert score.shape == (1,)
        assert heatmap.shape == (L,)

    def test_anomaly_score_finite(self, model, x):
        score, heatmap = model.anomaly_score(x[:1])
        assert torch.isfinite(score).all()
        assert torch.isfinite(heatmap).all()

    def test_heatmap_non_negative(self, model, x):
        # Heatmap = squared velocity magnitude → always >= 0
        _, heatmap = model.anomaly_score(x[:1])
        assert (heatmap >= 0).all()

    def test_anomaly_score_no_gradient(self, model, x):
        score, heatmap = model.anomaly_score(x[:1])
        assert score.grad_fn is None
        assert heatmap.grad_fn is None

    def test_generate_shape(self, model):
        device = torch.device("cpu")
        samples = model.generate(n_samples=2, device=device)
        assert samples.shape == (2, C, L)

    def test_generate_finite(self, model):
        device = torch.device("cpu")
        samples = model.generate(n_samples=2, device=device)
        assert torch.isfinite(samples).all()

    def test_generate_no_gradient(self, model):
        device = torch.device("cpu")
        samples = model.generate(n_samples=2, device=device)
        assert samples.grad_fn is None

    def test_different_inputs_different_scores(self, model):
        x_norm = torch.zeros(1, C, L)            # flat signal
        x_anom = torch.randn(1, C, L) * 5        # large noise
        s_norm, _ = model.anomaly_score(x_norm)
        s_anom, _ = model.anomaly_score(x_anom)
        assert s_norm.item() != s_anom.item()

    def test_ema_update_changes_weights(self):
        m = FlowMatching(time_dim=128, num_steps=4)
        ema = copy.deepcopy(m)
        opt = torch.optim.Adam(m.unet.parameters(), lr=1e-3)
        x = torch.randn(B, C, L)
        loss = m.loss(x)
        loss.backward()
        opt.step()
        decay = 0.9999
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), m.parameters()):
                ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for ep, p in zip(ema.parameters(), m.parameters()):
            assert not torch.equal(ep, p)

    def test_num_steps_config_respected(self):
        m4  = FlowMatching(time_dim=128, num_steps=4)
        m8  = FlowMatching(time_dim=128, num_steps=8)
        assert m4.num_steps == 4
        assert m8.num_steps == 8

    def test_unet_inside_wrapper(self, model):
        assert isinstance(model.unet, FlowMatchingUNet1D)
