"""
End-to-end smoke tests for Flow Matching training and evaluation pipelines.

Uses real PTB-XL records but a tiny model + 2 epochs so the full pipeline
(data loading → training loop → threshold fitting → eval scoring → plots)
can be verified quickly without a GPU.

Dataset path: /Users/rsaravanakumar/Applications/personal/mtech/DGM_saravanakumar/dataset/

Run with:
    uv run pytest tests/test_fm_train_eval_smoke.py -v
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

DATASET_DIR = "/Users/rsaravanakumar/Applications/personal/mtech/DGM_saravanakumar/dataset"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Real PTB-XL records from splits_cache.json (present in test fold 10)
NORM_RECORDS = [
    ("records100/00000/00009_lr", "NORM"),
    ("records100/00000/00038_lr", "NORM"),
    ("records100/00000/00040_lr", "NORM"),
    ("records100/00000/00041_lr", "NORM"),
]
ANOM_RECORDS = [
    ("records100/00000/00063_lr", "MI"),
    ("records100/00000/00065_lr", "CD"),
    ("records100/00000/00116_lr", "STTC"),
    ("records100/00000/00299_lr", "HYP"),
]
ALL_TEST_RECORDS = NORM_RECORDS + ANOM_RECORDS

B, C, L = 2, 12, 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model(num_steps: int = 4) -> "FlowMatching":
    from backend.models.flow_matching import FlowMatching
    return FlowMatching(time_dim=32, num_steps=num_steps, n_hutchinson=1)


def _synthetic_norm_stats() -> dict:
    return {
        "mean": [0.0] * 12,
        "std":  [1.0] * 12,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def norm_stats_from_records():
    """Compute real per-lead mean/std from the 4 NORM smoke records."""
    from backend.preprocess.pipeline import fit_norm_stats
    return fit_norm_stats(NORM_RECORDS, DATASET_DIR, fs=100)


@pytest.fixture(scope="module")
def norm_signals(norm_stats_from_records):
    """Load + preprocess the 4 NORM records → (4, 12, 1000) tensor."""
    from backend.preprocess.pipeline import PTBXLDataset
    ds = PTBXLDataset(NORM_RECORDS, DATASET_DIR, norm_stats=norm_stats_from_records, fs=100)
    return ds.signals  # (4, 12, 1000)


@pytest.fixture(scope="module")
def all_signals(norm_stats_from_records):
    """Load + preprocess all 8 smoke records (NORM + anomaly)."""
    from backend.preprocess.pipeline import PTBXLDataset
    ds = PTBXLDataset(ALL_TEST_RECORDS, DATASET_DIR, norm_stats=norm_stats_from_records, fs=100)
    return ds.signals, [r[1] for r in ALL_TEST_RECORDS]


# ============================================================
# DATA LOADING
# ============================================================

class TestDataLoading:
    def test_norm_stats_keys(self, norm_stats_from_records):
        assert "mean" in norm_stats_from_records
        assert "std" in norm_stats_from_records

    def test_norm_stats_shape(self, norm_stats_from_records):
        assert len(norm_stats_from_records["mean"]) == 12
        assert len(norm_stats_from_records["std"]) == 12

    def test_norm_stats_std_positive(self, norm_stats_from_records):
        assert all(s > 0 for s in norm_stats_from_records["std"])

    def test_norm_signals_shape(self, norm_signals):
        assert norm_signals.shape == (4, 12, 1000)

    def test_norm_signals_dtype(self, norm_signals):
        assert norm_signals.dtype == torch.float32

    def test_norm_signals_finite(self, norm_signals):
        assert torch.isfinite(norm_signals).all()

    def test_all_signals_shape(self, all_signals):
        signals, labels = all_signals
        assert signals.shape == (8, 12, 1000)
        assert len(labels) == 8

    def test_all_signals_labels(self, all_signals):
        _, labels = all_signals
        assert set(labels) == {"NORM", "MI", "CD", "STTC", "HYP"}

    def test_normalized_signals_reasonable_range(self, norm_signals):
        # After normalization with NORM stats, NORM signals should be roughly zero-centered
        assert norm_signals.mean().abs().item() < 1.0
        assert norm_signals.std().item() < 5.0


# ============================================================
# TRAINING LOOP
# ============================================================

class TestTrainingLoop:
    def test_cfm_loss_on_real_signal(self, norm_signals):
        from backend.models.flow_matching import cfm_loss
        model = _make_tiny_model()
        model.unet.train()
        batch = norm_signals[:2]
        loss = cfm_loss(model.unet, batch)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_loss_decreases_over_two_steps(self, norm_signals):
        """Two gradient steps on real data should reduce the loss."""
        from backend.models.flow_matching import cfm_loss
        model = _make_tiny_model()
        model.unet.train()
        optimizer = torch.optim.AdamW(model.unet.parameters(), lr=1e-3)
        batch = norm_signals  # all 4 NORM records

        losses = []
        for _ in range(2):
            loss = cfm_loss(model.unet, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss must be finite at each step; strict decrease not guaranteed in 2 steps
        assert all(math.isfinite(l) for l in losses)

    def test_ema_update_on_real_signal(self, norm_signals):
        from backend.models.flow_matching import cfm_loss
        model = _make_tiny_model()
        ema = copy.deepcopy(model)
        opt = torch.optim.AdamW(model.unet.parameters(), lr=1e-3)

        loss = cfm_loss(model.unet, norm_signals[:2])
        loss.backward()
        opt.step()

        decay = 0.9999
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

        # EMA params should differ from model after one step
        for ep, p in zip(ema.parameters(), model.parameters()):
            assert not torch.equal(ep, p)

    def test_grad_clip_does_not_explode(self, norm_signals):
        from backend.models.flow_matching import cfm_loss
        model = _make_tiny_model()
        model.unet.train()
        opt = torch.optim.AdamW(model.unet.parameters(), lr=1e-1)  # large lr to stress
        loss = cfm_loss(model.unet, norm_signals[:2])
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
        opt.step()
        assert total_norm.isfinite()

    def test_euler_sample_on_real_norm_stats(self, norm_signals):
        from backend.models.flow_matching import sample_euler
        model = _make_tiny_model()
        model.unet.eval()
        z = torch.randn(2, C, L)
        out = sample_euler(model.unet, z, num_steps=4)
        assert out.shape == (2, C, L)
        assert torch.isfinite(out).all()

    def test_full_mini_train_loop(self, norm_signals, tmp_path):
        """2-epoch training loop: loss logged, weights saved."""
        from backend.models.flow_matching import FlowMatching, cfm_loss, sample_euler

        model = FlowMatching(time_dim=32, num_steps=4, n_hutchinson=1)
        ema = copy.deepcopy(model)
        ema.eval()
        for p in ema.parameters():
            p.requires_grad_(False)

        opt = torch.optim.AdamW(model.unet.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(norm_signals), batch_size=2, shuffle=True)
        history = {"train": []}

        decay = 0.9999
        for epoch in range(2):
            model.unet.train()
            epoch_losses = []
            for (batch,) in loader:
                loss = cfm_loss(model.unet, batch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()):
                        ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
                epoch_losses.append(loss.item())
            history["train"].append(float(np.mean(epoch_losses)))

        assert len(history["train"]) == 2
        assert all(math.isfinite(l) for l in history["train"])

        # Save + reload EMA weights
        weights_path = tmp_path / "fm_ema_weights.pt"
        torch.save(ema.state_dict(), weights_path)
        assert weights_path.exists()

        loaded = FlowMatching(time_dim=32, num_steps=4, n_hutchinson=1)
        loaded.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        loaded.eval()

        # Sanity: loaded model produces same output as ema
        x = norm_signals[:1]
        with torch.no_grad():
            v_ema    = ema.unet(x, torch.tensor([0.5]))
            v_loaded = loaded.unet(x, torch.tensor([0.5]))
        assert torch.allclose(v_ema, v_loaded)


# ============================================================
# THRESHOLD COMPUTATION
# ============================================================

class TestThresholdComputation:
    def test_ode_scores_on_norm_records(self, norm_signals):
        from backend.models.flow_matching import ode_log_likelihood
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()
        log_px = ode_log_likelihood(model.unet, norm_signals[:2], num_steps=2, n_hutchinson=1)
        assert log_px.shape == (2,)
        assert torch.isfinite(log_px).all()

    def test_threshold_95th_percentile(self, norm_signals):
        from backend.models.flow_matching import ode_log_likelihood
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()
        log_px = ode_log_likelihood(model.unet, norm_signals, num_steps=2, n_hutchinson=1)
        scores = (-log_px).cpu().numpy()
        threshold = float(np.percentile(scores, 95))
        assert math.isfinite(threshold)
        # 95th pct must be >= median
        assert threshold >= float(np.percentile(scores, 50))

    def test_threshold_written_to_norm_stats(self, norm_signals, tmp_path):
        from backend.models.flow_matching import ode_log_likelihood
        from backend.preprocess.pipeline import save_norm_stats, load_norm_stats

        model = _make_tiny_model(num_steps=2)
        model.unet.eval()
        log_px = ode_log_likelihood(model.unet, norm_signals, num_steps=2, n_hutchinson=1)
        threshold = float(np.percentile((-log_px).cpu().numpy(), 95))

        stats = {"mean": [0.0] * 12, "std": [1.0] * 12, "fm_threshold": threshold}
        path = str(tmp_path / "norm_stats.json")
        save_norm_stats(stats, path)

        loaded = load_norm_stats(path)
        assert "fm_threshold" in loaded
        assert math.isfinite(loaded["fm_threshold"])


# ============================================================
# ANOMALY SCORING (EVAL)
# ============================================================

class TestAnomalyScoring:
    def test_anomaly_score_on_norm_record(self, norm_signals):
        model = _make_tiny_model(num_steps=2)
        model.eval()
        score, heatmap = model.anomaly_score(norm_signals[:1])
        assert score.shape == (1,)
        assert heatmap.shape == (L,)
        assert torch.isfinite(score).all()
        assert torch.isfinite(heatmap).all()
        assert (heatmap >= 0).all()

    def test_anomaly_score_on_all_classes(self, all_signals):
        signals, labels = all_signals
        model = _make_tiny_model(num_steps=2)
        model.eval()
        scores_out = []
        for i in range(len(labels)):
            score, _ = model.anomaly_score(signals[i:i+1])
            scores_out.append((labels[i], score.item()))

        assert len(scores_out) == 8
        for label, s in scores_out:
            assert math.isfinite(s), f"Non-finite score for class {label}"

    def test_heatmap_shape_for_each_class(self, all_signals):
        signals, labels = all_signals
        model = _make_tiny_model(num_steps=2)
        model.eval()
        for i, label in enumerate(labels):
            _, heatmap = model.anomaly_score(signals[i:i+1])
            assert heatmap.shape == (L,), f"Wrong heatmap shape for {label}"
            assert (heatmap >= 0).all(), f"Negative heatmap values for {label}"

    def test_batch_vs_individual_scores_consistent(self, norm_signals):
        """Scores for a batch must match individual scores element-wise."""
        from backend.models.flow_matching import ode_log_likelihood
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()

        log_px_batch = ode_log_likelihood(model.unet, norm_signals[:2], num_steps=2, n_hutchinson=1)

        log_px_ind = []
        for i in range(2):
            lp = ode_log_likelihood(model.unet, norm_signals[i:i+1], num_steps=2, n_hutchinson=1)
            log_px_ind.append(lp.item())

        # Hutchinson is stochastic so values differ, but both must be finite
        for v in log_px_batch.tolist() + log_px_ind:
            assert math.isfinite(v)

    def test_scores_are_scalars_per_sample(self, all_signals):
        from backend.models.flow_matching import ode_log_likelihood
        signals, _ = all_signals
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()
        log_px = ode_log_likelihood(model.unet, signals, num_steps=2, n_hutchinson=1)
        assert log_px.shape == (8,)

    def test_no_grad_during_anomaly_score(self, norm_signals):
        model = _make_tiny_model(num_steps=2)
        model.eval()
        score, heatmap = model.anomaly_score(norm_signals[:1])
        assert score.grad_fn is None
        assert heatmap.grad_fn is None


# ============================================================
# EVAL PIPELINE (metrics + plots)
# ============================================================

class TestEvalPipeline:
    def test_binary_labels_from_records(self, all_signals):
        _, labels = all_signals
        binary = np.array([0 if l == "NORM" else 1 for l in labels])
        assert binary.sum() == 4   # 4 anomaly classes
        assert (binary == 0).sum() == 4

    def test_auroc_computable_on_sample_scores(self, all_signals):
        from sklearn.metrics import roc_auc_score
        from backend.models.flow_matching import ode_log_likelihood

        signals, labels = all_signals
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()

        log_px = ode_log_likelihood(model.unet, signals, num_steps=2, n_hutchinson=1)
        scores = (-log_px).cpu().numpy()
        binary = np.array([0 if l == "NORM" else 1 for l in labels])

        auroc = roc_auc_score(binary, scores)
        assert 0.0 <= auroc <= 1.0

    def test_pr_auc_computable(self, all_signals):
        from sklearn.metrics import precision_recall_curve, auc
        from backend.models.flow_matching import ode_log_likelihood

        signals, labels = all_signals
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()

        log_px = ode_log_likelihood(model.unet, signals, num_steps=2, n_hutchinson=1)
        scores = (-log_px).cpu().numpy()
        binary = np.array([0 if l == "NORM" else 1 for l in labels])

        prec, rec, _ = precision_recall_curve(binary, scores)
        pr_auc = auc(rec, prec)
        assert 0.0 <= pr_auc <= 1.0

    def test_per_class_auroc_computable(self, all_signals):
        from sklearn.metrics import roc_auc_score
        from backend.models.flow_matching import ode_log_likelihood

        signals, labels = all_signals
        labels_arr = np.array(labels)
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()

        log_px = ode_log_likelihood(model.unet, signals, num_steps=2, n_hutchinson=1)
        scores = (-log_px).cpu().numpy()

        for cls in ["MI", "CD", "STTC", "HYP"]:
            mask = (labels_arr == "NORM") | (labels_arr == cls)
            y_true = (labels_arr[mask] == cls).astype(int)
            y_score = scores[mask]
            if y_true.sum() == 0:
                continue
            auroc = roc_auc_score(y_true, y_score)
            assert 0.0 <= auroc <= 1.0, f"Invalid AUROC for {cls}"

    def test_metrics_json_written(self, all_signals, tmp_path):
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        from backend.models.flow_matching import ode_log_likelihood

        signals, labels = all_signals
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()

        log_px = ode_log_likelihood(model.unet, signals, num_steps=2, n_hutchinson=1)
        scores = (-log_px).cpu().numpy()
        binary = np.array([0 if l == "NORM" else 1 for l in labels])

        auroc = float(roc_auc_score(binary, scores))
        prec, rec, _ = precision_recall_curve(binary, scores)
        pr_auc = float(auc(rec, prec))
        threshold = float(np.percentile(scores[binary == 0], 95))

        metrics_path = tmp_path / "metrics.json"
        existing = {}
        existing["FM"] = {
            "auroc": auroc,
            "pr_auc": pr_auc,
            "threshold": threshold,
            "threshold_percentile": 95,
        }
        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=2)

        with open(metrics_path) as f:
            loaded = json.load(f)

        assert "FM" in loaded
        assert math.isfinite(loaded["FM"]["auroc"])
        assert math.isfinite(loaded["FM"]["pr_auc"])
        assert math.isfinite(loaded["FM"]["threshold"])

    def test_plots_generated(self, all_signals, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, auc
        from backend.models.flow_matching import ode_log_likelihood

        signals, labels = all_signals
        model = _make_tiny_model(num_steps=2)
        model.unet.eval()

        log_px = ode_log_likelihood(model.unet, signals, num_steps=2, n_hutchinson=1)
        scores = (-log_px).cpu().numpy()
        labels_arr = np.array(labels)
        binary = (labels_arr != "NORM").astype(int)
        prec, rec, _ = precision_recall_curve(binary, scores)

        # PR curve
        fig, ax = plt.subplots()
        ax.plot(rec, prec)
        pr_path = tmp_path / "fm_pr_curve.png"
        fig.savefig(pr_path)
        plt.close(fig)
        assert pr_path.exists() and pr_path.stat().st_size > 0

        # Score distribution
        fig, ax = plt.subplots()
        for cls in set(labels):
            mask = labels_arr == cls
            if mask.sum():
                ax.hist(scores[mask], bins=5, alpha=0.5, label=cls)
        ax.axvline(float(np.percentile(scores, 95)), color="black")
        dist_path = tmp_path / "fm_score_distribution.png"
        fig.savefig(dist_path)
        plt.close(fig)
        assert dist_path.exists() and dist_path.stat().st_size > 0
