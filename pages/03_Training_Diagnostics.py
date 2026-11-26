import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CONFIG_PATH = ROOT / "config.toml"

st.set_page_config(page_title="Training Diagnostics", page_icon="🔬", layout="wide")
st.title("Training Diagnostics")


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


@st.cache_data
def load_loss(model_lower: str):
    path = ARTIFACTS / f"{model_lower}_loss.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_norm_stats_data():
    path = ARTIFACTS / "norm_stats.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def vae_param_count(config):
    from backend.models.vae import VAE
    m = VAE(config)
    return sum(p.numel() for p in m.parameters())


def ddpm_param_count(config):
    from backend.models.ddpm import DDPM
    m = DDPM(config)
    return sum(p.numel() for p in m.unet.parameters())


def fm_param_count(config):
    from backend.models.flow_matching import FlowMatching
    fm_cfg = config.get("flow_matching", {})
    m = FlowMatching(time_dim=fm_cfg.get("time_dim", 128))
    return sum(p.numel() for p in m.parameters())


config = load_config()
norm_stats = load_norm_stats_data()

selected_model = st.selectbox("Select Model", ["VAE", "DDPM", "FM"])
model_lower = "fm" if selected_model == "FM" else selected_model.lower()

loss_data = load_loss(model_lower)

st.subheader("Loss Curve")
if loss_data:
    # FM uses keys "train"/"val"; VAE/DDPM use "train_loss"/"val_loss"
    train_loss = loss_data.get("train_loss") or loss_data.get("train", [])
    val_loss = loss_data.get("val_loss") or loss_data.get("val", [])
    epochs = list(range(1, len(train_loss) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines", name="Train", line=dict(color="#3B82F6")))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines", name="Val", line=dict(color="#EF4444", dash="dash")))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        title=f"{selected_model} Training Loss",
        height=350,
        legend=dict(x=0.8, y=0.95),
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Train Loss", f"{train_loss[-1]:.4f}" if train_loss else "—")
    with col2:
        st.metric("Final Val Loss", f"{val_loss[-1]:.4f}" if val_loss else "—")
    with col3:
        st.metric("Epochs", len(train_loss))
else:
    st.info(f"No loss file found at artifacts/{model_lower}_loss.json.")

st.divider()
st.subheader("Model Info")

if selected_model == "VAE":
    m_cfg = config["model"]
    try:
        n_params = vae_param_count(config)
        param_str = f"{n_params:,}"
    except Exception:
        param_str = "N/A"

    info = {
        "Architecture": "1D CNN Encoder → Latent({latent_dim}) → 1D CNN Decoder".format(**m_cfg),
        "Latent dim": m_cfg["latent_dim"],
        "Encoder channels": str(list(m_cfg["encoder_channels"])),
        "Kernel sizes": str(list(m_cfg["encoder_kernels"])),
        "Strides": str(list(m_cfg["encoder_strides"])),
        "Parameters": param_str,
        "Epochs": config["training"]["epochs"],
        "Batch size": config["training"]["batch_size"],
        "Learning rate": config["training"]["learning_rate"],
        "Threshold": f"{norm_stats.get('threshold', 'N/A'):.4f}" if norm_stats.get("threshold") else "N/A",
        "Threshold pct": f"{norm_stats.get('threshold_percentile', config['anomaly']['threshold_percentile'])}th",
    }

elif selected_model == "DDPM":
    d_cfg = config["ddpm"]
    try:
        n_params = ddpm_param_count(config)
        param_str = f"{n_params:,}"
    except Exception:
        param_str = "N/A"

    info = {
        "Architecture": "ECG UNet1d (1D ResNet + Self-Attention) with DDPM noise schedule",
        "Diffusion steps T": d_cfg["T"],
        "t* (partial diffusion)": d_cfg["t_star"],
        "Beta start": d_cfg["beta_start"],
        "Beta end": d_cfg["beta_end"],
        "VLB loss": d_cfg["vlb"],
        "EMA decay": d_cfg["ema_decay"],
        "Parameters (UNet)": param_str,
        "Epochs": config["training"]["epochs"],
        "Batch size": config["training"]["batch_size"],
        "Learning rate": config["training"]["learning_rate"],
        "Threshold": f"{norm_stats.get('ddpm_threshold', 'N/A'):.4f}" if norm_stats.get("ddpm_threshold") else "N/A",
        "Threshold pct": f"{norm_stats.get('ddpm_threshold_percentile', config['anomaly']['threshold_percentile'])}th",
    }

elif selected_model == "FM":
    fm_cfg = config.get("flow_matching", {})
    try:
        n_params = fm_param_count(config)
        param_str = f"{n_params:,}"
    except Exception:
        param_str = "N/A"

    info = {
        "Architecture": "1D U-Net velocity field (Conditional Flow Matching, straight paths)",
        "Scoring method": "CFM reconstruction error E[||v_θ(x_t,t) − (x−z)||²]",
        "Time embedding dim": fm_cfg.get("time_dim", 128),
        "ODE steps (generation)": fm_cfg.get("num_steps", 50),
        "Score samples (eval)": fm_cfg.get("score_samples", 20),
        "EMA decay": fm_cfg.get("ema_decay", 0.9999),
        "Parameters": param_str,
        "Epochs": fm_cfg.get("epochs", "N/A"),
        "Batch size": fm_cfg.get("batch_size", "N/A"),
        "Learning rate": fm_cfg.get("lr", "N/A"),
        "Grad clip": fm_cfg.get("grad_clip", "N/A"),
        "Threshold": f"{norm_stats.get('fm_threshold', 'N/A'):.4f}" if norm_stats.get("fm_threshold") else "N/A",
        "Threshold pct": f"{fm_cfg.get('threshold_percentile', config['anomaly']['threshold_percentile'])}th",
    }

for k, v in info.items():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"**{k}**")
    with col2:
        st.write(str(v))

if selected_model == "VAE":
    st.divider()
    st.subheader("Latent Space")
    st.info(
        "t-SNE 2D projection of the test set latent vectors is available after running inference. "
        "This visualization requires loading the full test set through the encoder — "
        "use the Upload & Analyze page to inspect individual samples."
    )

per_class_img = ARTIFACTS / f"{model_lower}_per_class_auroc.png"
eval_script = {"vae": "eval_vae.py", "ddpm": "eval_ddpm.py", "fm": "eval_flow_matching.py"}[model_lower]
st.divider()
st.subheader("Per-Class AUROC")
if per_class_img.exists():
    st.image(str(per_class_img), caption=f"{selected_model} Per-Class AUROC (one-vs-NORM)", width="stretch")
else:
    st.info(f"Run src/eval/{eval_script} to generate the per-class AUROC chart.")
