import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from backend.inference import load_models, run_inference
from backend.preprocess.pipeline import bandpass_filter, load_norm_stats, load_record
from components.disagreement_banner import disagreement_banner
from components.ecg_viewer import ecg_viewer
from components.model_card import model_card
from backend.utils.heatmap import timestep_heatmap

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CONFIG_PATH = ROOT / "config.toml"


@st.cache_data
def _load_norm_stats():
    path = ARTIFACTS / "norm_stats.json"
    if path.exists():
        return load_norm_stats(str(path))
    return {}


@st.cache_data
def _load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


@st.cache_resource
def _load_models(device_str: str):
    config = _load_config()
    device = torch.device(device_str)
    return load_models(config, ARTIFACTS, device)


def _get_thresholds(norm_stats: dict) -> dict:
    return {
        "VAE": norm_stats.get("threshold", 1.0),
        "DDPM": norm_stats.get("ddpm_threshold", 0.5),
    }


st.set_page_config(page_title="Upload & Analyze", page_icon="🫀", layout="wide")
st.title("Upload & Analyze")

norm_stats = _load_norm_stats()
config = _load_config()
thresholds = _get_thresholds(norm_stats)

device_str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Input mode", ["Upload file", "PTB-XL Record ID"])

    uploaded_file = None
    record_id = None
    if input_mode == "Upload file":
        uploaded_file = st.file_uploader("ECG file (.csv or .npy)", type=["csv", "npy"])
    else:
        record_id = st.text_input("PTB-XL Record ID (e.g. 00001)")

    st.header("Models")
    selected_models = []
    if st.checkbox("VAE", value=True):
        selected_models.append("VAE")
    if st.checkbox("DDPM", value=True):
        selected_models.append("DDPM")

    st.header("Leads")
    lead_mode = st.radio("Lead selection", ["All 12", "Lead II", "Custom"])
    if lead_mode == "All 12":
        selected_leads = list(range(12))
    elif lead_mode == "Lead II":
        selected_leads = [1]
    else:
        lead_labels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        chosen = st.multiselect("Select leads", lead_labels, default=lead_labels[:6])
        selected_leads = [lead_labels.index(l) for l in chosen]

    st.header("Preprocessing")
    apply_bandpass = st.checkbox("Bandpass filter (0.5–40 Hz)", value=True)
    apply_normalize = st.checkbox("Normalize", value=True)

    analyze = st.button("ANALYZE", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Load signal
# ---------------------------------------------------------------------------
signal = None
if analyze:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".npy":
            arr = np.load(uploaded_file)
        else:
            import pandas as pd
            df = pd.read_csv(uploaded_file, header=None)
            arr = df.values
        if arr.shape[0] == 1000 and arr.shape[1] == 12:
            arr = arr.T
        if arr.shape != (12, 1000):
            st.error(f"Expected shape (12, 1000) or (1000, 12), got {arr.shape}.")
            st.stop()
        signal = arr.astype(np.float32)

    elif record_id is not None and record_id.strip():
        rid = record_id.strip().zfill(5)
        record_path = str(ROOT / "dataset" / "records100" / rid[:2] / f"{rid}_lr")
        try:
            signal = load_record(record_path)
        except Exception as e:
            st.error(f"Failed to load record {rid}: {e}")
            st.stop()
    else:
        st.warning("Provide an ECG file or record ID before clicking Analyze.")
        st.stop()

    fs = config["data"]["sampling_rate"]
    if apply_bandpass:
        signal = bandpass_filter(signal, fs=fs)

    raw_signal = signal.copy()

    if apply_normalize and norm_stats:
        mean = np.array(norm_stats["mean"])
        std = np.array(norm_stats["std"])
        signal = (signal - mean[:, None]) / (std[:, None] + 1e-8)

    signal_tensor = torch.from_numpy(signal).unsqueeze(0).float()

    with st.spinner("Running inference..."):
        all_models = _load_models(device_str)
        active_models = {k: v for k, v in all_models.items() if k in selected_models}
        if not active_models:
            st.error("No model weights found for selected models.")
            st.stop()
        results = run_inference(active_models, signal_tensor, norm_stats, config, torch.device(device_str))

    st.session_state["results"] = results
    st.session_state["ecg_signal"] = signal
    st.session_state["ecg_raw"] = raw_signal if apply_normalize else signal
    st.session_state["selected_leads"] = selected_leads
    st.session_state["thresholds"] = thresholds

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    ecg_sig = st.session_state["ecg_signal"]
    leads = st.session_state.get("selected_leads", list(range(12)))
    thr = st.session_state.get("thresholds", thresholds)

    heatmaps = {k: timestep_heatmap(v["heatmap"]) for k, v in results.items()}

    st.subheader("ECG Waveform")
    fig = ecg_viewer(ecg_sig, heatmaps=heatmaps, leads_to_show=leads)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Results")
    cols = st.columns(len(results))
    for col, (model_name, result) in zip(cols, results.items()):
        model_threshold = thr.get(model_name, 0.5)
        model_card(model_name, result, model_threshold, col)

    disagreement_banner(results, thr)

    st.subheader("Anomaly Heatmap (per model)")
    model_names = list(results.keys())
    z = np.stack([timestep_heatmap(results[m]["heatmap"]) for m in model_names], axis=0)
    heat_fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(range(1000)),
            y=model_names,
            colorscale="RdYlGn_r",
            zmin=0, zmax=1,
            colorbar=dict(title="Anomaly"),
        )
    )
    heat_fig.update_layout(
        xaxis_title="Timestep",
        yaxis_title="Model",
        height=200 + 60 * len(model_names),
        margin=dict(l=60, r=20, t=20, b=40),
    )
    st.plotly_chart(heat_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Model Comparison →"):
            st.switch_page("pages/02_Model_Comparison.py")
    with col2:
        if st.button("View Training Diagnostics →"):
            st.switch_page("pages/03_Training_Diagnostics.py")

else:
    st.info("Upload an ECG file or enter a PTB-XL Record ID in the sidebar, then click **ANALYZE**.")
