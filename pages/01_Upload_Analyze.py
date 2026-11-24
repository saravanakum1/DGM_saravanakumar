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
from backend.preprocess.registry import get_spec
from components.disagreement_banner import disagreement_banner
from components.ecg_viewer import ecg_viewer
from components.model_card import model_card
from backend.utils.heatmap import timestep_heatmap

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.toml"

# Dataset options shown in the selector
_DATASET_OPTIONS = {
    "PTB-XL": "ptbxl",
    "MIMIC-IV-ECG": "mimic-iv-ecg",
}

# Per-dataset artifact directories (each dataset trains into its own subfolder)
_ARTIFACTS_DIR = {
    "ptbxl":        ROOT / "artifacts",
    "mimic-iv-ecg": ROOT / "artifacts" / "mimic",
}


@st.cache_data
def _load_config():
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


@st.cache_data
def _load_norm_stats(dataset_key: str):
    art_dir = _ARTIFACTS_DIR.get(dataset_key, ROOT / "artifacts")
    path = art_dir / "norm_stats.json"
    if path.exists():
        return load_norm_stats(str(path))
    # Fallback to root artifacts for ptbxl
    fallback = ROOT / "artifacts" / "norm_stats.json"
    if fallback.exists():
        return load_norm_stats(str(fallback))
    return {}


@st.cache_resource
def _load_models_for(dataset_key: str, device_str: str):
    config = _load_config()
    # Override dataset name in config for the selected dataset
    config = {**config, "data": {**config["data"], "name": dataset_key}}
    art_dir = _ARTIFACTS_DIR.get(dataset_key, ROOT / "artifacts")
    device = torch.device(device_str)
    return load_models(config, art_dir, device)


def _get_thresholds(norm_stats: dict) -> dict:
    return {
        "VAE":  norm_stats.get("threshold", 1.0),
        "DDPM": norm_stats.get("ddpm_threshold", 0.5),
        "FM":   norm_stats.get("fm_threshold", 10.0),
    }


st.set_page_config(page_title="Upload & Analyze", page_icon="🫀", layout="wide")
st.title("Upload & Analyze")

config = _load_config()
device_str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dataset")
    # Default selection mirrors what's in config.toml so the user doesn't have to re-pick
    _config_dataset = config["data"].get("name", "ptbxl")
    _default_label = next(
        (label for label, key in _DATASET_OPTIONS.items() if key == _config_dataset),
        "PTB-XL",
    )
    selected_dataset_label = st.selectbox(
        "Inference dataset",
        list(_DATASET_OPTIONS.keys()),
        index=list(_DATASET_OPTIONS.keys()).index(_default_label),
        help="Choose which dataset's model weights and norm stats to use for inference.",
    )
    selected_dataset_key = _DATASET_OPTIONS[selected_dataset_label]

    # Resolve spec for the selected dataset (class colors, normal label, seq_len)
    _spec_config = {**config, "data": {**config["data"], "name": selected_dataset_key}}
    spec = get_spec(_spec_config)

    norm_stats = _load_norm_stats(selected_dataset_key)
    thresholds = _get_thresholds(norm_stats)

    art_dir = _ARTIFACTS_DIR.get(selected_dataset_key, ROOT / "artifacts")
    if not any(art_dir.glob("*.pt")):
        st.warning(f"No model weights found in `{art_dir.relative_to(ROOT)}`. Train the model first.")

    st.divider()
    st.header("Input")
    # PTB-XL Record ID shortcut is only valid for PTB-XL
    if selected_dataset_key == "ptbxl":
        input_mode = st.radio("Input mode", ["Upload file", "PTB-XL Record ID"])
    else:
        input_mode = "Upload file"
        st.info("Upload a `.csv` or `.npy` ECG file (12 leads × 5000 samples).")

    uploaded_file = None
    record_id = None
    ptbxl_res = "500 Hz (records500)"  # default; overridden below when input_mode == PTB-XL Record ID
    if input_mode == "Upload file":
        uploaded_file = st.file_uploader("ECG file (.csv or .npy)", type=["csv", "npy"])
    else:
        record_id = st.text_input("PTB-XL Record ID (e.g. 00001)")
        ptbxl_res = st.radio("PTB-XL resolution", ["500 Hz (records500)", "100 Hz (records100)"], index=0)

    st.header("Models")
    selected_models = []
    if st.checkbox("VAE", value=True):
        selected_models.append("VAE")
    if st.checkbox("DDPM", value=True):
        selected_models.append("DDPM")
    if st.checkbox("Flow Matching", value=True):
        selected_models.append("FM")

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
        # Auto-transpose if loaded as (5000, 12) instead of (12, 5000)
        if arr.ndim == 2 and arr.shape == (5000, 12):
            arr = arr.T
        if arr.shape != (12, 5000):
            st.error(f"Expected shape (12, 5000) or (5000, 12), got {arr.shape}.")
            st.stop()
        signal = arr.astype(np.float32)

    elif record_id is not None and record_id.strip():
        rid = record_id.strip().zfill(5)
        ptbxl_dir = Path(config["data"].get("dataset_dir", str(ROOT / "dataset")))
        # ptbxl_res is only defined when input_mode == "PTB-XL Record ID"
        use_500 = "500" in ptbxl_res
        rec_folder = "records500" if use_500 else "records100"
        rec_suffix = "_hr" if use_500 else "_lr"
        record_path = str(ptbxl_dir / rec_folder / rid[:2] / f"{rid}{rec_suffix}")
        try:
            signal = load_record(record_path)
        except Exception as e:
            st.error(f"Failed to load record {rid} from {rec_folder}: {e}")
            st.stop()
    else:
        st.warning("Provide an ECG file or record ID before clicking Analyze.")
        st.stop()

    fs = config["data"].get("sampling_rate", 500)
    if apply_bandpass:
        signal = bandpass_filter(signal, fs=fs)

    raw_signal = signal.copy()

    if apply_normalize and norm_stats:
        mean = np.array(norm_stats["mean"])
        std = np.array(norm_stats["std"])
        signal = (signal - mean[:, None]) / (std[:, None] + 1e-8)

    signal_tensor = torch.from_numpy(signal).unsqueeze(0).float()

    with st.spinner("Loading model weights (downloading from HF Hub on first run)..."):
        all_models = _load_models_for(selected_dataset_key, device_str)
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
            x=list(range(5000)),
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
