import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="ECG Anomaly Detection", page_icon="🫀", layout="wide")

st.title("🫀 ECG Anomaly Detection Dashboard")
st.markdown("""
Compare VAE and DDPM models for ECG anomaly detection on the PTB-XL dataset.

**Pages:**
- **Upload & Analyze** — Upload an ECG signal and run inference
- **Model Comparison** — Compare pre-computed metrics (AUROC, PR-AUC)
- **Training Diagnostics** — Loss curves and model info
""")

st.divider()

metrics_path = Path(__file__).resolve().parent / "artifacts" / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

    st.subheader("Pre-computed Evaluation Metrics")
    from components.metrics_table import metrics_table
    metrics_table(metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("VAE AUROC", f"{metrics.get('VAE', {}).get('auroc', 0):.4f}")
    with col2:
        st.metric("DDPM AUROC", f"{metrics.get('DDPM', {}).get('auroc', 0):.4f}")
else:
    st.info("No pre-computed metrics found. Run `eval_vae.py` and `eval_ddpm.py` to generate them.")
