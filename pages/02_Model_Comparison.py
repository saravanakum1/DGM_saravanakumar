import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from components.metrics_table import metrics_table

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
st.title("Model Comparison")


@st.cache_data
def load_metrics():
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


metrics = load_metrics()
if not metrics:
    st.warning("No metrics.json found. Run eval_vae.py and eval_ddpm.py first.")
    st.stop()

tab_metrics, tab_radar, tab_dist = st.tabs(["Metrics", "Radar Chart", "Score Distribution"])

with tab_metrics:
    st.subheader("Pre-computed Evaluation Metrics")
    metrics_table(metrics)

    model_names = list(metrics.keys())
    auroc_vals = [metrics[m].get("auroc", 0) for m in model_names]
    prauc_vals = [metrics[m].get("pr_auc", 0) for m in model_names]

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(name="AUROC", x=model_names, y=auroc_vals, marker_color="#3B82F6"))
    bar_fig.add_trace(go.Bar(name="PR-AUC", x=model_names, y=prauc_vals, marker_color="#8B5CF6"))
    bar_fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Model",
        title="AUROC vs PR-AUC",
        height=350,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    if "results" in st.session_state and st.session_state["results"]:
        st.subheader("Live Inference Scores (from Upload & Analyze)")
        live = st.session_state["results"]
        thr = st.session_state.get("thresholds", {})
        for model_name, res in live.items():
            threshold = thr.get(model_name, 0.5)
            verdict = "ANOMALY" if res["anomaly_score"] > threshold else "NORMAL"
            st.write(
                f"**{model_name}**: score={res['anomaly_score']:.4f}, "
                f"threshold={threshold:.4f}, verdict={verdict}, "
                f"inference={res['inference_ms']} ms"
            )

with tab_radar:
    st.subheader("Model Radar Chart")
    model_names = list(metrics.keys())
    axes = ["AUROC", "PR-AUC", "Speed"]

    radar_fig = go.Figure()
    colors = ["#3B82F6", "#EF4444", "#22C55E"]
    for i, model_name in enumerate(model_names):
        m = metrics[model_name]
        values = [
            m.get("auroc", 0),
            m.get("pr_auc", 0),
            1.0,
        ]
        values_closed = values + [values[0]]
        axes_closed = axes + [axes[0]]
        radar_fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=axes_closed,
                fill="toself",
                name=model_name,
                line_color=colors[i % len(colors)],
                opacity=0.6,
            )
        )
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450,
    )
    st.plotly_chart(radar_fig, use_container_width=True)

with tab_dist:
    st.subheader("Score Distribution Plots")
    vae_img = ARTIFACTS / "vae_score_distribution.png"
    ddpm_img = ARTIFACTS / "ddpm_score_distribution.png"

    has_any = vae_img.exists() or ddpm_img.exists()
    if not has_any:
        st.info("Score distribution images not found. Run eval_vae.py and eval_ddpm.py to generate them.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if vae_img.exists():
                st.image(str(vae_img), caption="VAE Score Distribution", use_container_width=True)
            else:
                st.info("VAE score distribution image not found.")
        with col2:
            if ddpm_img.exists():
                st.image(str(ddpm_img), caption="DDPM Score Distribution", use_container_width=True)
            else:
                st.info("DDPM score distribution image not found.")
