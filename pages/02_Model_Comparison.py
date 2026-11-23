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

tab_metrics, tab_per_class, tab_radar, tab_dist = st.tabs(["Metrics", "Per-Class Best Model", "Radar Chart", "Score Distribution"])

MODEL_COLORS = {"VAE": "#3B82F6", "DDPM": "#EF4444", "FM": "#8B5CF6"}

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

with tab_per_class:
    st.subheader("Per-Class AUROC: Which Model Wins?")
    st.caption("One-vs-NORM AUROC per anomaly class. The highest-scoring model for each class is the recommended detector.")

    # Collect per-class AUROC across all models that have it
    all_classes = set()
    per_class_data: dict[str, dict[str, float]] = {}  # class -> {model -> auroc}
    for model_name, m in metrics.items():
        for cls, val in m.get("per_class_auroc", {}).items():
            all_classes.add(cls)
            per_class_data.setdefault(cls, {})[model_name] = val

    if not per_class_data:
        st.info("No per-class AUROC data found. Run eval scripts to generate metrics.json with per_class_auroc.")
    else:
        classes = sorted(all_classes)
        model_names_with_class = sorted({m for v in per_class_data.values() for m in v})

        # Grouped bar chart: one group per class, one bar per model
        bar_fig2 = go.Figure()
        for model_name in model_names_with_class:
            vals = [per_class_data.get(cls, {}).get(model_name, None) for cls in classes]
            bar_fig2.add_trace(go.Bar(
                name=model_name,
                x=classes,
                y=vals,
                marker_color=MODEL_COLORS.get(model_name, "#6B7280"),
            ))
        bar_fig2.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1], title="AUROC (one-vs-NORM)"),
            xaxis_title="Anomaly Class",
            title="Per-Class AUROC by Model",
            height=400,
            legend=dict(x=0.01, y=0.99),
        )
        bar_fig2.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="Random (0.5)")
        st.plotly_chart(bar_fig2, use_container_width=True)

        # Winner table
        st.subheader("Best Model per Class")
        cols = st.columns(len(classes))
        for col, cls in zip(cols, classes):
            class_scores = per_class_data.get(cls, {})
            if not class_scores:
                continue
            best_model = max(class_scores, key=class_scores.get)
            best_score = class_scores[best_model]
            color = MODEL_COLORS.get(best_model, "#6B7280")
            col.markdown(
                f"**{cls}**  \n"
                f"<span style='background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold'>{best_model}</span>  \n"
                f"AUROC: `{best_score:.3f}`",
                unsafe_allow_html=True,
            )
            # Show all models ranked
            ranked = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (mn, sc) in enumerate(ranked, 1):
                col.caption(f"{rank}. {mn}: {sc:.3f}")

with tab_radar:
    st.subheader("Model Radar Chart")
    model_names = list(metrics.keys())
    axes = ["AUROC", "PR-AUC", "Speed"]

    radar_fig = go.Figure()
    colors = ["#3B82F6", "#EF4444", "#8B5CF6"]
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
    dist_imgs = {
        "VAE": ARTIFACTS / "vae_score_distribution.png",
        "DDPM": ARTIFACTS / "ddpm_score_distribution.png",
        "FM": ARTIFACTS / "fm_score_distribution.png",
    }
    available = {k: v for k, v in dist_imgs.items() if v.exists()}

    if not available:
        st.info("Score distribution images not found. Run eval scripts to generate them.")
    else:
        cols = st.columns(len(available))
        for col, (model_name, img_path) in zip(cols, available.items()):
            col.image(str(img_path), caption=f"{model_name} Score Distribution", use_container_width=True)
