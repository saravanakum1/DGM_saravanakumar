import pandas as pd
import streamlit as st


def metrics_table(metrics: dict) -> None:
    rows = []
    for model_name, m in metrics.items():
        rows.append({
            "Model": model_name,
            "AUROC": f"{m.get('auroc', 0):.4f}",
            "PR-AUC": f"{m.get('pr_auc', 0):.4f}",
            "Threshold": f"{m.get('threshold', 0):.4f}",
            "Threshold Pct": f"{m.get('threshold_percentile', 95)}th",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
