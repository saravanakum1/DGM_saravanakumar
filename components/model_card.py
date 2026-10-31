import streamlit as st


def model_card(model_name: str, result: dict, threshold: float, st_col) -> None:
    raw_score = result["anomaly_score"]
    normalized = min(raw_score / (2 * threshold), 1.0)
    verdict = "ANOMALY" if raw_score > threshold else "NORMAL"
    conf = min(abs(raw_score - threshold) / (threshold + 1e-8), 1.0) * 100
    verdict_color = "#EF4444" if verdict == "ANOMALY" else "#22C55E"

    with st_col:
        st.markdown(f"### {model_name}")
        st.markdown(
            f'<span style="background:{verdict_color};color:white;padding:4px 10px;'
            f'border-radius:4px;font-weight:bold">{verdict}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.caption(f"Anomaly Score: {raw_score:.4f}")
        st.progress(float(normalized))
        st.caption(f"Confidence: {conf:.1f}%")
        st.caption(f"Inference: {result['inference_ms']} ms")
