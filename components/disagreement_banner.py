import streamlit as st


def disagreement_banner(results: dict, thresholds: dict) -> None:
    verdicts = {}
    for model_name, result in results.items():
        thr = thresholds.get(model_name, 0.5)
        verdicts[model_name] = "ANOMALY" if result["anomaly_score"] > thr else "NORMAL"

    unique_verdicts = set(verdicts.values())
    if len(unique_verdicts) > 1:
        anomaly_models = [m for m, v in verdicts.items() if v == "ANOMALY"]
        normal_models = [m for m, v in verdicts.items() if v == "NORMAL"]
        st.warning(
            f"⚠ Models disagree — "
            f"{', '.join(anomaly_models)} predict ANOMALY while "
            f"{', '.join(normal_models)} predict NORMAL."
        )
