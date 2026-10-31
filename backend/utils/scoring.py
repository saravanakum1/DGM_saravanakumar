def normalize_score(score: float, threshold: float) -> float:
    return min(score / (2 * threshold), 1.0)


def confidence(score: float, threshold: float) -> float:
    return min(abs(score - threshold) / (threshold + 1e-8), 1.0)


def predict_verdict(score: float, threshold: float) -> str:
    return "ANOMALY" if score > threshold else "NORMAL"
