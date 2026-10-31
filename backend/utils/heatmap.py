import numpy as np


def timestep_heatmap(raw_heatmap: np.ndarray, colorscale_thresholds=(0.3, 0.5, 0.7)) -> np.ndarray:
    if raw_heatmap.max() > 0:
        return raw_heatmap / raw_heatmap.max()
    return raw_heatmap
