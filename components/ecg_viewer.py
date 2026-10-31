import numpy as np
import plotly.graph_objects as go

LEAD_LABELS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_HEATMAP_COLORS = {0.3: "#FDE68A", 0.5: "#FB923C", 0.7: "#EF4444"}


def ecg_viewer(
    signal: np.ndarray,
    heatmaps: dict[str, np.ndarray] | None = None,
    leads_to_show: list[int] | None = None,
    title: str = "ECG Signal",
) -> go.Figure:
    if leads_to_show is None:
        leads_to_show = list(range(12))

    fig = go.Figure()
    timesteps = np.arange(signal.shape[1])

    for idx, lead_idx in enumerate(leads_to_show):
        offset = idx * -3
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=signal[lead_idx] + offset,
                mode="lines",
                name=LEAD_LABELS[lead_idx],
                line=dict(width=1, color="#1E40AF"),
                hovertemplate=f"{LEAD_LABELS[lead_idx]}: %{{y:.3f}}<extra></extra>",
            )
        )

    if heatmaps:
        first_heatmap = next(iter(heatmaps.values()))
        norm_max = first_heatmap.max()
        if norm_max > 0:
            norm_heatmap = first_heatmap / norm_max
        else:
            norm_heatmap = first_heatmap

        thresholds = sorted(_HEATMAP_COLORS.keys(), reverse=True)
        for t in thresholds:
            color = _HEATMAP_COLORS[t]
            regions = np.where(norm_heatmap >= t)[0]
            if len(regions) == 0:
                continue
            breaks = np.where(np.diff(regions) > 1)[0]
            starts = np.concatenate([[regions[0]], regions[breaks + 1]])
            ends = np.concatenate([regions[breaks], [regions[-1]]])
            for s, e in zip(starts, ends):
                fig.add_vrect(
                    x0=int(s), x1=int(e) + 1,
                    fillcolor=color, opacity=0.25, line_width=0,
                    layer="below",
                )

    n_leads = len(leads_to_show)
    tick_vals = [i * -3 for i in range(n_leads)]
    tick_text = [LEAD_LABELS[leads_to_show[i]] for i in range(n_leads)]

    fig.update_layout(
        title=title,
        xaxis_title="Timestep",
        yaxis=dict(tickvals=tick_vals, ticktext=tick_text, showgrid=False),
        showlegend=False,
        height=max(300, n_leads * 60),
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor="#F8FAFC",
    )
    return fig
