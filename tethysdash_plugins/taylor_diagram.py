import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin
from tethysapp.tethysdash.exceptions import VisualizationError

from tethysdash_plugins._cache import load_station_data, STREAMFLOW_CACHE_DIR


class TaylorDiagram(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_taylor_diagram"
    label = "Geoglows Taylor Diagram"
    group = "TGF"
    description = (
        "Taylor diagram summarising standard deviation, correlation, and RMSE for "
        "raw and bias-corrected GEOGloWS streamflow against observations."
    )
    tags = ["Geoglows", "Bias", "Metrics", "Nepal"]
    args = {
        "comid": "text",
        "station_id": "text",
        "start_date": "date",
        "end_date": "date",
    }

    def run(self):
        start = pd.Timestamp(self.start_date).tz_convert(None)
        end = pd.Timestamp(self.end_date).tz_convert(None)
        df = load_station_data(self.station_id, self.comid, self.send_update)
        merged_full = df.loc[start:end]
        merged = merged_full.dropna()  # inner join — rows with all three columns
        if len(merged) < 2:
            raise VisualizationError(
                f"No overlapping observed and simulated data for station '{self.station_id}' "
                f"between {start.date()} and {end.date()}. "
                "Try a wider date range or check that the station CSV covers this period."
            )

        self.send_update("Computing Taylor diagram statistics...")

        obs = merged["Q_obs"].values
        sim = merged["Q_sim"].values
        cor = merged["Q_cor"].values

        std_obs = float(np.std(obs))
        std_sim = float(np.std(sim))
        std_cor = float(np.std(cor))

        r_sim = float(pearsonr(obs, sim)[0])
        r_cor = float(pearsonr(obs, cor)[0])

        rmse_sim = float(np.sqrt(np.mean((obs - sim) ** 2)))
        rmse_cor = float(np.sqrt(np.mean((obs - cor) ** 2)))

        # Taylor-diagram polar coordinates: x = std * r,  y = std * sqrt(1 - r^2)
        def _td_xy(std, r):
            return std * r, std * math.sqrt(max(0.0, 1 - r**2))

        x_obs, y_obs = std_obs, 0.0  # reference point always on x-axis
        x_sim, y_sim = _td_xy(std_sim, r_sim)
        x_cor, y_cor = _td_xy(std_cor, r_cor)

        max_axis_val = (
            max(std_obs, std_sim, std_cor, std_obs + rmse_sim, std_obs + rmse_cor) * 1.2
        )

        self.send_update("Building Taylor diagram...")

        fig = go.Figure()

        # ── Data points ────────────────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=[x_obs],
                y=[y_obs],
                mode="markers",
                name="Observed (Reference)",
                marker=dict(color="black", symbol="star", size=14),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_sim],
                y=[y_sim],
                mode="markers",
                name="Simulated Raw",
                marker=dict(color="#ff7f0e", symbol="circle", size=10),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_cor],
                y=[y_cor],
                mode="markers",
                name="Simulated Corrected",
                marker=dict(color="#2ca02c", symbol="circle", size=10),
            )
        )

        # ── Correlation arc lines (r = 0.1 … 0.9) ─────────────────────────
        for r_val in np.arange(0.1, 1.0, 0.1):
            angle = math.acos(r_val)  # angle from x-axis (radians)
            x_end = max_axis_val * math.cos(angle)
            y_end = max_axis_val * math.sin(angle)
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=x_end,
                y1=y_end,
                line=dict(color="lightgray", width=1, dash="dot"),
            )
            # Label at 95% of max_axis_val
            lx = 0.95 * max_axis_val * math.cos(angle)
            ly = 0.95 * max_axis_val * math.sin(angle)
            fig.add_annotation(
                x=lx,
                y=ly,
                text=f"R={r_val:.1f}",
                showarrow=False,
                font=dict(size=9, color="gray"),
                xanchor="center",
                yanchor="middle",
            )

        # ── Standard-deviation arcs centred at origin (first quadrant only) ──
        _theta = np.linspace(0, np.pi / 2, 100)
        max_std = max(std_obs, std_sim, std_cor)
        n_std_circles = 4
        for i in range(1, n_std_circles + 1):
            radius = max_std * i / n_std_circles
            if radius > max_axis_val:
                break
            fig.add_trace(
                go.Scatter(
                    x=radius * np.cos(_theta),
                    y=radius * np.sin(_theta),
                    mode="lines",
                    line=dict(color="lightblue", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_annotation(
                x=radius,
                y=0,
                text=f"{radius:.2f}",
                showarrow=False,
                font=dict(size=8, color="steelblue"),
                yanchor="top",
            )

        # ── RMSE arcs centred at the observed reference point ──────────────
        max_rmse = max(rmse_sim, rmse_cor)
        n_rmse_arcs = 3
        for j in range(1, n_rmse_arcs + 1):
            r_arc = max_rmse * j / n_rmse_arcs
            # Only draw the portion of the arc where x >= 0
            _arc_theta = np.linspace(0, np.pi, 200)
            _arc_x = x_obs + r_arc * np.cos(_arc_theta)
            _arc_y = y_obs + r_arc * np.sin(_arc_theta)
            mask = _arc_x >= 0
            fig.add_trace(
                go.Scatter(
                    x=_arc_x[mask],
                    y=_arc_y[mask],
                    mode="lines",
                    line=dict(color="pink", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Annotate on the y=0 line to the right of obs reference
            fig.add_annotation(
                x=x_obs + r_arc,
                y=0,
                text=f"RMSE={r_arc:.2f}",
                showarrow=False,
                font=dict(size=8, color="salmon"),
                yanchor="bottom",
            )

        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=50),
            template="plotly_white",
            hovermode="closest",
            xaxis=dict(
                minallowed=-10,
                title="Standard Deviation",
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                minallowed=-10,
                title="Standard Deviation",
            ),
            legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": True, "responsive": True},
        }
