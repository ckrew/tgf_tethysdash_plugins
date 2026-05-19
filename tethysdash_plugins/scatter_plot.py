import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin
from tethysapp.tethysdash.exceptions import VisualizationError

from tethysdash_plugins._cache import load_station_data, STREAMFLOW_CACHE_DIR


class ScatterPlot(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_scatter_plot"
    label = "Geoglows Scatter Plot"
    group = "TGF"
    description = (
        "Scatter plot of observed vs simulated (raw and bias-corrected) streamflow "
        "with a 1:1 reference line."
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

        self.send_update("Building scatter plot...")

        obs = merged["Q_obs"].values
        sim = merged["Q_sim"].values
        cor = merged["Q_cor"].values

        mv = max(obs.max(), sim.max(), cor.max())

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=obs,
                y=sim,
                mode="markers",
                name="Obs vs Simulated (Raw)",
                marker=dict(color="#ff7f0e", size=4, opacity=0.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=obs,
                y=cor,
                mode="markers",
                name="Obs vs Simulated (Corrected)",
                marker=dict(color="#2ca02c", size=4, opacity=0.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, mv],
                y=[0, mv],
                name="1:1",
                mode="lines",
                line=dict(color="black", dash="dot", width=1),
                showlegend=True,
            )
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=50),
            template="plotly_white",
            xaxis=dict(title="Observed Q (m³/s)"),
            yaxis=dict(title="Simulated Q (m³/s)"),
            legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": True, "responsive": True},
        }
