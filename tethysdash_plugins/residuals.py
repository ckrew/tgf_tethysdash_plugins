import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin
from tethysapp.tethysdash.exceptions import VisualizationError

from tethysdash_plugins._cache import load_station_data, STREAMFLOW_CACHE_DIR


class Residuals(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_residuals"
    label = "Geoglows Residuals"
    group = "TGF"
    description = (
        "Time-series of model residuals (simulated minus observed) for raw and "
        "bias-corrected GEOGloWS streamflow."
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

        self.send_update("Building residuals plot...")

        dates = merged.index
        obs = merged["Q_obs"].values
        sim = merged["Q_sim"].values
        cor = merged["Q_cor"].values

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sim - obs,
                name="Simulated (Raw) − Obs",
                line=dict(color="#d62728", width=1, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cor - obs,
                name="Simulated (Corrected) − Obs",
                line=dict(color="#9467bd", width=1, dash="dot"),
            )
        )

        fig.add_hline(y=0, line_dash="dot", line_color="gray")

        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=50),
            template="plotly_white",
            hovermode="x unified",
            yaxis=dict(title="Sim − Obs (m³/s)"),
            xaxis=dict(
                type="date",
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL"),
                    ]
                ),
            ),
            legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": True, "responsive": True},
        }
