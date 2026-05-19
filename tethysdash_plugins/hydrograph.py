import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin

from tethysdash_plugins._cache import load_station_data, STREAMFLOW_CACHE_DIR


class Hydrograph(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_hydrograph"
    label = "Geoglows Hydrograph"
    group = "TGF"
    description = (
        "Time-series hydrograph comparing GEOGloWS simulated (raw and bias-corrected) "
        "streamflow against observed gauge data."
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

        # Use merged_full (all sim dates) so gaps in obs appear as NaN, not
        # connected lines.
        merged_full = df.loc[start:end]
        merged = merged_full.dropna()  # inner join — rows with all three columns

        self.send_update("Building hydrograph...")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=merged_full.index,
                y=merged_full["Q_obs"],
                name="Observed",
                line=dict(color="#1f77b4", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged_full.index,
                y=merged_full["Q_sim"],
                name="Simulated (Raw)",
                line=dict(color="#ff7f0e", width=1.2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged_full.index,
                y=merged_full["Q_cor"],
                name="Simulated (Corrected)",
                line=dict(color="#2ca02c", width=1.2, dash="dot"),
            )
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=50),
            template="plotly_white",
            hovermode="x unified",
            yaxis=dict(title="Streamflow (m³/s)"),
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
