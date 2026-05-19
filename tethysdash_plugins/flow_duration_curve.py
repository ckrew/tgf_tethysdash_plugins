import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin

from tethysdash_plugins._cache import load_station_data, STREAMFLOW_CACHE_DIR


class FlowDurationCurve(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_flow_duration_curve"
    label = "Geoglows Flow Duration Curve"
    group = "TGF"
    description = (
        "Flow duration curves for observed, raw simulated, and bias-corrected "
        "streamflow on a log-scale y-axis."
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
        window = df.loc[start:end]

        self.send_update("Building flow duration curve...")

        def fdc(arr):
            s = np.sort(arr[arr > 0])[::-1]
            return np.arange(1, len(s) + 1) / len(s) * 100, s

        fig = go.Figure()

        # Sim and corrected — always available
        sim = window["Q_sim"].dropna().values
        cor = window["Q_cor"].dropna().values
        if len(sim) > 1:
            exc_s, sim_s = fdc(sim)
            fig.add_trace(go.Scatter(x=exc_s, y=sim_s, name="Simulated (Raw)",
                line=dict(color="#ff7f0e", width=1.2, dash="dash")))
        if len(cor) > 1:
            exc_c, cor_s = fdc(cor)
            fig.add_trace(go.Scatter(x=exc_c, y=cor_s, name="Simulated (Corrected)",
                line=dict(color="#2ca02c", width=1.2, dash="dot")))

        # Observed — only when available in this window
        obs = window["Q_obs"].dropna().values
        if len(obs) > 1:
            exc_o, obs_s = fdc(obs)
            fig.add_trace(go.Scatter(x=exc_o, y=obs_s, name="Observed",
                line=dict(color="#1f77b4", width=1.5)))

        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=50),
            template="plotly_white",
            xaxis=dict(title="Exceedance Probability (%)"),
            yaxis=dict(title="Q (m³/s)", type="log"),
            legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": True, "responsive": True},
        }
