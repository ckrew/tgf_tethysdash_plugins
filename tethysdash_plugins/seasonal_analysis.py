import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin
from tethysapp.tethysdash.exceptions import VisualizationError

from tethysdash_plugins._cache import load_station_data, STREAMFLOW_CACHE_DIR


def _safe_div(a, b):
    return a / b if b != 0 else np.nan


def _nse(obs, sim):
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - _safe_div(np.sum((obs - sim) ** 2), denom)


def _kge09(obs, sim):
    r = pearsonr(obs, sim)[0] if np.std(obs) > 0 and np.std(sim) > 0 else np.nan
    alpha = _safe_div(np.std(sim), np.std(obs))
    beta = _safe_div(np.mean(sim), np.mean(obs))
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


SEASONS = {
    "Winter (DJF)": [12, 1, 2],
    "Pre-monsoon (MAM)": [3, 4, 5],
    "Monsoon (JJAS)": [6, 7, 8, 9],
    "Post-monsoon (ON)": [10, 11],
}


class SeasonalAnalysis(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_seasonal_analysis"
    label = "Geoglows Seasonal Analysis"
    group = "TGF"
    description = (
        "Grouped bar chart of NSE and KGE (2009) by hydrological season, with "
        "performance threshold reference lines."
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

        self.send_update("Computing seasonal metrics...")

        season_names = []
        nse_raw_vals = []
        nse_cor_vals = []
        kge_raw_vals = []
        kge_cor_vals = []

        for season, months in SEASONS.items():
            mask = merged.index.month.isin(months)
            sub = merged[mask]
            if len(sub) < 10:
                continue

            obs = sub["Q_obs"].values
            sim = sub["Q_sim"].values
            cor = sub["Q_cor"].values

            season_names.append(season)
            nse_raw_vals.append(round(_nse(obs, sim), 4))
            nse_cor_vals.append(round(_nse(obs, cor), 4))
            kge_raw_vals.append(round(_kge09(obs, sim), 4))
            kge_cor_vals.append(round(_kge09(obs, cor), 4))

        self.send_update("Building seasonal analysis chart...")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=season_names,
                y=nse_raw_vals,
                name="NSE (Raw)",
                marker_color="#2563eb",
                opacity=0.7,
            )
        )
        fig.add_trace(
            go.Bar(
                x=season_names,
                y=nse_cor_vals,
                name="NSE (Corrected)",
                marker_color="#2563eb",
                opacity=1.0,
            )
        )
        fig.add_trace(
            go.Bar(
                x=season_names,
                y=kge_raw_vals,
                name="KGE 2009 (Raw)",
                marker_color="#16a34a",
                opacity=0.7,
            )
        )
        fig.add_trace(
            go.Bar(
                x=season_names,
                y=kge_cor_vals,
                name="KGE 2009 (Corrected)",
                marker_color="#16a34a",
                opacity=1.0,
            )
        )

        # Performance threshold reference lines
        fig.add_hline(
            y=0.75,
            line_dash="dash",
            line_color="green",
            annotation_text="Very Good (0.75)",
            annotation_position="top left",
        )
        fig.add_hline(
            y=0.65,
            line_dash="dash",
            line_color="orange",
            annotation_text="Good (0.65)",
            annotation_position="top left",
        )
        fig.add_hline(
            y=0.50,
            line_dash="dash",
            line_color="red",
            annotation_text="Acceptable (0.50)",
            annotation_position="top left",
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=50),
            template="plotly_white",
            barmode="group",
            yaxis=dict(title="Metric Value", range=[-1, 1]),
            legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": True, "responsive": True},
        }
