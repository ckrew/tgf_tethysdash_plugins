import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin
from tethysapp.tethysdash.exceptions import VisualizationError

from tethysdash_plugins._cache import load_station_data
from tethysdash_plugins.compute_metrics import classify


def _compute(obs, sim) -> list:
    def safe_div(a, b):
        return a / b if b != 0 else np.nan

    r = pearsonr(obs, sim)[0] if np.std(obs) > 0 and np.std(sim) > 0 else np.nan
    alpha = safe_div(np.std(sim), np.std(obs))
    beta = safe_div(np.mean(sim), np.mean(obs))
    cv_obs = safe_div(np.std(obs), np.mean(obs))
    cv_sim = safe_div(np.std(sim), np.mean(sim))
    nse = round(1 - safe_div(np.sum((obs - sim) ** 2), np.sum((obs - np.mean(obs)) ** 2)), 4)
    kge09 = round(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), 4)
    kge12 = round(1 - np.sqrt((r - 1) ** 2 + (safe_div(cv_sim, cv_obs) - 1) ** 2 + (beta - 1) ** 2), 4)
    rmse = round(np.sqrt(np.mean((obs - sim) ** 2)), 4)
    mae = round(np.mean(np.abs(obs - sim)), 4)
    pbias = round(100 * np.sum(obs - sim) / np.sum(obs), 2)
    r2 = round(r ** 2, 4)
    pr = round(r, 4)
    sr = round(spearmanr(obs, sim)[0], 4)
    return [
        f"{nse} → {classify(nse)}",
        f"{kge09} → {classify(kge09)}",
        f"{kge12} → {classify(kge12)}",
        f"{rmse} m³/s",
        f"{mae} m³/s",
        f"{pbias}%",
        str(r2),
        str(pr),
        str(sr),
    ]


STAT_NAMES = ["NSE", "KGE (2009)", "KGE (2012)", "RMSE", "MAE", "PBIAS", "R²", "Pearson R", "Spearman R"]


class MetricsTable(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_metrics_table"
    label = "Geoglows Metrics Table"
    group = "TGF"
    description = "Performance metrics table (NSE, KGE, RMSE, MAE, PBIAS, R², Pearson R, Spearman R) for raw and bias-corrected GEOGloWS streamflow against observations."
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
        merged = df.loc[start:end].dropna()
        if len(merged) < 2:
            raise VisualizationError(
                f"No overlapping observed and simulated data for station '{self.station_id}' "
                f"between {start.date()} and {end.date()}. "
                "Try a wider date range or check that the station CSV covers this period."
            )

        self.send_update("Computing metrics...")
        obs = merged["Q_obs"].values
        sim = merged["Q_sim"].values
        cor = merged["Q_cor"].values

        raw_vals = _compute(obs, sim)
        cor_vals = _compute(obs, cor)

        fig = go.Figure(go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Simulated (Raw)</b>", "<b>Simulated (Corrected)</b>"],
                fill_color="#f0f0f0",
                align="left",
                font=dict(size=13),
                height=32,
            ),
            cells=dict(
                values=[STAT_NAMES, raw_vals, cor_vals],
                align="left",
                font=dict(size=12),
                height=30,
            ),
        ))

        fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white",
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": False, "responsive": True},
        }
