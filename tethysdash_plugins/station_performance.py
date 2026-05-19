import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin

from tethysdash_plugins._cache import STREAMFLOW_CACHE_DIR

_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "static", "NepalStations.geojson")

with open(_GEOJSON_PATH) as _f:
    _STATION_MAP = {
        feat["properties"]["samplingFeatureCode"]: feat["properties"]["name"]
        for feat in json.load(_f)["features"]
    }


def _bar_color(nse: float) -> str:
    if nse >= 0.5:
        return "#16a34a"
    if nse >= 0.3:
        return "#f59e0b"
    return "#ef4444"


def _metrics(obs, sim) -> dict | None:
    if np.std(obs) == 0 or np.std(sim) == 0:
        return None
    r = float(pearsonr(obs, sim)[0])
    alpha = float(np.std(sim) / np.std(obs))
    beta = float(np.mean(sim) / np.mean(obs))
    cv_obs = float(np.std(obs) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    cv_sim = float(np.std(sim) / np.mean(sim)) if np.mean(sim) != 0 else np.nan
    nse = round(float(1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)), 4)
    kge09 = round(float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)), 4)
    kge12 = round(float(1 - np.sqrt((r - 1) ** 2 + ((cv_sim / cv_obs) - 1) ** 2 + (beta - 1) ** 2)), 4)
    return {"NSE": nse, "KGE (2009)": kge09, "KGE (2012)": kge12}


class StationPerformance(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_station_performance"
    label = "Geoglows Station Performance"
    group = "TGF"
    description = "All-station performance overview: NSE and KGE (2012) for raw and bias-corrected GEOGloWS streamflow across all cached stations."
    tags = ["Geoglows", "Bias", "Metrics", "Nepal"]
    args = {"start_date": "date", "end_date": "date"}

    def run(self):
        start = pd.Timestamp(self.start_date).tz_convert(None)
        end = pd.Timestamp(self.end_date).tz_convert(None)

        self.send_update("Computing metrics for all stations...")
        records = []
        for fname in sorted(os.listdir(STREAMFLOW_CACHE_DIR)):
            if not fname.endswith(".parquet"):
                continue
            station_id = fname.replace(".parquet", "")
            df = pd.read_parquet(os.path.join(STREAMFLOW_CACHE_DIR, fname), engine="pyarrow")
            merged = df.loc[start:end].dropna()
            if len(merged) < 30:
                continue
            obs = merged["Q_obs"].values
            sim = merged["Q_sim"].values
            cor = merged["Q_cor"].values
            m_raw = _metrics(obs, sim)
            if m_raw is None:
                continue
            records.append({
                "name": _STATION_MAP.get(station_id, station_id),
                "raw": m_raw,
            })

        self.send_update("Building chart...")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["NSE", "KGE (2012)"],
            horizontal_spacing=0.10,
        )

        panels = [
            ("NSE",        1),
            ("KGE (2012)", 2),
        ]

        for metric, col in panels:
            sorted_records = sorted(records, key=lambda r: r["raw"][metric])
            names  = [r["name"] for r in sorted_records]
            values = [r["raw"][metric] for r in sorted_records]
            colors = [_bar_color(r["raw"][metric]) for r in sorted_records]

            fig.add_trace(go.Bar(
                x=values, y=names,
                orientation="h",
                marker_color=colors,
                showlegend=False,
                hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.4f}}<extra></extra>",
            ), row=1, col=col)

            fig.add_vline(x=0.5, line_dash="dot", line_color="gray", row=1, col=col)
            fig.add_vline(x=0.0, line_dash="solid", line_color="lightgray", line_width=1, row=1, col=col)

        for label, color in [
            ("≥ 0.5 — Acceptable+ (NSE) / Good (KGE)", "#16a34a"),
            ("0.3–0.5 — Weak (NSE & KGE)",              "#f59e0b"),
            ("< 0.3 — Poor (NSE & KGE)",                "#ef4444"),
        ]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=color, size=10, symbol="square"),
                name=label, showlegend=True,
            ))

        fig.update_layout(
            title=dict(
                text=f"All-Station Performance  ·  {start.date()} to {end.date()}",
                x=0.5, xanchor="center",
            ),
            template="plotly_white",
            autosize=True,
            margin=dict(l=120, r=20, t=60, b=30),
            legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": True, "responsive": True},
        }
