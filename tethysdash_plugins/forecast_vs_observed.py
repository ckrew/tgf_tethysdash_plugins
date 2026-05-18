import os
import geoglows
import pandas as pd
import requests
import gdown
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin

OBSERVED_FOLDER_ID = "1JJ8gQWUdpH-0QAIUdgXVHF5TEw2TyHcd"
OBSERVED_CACHE_DIR = f"/tmp/tgf_observed_{OBSERVED_FOLDER_ID}"


def _ensure_cache():
    """Download the observed folder from Drive once; subsequent calls are no-ops."""
    if not os.path.isdir(OBSERVED_CACHE_DIR) or not os.listdir(OBSERVED_CACHE_DIR):
        os.makedirs(OBSERVED_CACHE_DIR, exist_ok=True)
        gdown.download_folder(
            id=OBSERVED_FOLDER_ID, output=OBSERVED_CACHE_DIR, quiet=True
        )


def load_observed(station_id: str) -> pd.DataFrame | None:
    """Find a cached CSV whose filename stem matches station_id. Expects Date index and Q column."""
    _ensure_cache()
    filename = f"{station_id}_Q"
    for fname in os.listdir(OBSERVED_CACHE_DIR):
        if os.path.splitext(fname)[0] == filename:
            df = pd.read_csv(
                os.path.join(OBSERVED_CACHE_DIR, fname), index_col=0, parse_dates=True
            )
            df = df.apply(pd.to_numeric, errors="coerce")
            df.columns = ["Q_obs"]
            df[df < 0] = 0
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df.dropna()
    return None


class ForecastVsObserved(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_forecast_vs_observed"
    label = "Geoglows Forecast Vs Observed"
    group = "TGF"
    description = "Plot GeoGLOWS retrospective streamflow against observed data for a specified COMID and date range."
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

        self.send_update("Loading retrospective streamflow from GeoGLOWS API...")
        retro = requests.get(
            f"https://geoglows.ecmwf.int/api/v2/retrospectivedaily/{self.comid}?format=json"
        ).json()
        flow_key = next(k for k in retro if k != "datetime")
        sim_df = pd.DataFrame(
            {"Q_sim": retro[flow_key]},
            index=pd.to_datetime(retro["datetime"]),
        )
        sim_df[sim_df < 0] = 0
        sim_df.index = sim_df.index.tz_localize(None)

        self.send_update("Loading observed data...")
        obs_df = load_observed(self.station_id)

        self.send_update("Applying bias correction...")
        cor_df = geoglows.bias.correct_historical(sim_df, obs_df)
        cor_df.columns = ["Q_cor"]

        # Subset all series to the requested display window
        sim_df = sim_df.loc[start:end]
        cor_df = cor_df.loc[start:end]
        obs_df = obs_df.loc[start:end]

        def to_trace(df, col, name, color, dash=None):
            line = {"color": color, "width": 1.5}
            if dash:
                line["dash"] = dash
            return {
                "type": "scatter",
                "x": df.index.strftime("%Y-%m-%d").tolist(),
                "y": df[col].tolist(),
                "name": name,
                "line": line,
            }

        traces = [
            to_trace(obs_df, "Q_obs", "Observed", "#1f77b4"),
            to_trace(sim_df, "Q_sim", "Simulated (Raw)", "#ff7f0e", dash="dash"),
            to_trace(cor_df, "Q_cor", "Simulated (Corrected)", "#2ca02c"),
        ]

        layout = {
            "title": (
                f"GeoGLOWS Retrospective vs Observed — COMID {self.comid} / Station {self.station_id}"
                f"<br><sup>{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}</sup>"
            ),
            "yaxis": {"title": "Streamflow (m³/s)"},
            "xaxis": {
                "title": "Date",
                "type": "date",
                "rangeselector": {
                    "buttons": [
                        {
                            "count": 1,
                            "label": "1M",
                            "step": "month",
                            "stepmode": "backward",
                        },
                        {
                            "count": 6,
                            "label": "6M",
                            "step": "month",
                            "stepmode": "backward",
                        },
                        {
                            "count": 1,
                            "label": "1Y",
                            "step": "year",
                            "stepmode": "backward",
                        },
                        {"step": "all", "label": "ALL"},
                    ]
                },
            },
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend": {"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
        }

        return {"data": traces, "layout": layout, "config": {"displayModeBar": True}}
