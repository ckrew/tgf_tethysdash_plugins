import json
import os
import geoglows
import numpy as np
import pandas as pd
import requests
import gdown
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin
from tethysdash_plugins.compute_metrics import classify

OBSERVED_FOLDER_ID = "1JJ8gQWUdpH-0QAIUdgXVHF5TEw2TyHcd"
OBSERVED_CACHE_DIR = f"/tmp/tgf_observed_{OBSERVED_FOLDER_ID}"
STREAMFLOW_CACHE_DIR = os.path.join(os.path.dirname(__file__), "streamflow_cache")


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


def _metrics_table(obs, sim_raw, sim_cor) -> tuple[list, list, list]:
    """Return (stat_names, raw_values, corrected_values) formatted for a plotly table."""

    def safe_div(a, b):
        return a / b if b != 0 else np.nan

    def _compute(obs, sim):
        r = pearsonr(obs, sim)[0] if np.std(obs) > 0 and np.std(sim) > 0 else np.nan
        alpha = safe_div(np.std(sim), np.std(obs))
        beta = safe_div(np.mean(sim), np.mean(obs))
        cv_obs = safe_div(np.std(obs), np.mean(obs))
        cv_sim = safe_div(np.std(sim), np.mean(sim))
        nse = round(
            1 - safe_div(np.sum((obs - sim) ** 2), np.sum((obs - np.mean(obs)) ** 2)), 4
        )
        kge09 = round(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), 4)
        kge12 = round(
            1
            - np.sqrt(
                (r - 1) ** 2 + (safe_div(cv_sim, cv_obs) - 1) ** 2 + (beta - 1) ** 2
            ),
            4,
        )
        rmse = round(np.sqrt(np.mean((obs - sim) ** 2)), 4)
        mae = round(np.mean(np.abs(obs - sim)), 4)
        pbias = round(100 * np.sum(obs - sim) / np.sum(obs), 2)
        r2 = round(r**2, 4)
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

    stats = [
        "NSE",
        "KGE (2009)",
        "KGE (2012)",
        "RMSE",
        "MAE",
        "PBIAS",
        "R²",
        "Pearson R",
        "Spearman R",
    ]
    return stats, _compute(obs, sim_raw), _compute(obs, sim_cor)


class ForecastVsObserved(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_forecast_vs_observed"
    label = "Geoglows Forecast Vs Observed"
    group = "TGF"
    description = "Validation dashboard: hydrograph, scatter, flow duration curve, and residuals for GeoGLOWS retrospective vs observed streamflow."
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

        parquet_path = os.path.join(STREAMFLOW_CACHE_DIR, f"{self.station_id}.parquet")

        if os.path.exists(parquet_path):
            self.send_update("Loading data from local cache...")
            merged_full = pd.read_parquet(parquet_path, engine="pyarrow")
        else:
            self.send_update("Cache miss — fetching from GeoGLOWS API...")
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

            merged_full = sim_df.join(cor_df, how="left").join(obs_df, how="left")

        # Subset to the requested display window and drop rows missing any series
        merged = merged_full.loc[start:end, ["Q_sim", "Q_cor", "Q_obs"]].dropna()
        dates = merged.index
        obs = merged["Q_obs"].values
        sim = merged["Q_sim"].values
        cor = merged["Q_cor"].values

        self.send_update("Computing metrics...")
        stat_names, raw_vals, cor_vals = _metrics_table(obs, sim, cor)

        # ── Build 2×2 chart figure (no table — injected as raw dict later) ──
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Hydrograph",
                "Scatter (Obs vs Sim)",
                "Flow Duration Curve",
                "Residuals",
            ),
            vertical_spacing=0.16,
            horizontal_spacing=0.08,
        )

        def _header(label):
            """Invisible trace used as a legend section header."""
            return go.Scatter(
                x=[None],
                y=[None],
                name=f"<b>{label}</b>",
                mode="markers",
                marker=dict(size=0, color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
            )

        # 1. Hydrograph
        fig.add_trace(_header("Hydrograph"), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=dates, y=obs, name="  Observed", line=dict(color="#1f77b4", width=1.5)
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sim,
                name="  Simulated (Raw)",
                line=dict(color="#ff7f0e", width=1.2, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cor,
                name="  Simulated (Corrected)",
                line=dict(color="#2ca02c", width=1.2, dash="dot"),
            ),
            row=1,
            col=1,
        )

        # 2. Scatter 1:1
        mv = max(obs.max(), sim.max(), cor.max())
        fig.add_trace(_header("Scatter"), row=1, col=2)
        fig.add_trace(
            go.Scatter(
                x=obs,
                y=sim,
                mode="markers",
                name="  Raw",
                marker=dict(color="#ff7f0e", size=4, opacity=0.5),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=obs,
                y=cor,
                mode="markers",
                name="  Corrected",
                marker=dict(color="#2ca02c", size=4, opacity=0.5),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[0, mv],
                y=[0, mv],
                name="1:1",
                line=dict(color="black", dash="dot", width=1),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Flow Duration Curve
        def fdc(arr):
            s = np.sort(arr[arr > 0])[::-1]
            return np.arange(1, len(s) + 1) / len(s) * 100, s

        exc_o, obs_s = fdc(obs)
        exc_s, sim_s = fdc(sim)
        exc_c, cor_s = fdc(cor)
        fig.add_trace(_header("Flow Duration Curve"), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=exc_o, y=obs_s, name="  Observed", line=dict(color="#1f77b4")),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exc_s,
                y=sim_s,
                name="  Simulated (Raw)",
                line=dict(color="#ff7f0e", dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exc_c,
                y=cor_s,
                name="  Simulated (Corrected)",
                line=dict(color="#2ca02c", dash="dot"),
            ),
            row=2,
            col=1,
        )

        # 4. Residuals
        fig.add_trace(_header("Residuals"), row=2, col=2)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sim - obs,
                name="  Simulated (Raw)",
                line=dict(color="#d62728", width=1, dash="dash"),
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cor - obs,
                name="  Simulated (Corrected)",
                line=dict(color="#9467bd", width=1, dash="dot"),
            ),
            row=2,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)

        fig.update_yaxes(title_text="Q (m³/s)", row=1, col=1)
        fig.update_xaxes(title_text="Observed Q (m³/s)", row=1, col=2)
        fig.update_yaxes(title_text="Simulated Q (m³/s)", row=1, col=2)
        fig.update_xaxes(title_text="Exceedance Probability (%)", row=2, col=1)
        fig.update_yaxes(title_text="Q (m³/s)", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Sim − Obs (m³/s)", row=2, col=2)
        fig.update_xaxes(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
            type="date",
            row=1,
            col=1,
        )
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            autosize=True,
            margin=dict(l=50, r=10, t=40, b=60),
            legend=dict(orientation="h", y=-0.1, x=0.36, xanchor="center"),
        )

        # Serialize charts, then compress x-domains to 75% width and inject table
        fig_dict = json.loads(fig.to_json())
        layout = fig_dict["layout"]

        # Scale all x-axis domains to [0, 0.72] to leave room for the table
        for key, val in layout.items():
            if isinstance(val, dict) and "domain" in val and key.startswith("xaxis"):
                val["domain"] = [d * 0.72 for d in val["domain"]]
        # Also shift subplot title annotations
        for ann in layout.get("annotations", []):
            if "x" in ann:
                ann["x"] = ann["x"] * 0.72

        # Inject table as a plain dict — bypasses plotly's trace validation
        fig_dict["data"].append(
            {
                "type": "table",
                "domain": {"x": [0.76, 1.0], "y": [0.15, 0.85]},
                "header": {
                    "values": [
                        "<b>Metric</b>",
                        "<b>Simulated (Raw)</b>",
                        "<b>Simulated (Corrected)</b>",
                    ],
                    "fill": {"color": "#f0f0f0"},
                    "align": "left",
                    "font": {"size": 12},
                },
                "cells": {
                    "values": [stat_names, raw_vals, cor_vals],
                    "align": "left",
                    "font": {"size": 11},
                    "height": 28,
                },
            }
        )

        return {
            "data": fig_dict["data"],
            "layout": layout,
            "config": {"displayModeBar": True},
        }


# ---------------------------------------------------------------------------
# Offline cache builder — run once to pre-fetch all stations
# ---------------------------------------------------------------------------


def build_streamflow_cache(
    stations: dict,
    output_dir: str = None,
) -> None:
    """Fetch the full GEOGloWS retrospective for every station, merge with
    observed data, apply bias correction, and save one Parquet file per
    station.  Parquet is used because it loads in milliseconds and supports
    fast column / date-range slicing without reading the whole file.

    Output schema per file (index = date):
        Q_sim  — raw GEOGloWS retrospective (all available years)
        Q_cor  — bias-corrected retrospective (all available years)
        Q_obs  — gauge observations (NaN outside the observation window)

    Args:
        stations: dict mapping station_id (CSV filename stem, e.g. "240")
                  to COMID (int), e.g. {"240": 441156246, ...}
        output_dir: directory to write .parquet files; defaults to a
                    "streamflow_cache" folder next to this file.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "streamflow_cache")
    os.makedirs(output_dir, exist_ok=True)

    _ensure_cache()

    total = len(stations)
    for i, (station_id, comid) in enumerate(stations.items(), 1):
        print(f"[{i}/{total}] {station_id}  (COMID={comid}) ... ", end="", flush=True)

        # ── Observed ───────────────────────────────────────────────────
        obs_df = load_observed(station_id)
        if obs_df is None:
            print("SKIP — observed CSV not found")
            continue

        # ── Full retrospective via API ─────────────────────────────────
        try:
            retro = requests.get(
                f"https://geoglows.ecmwf.int/api/v2/retrospectivedaily/{comid}?format=json",
                timeout=120,
            ).json()
        except Exception as exc:
            print(f"SKIP — API error: {exc}")
            continue

        flow_key = next(k for k in retro if k != "datetime")
        sim_df = pd.DataFrame(
            {"Q_sim": retro[flow_key]},
            index=pd.to_datetime(retro["datetime"]),
        )
        sim_df[sim_df < 0] = 0
        sim_df.index = sim_df.index.tz_localize(None)

        # ── Bias correction (fit on overlap, apply to full series) ─────
        try:
            cor_df = geoglows.bias.correct_historical(sim_df, obs_df)
            cor_df.columns = ["Q_cor"]
        except Exception as exc:
            print(f"SKIP — bias correction error: {exc}")
            continue

        # ── Merge: keep all retrospective dates, obs as NaN outside window
        merged = sim_df.join(cor_df, how="left").join(obs_df, how="left")
        merged = merged[["Q_sim", "Q_cor", "Q_obs"]]

        # ── Save ───────────────────────────────────────────────────────
        out_path = os.path.join(output_dir, f"{station_id}.parquet")
        merged.to_parquet(out_path, engine="pyarrow", compression="snappy")
        overlap = merged["Q_obs"].notna().sum()
        print(f"{len(merged):,} rows  ({overlap:,} observed)  → {out_path}")

    print(f"\nDone. Parquet files written to: {output_dir}")


if __name__ == "__main__":
    import json as _json

    _geojson_path = os.path.join(
        os.path.dirname(__file__), "static", "NepalStations.geojson"
    )
    with open(_geojson_path) as f:
        _gj = _json.load(f)

    STATIONS = {
        feat["properties"]["samplingFeatureCode"]: int(feat["properties"]["COMID_v2"])
        for feat in _gj["features"]
    }

    print(f"Loaded {len(STATIONS)} stations from GeoJSON")
    build_streamflow_cache(STATIONS)
