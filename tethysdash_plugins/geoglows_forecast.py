import json

import plotly.graph_objects as go
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin

from tethysdash_plugins._icimod import icimod_get


class GeoglowsForecast(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_forecast"
    label = "Geoglows Forecast"
    group = "TGF"
    description = "GEOGloWS ensemble forecast: HRES, ensemble mean/min/max, and return-period flood thresholds."
    tags = ["Geoglows", "Forecast", "Nepal"]
    args = {"comid": "text"}

    def run(self):
        self.send_update("Fetching forecast data...")
        data = icimod_get(f"/apps/streamflownepal/chart/?stID={self.comid}")

        dates = data["dates"]
        hres_dates = data["hres_dates"]
        mean_values = data["mean_values"]
        hres_values = data["hres_values"]
        min_values = data["min_values"]
        max_values = data["max_values"]

        r2 = float(data["return_2"])
        r5 = float(data["return_5"])
        r10 = float(data["return_10"])
        r20 = float(data["return_20"])
        r50 = float(data["return_50"])
        r_max = float(data["return_max"])

        fig = go.Figure()

        # ── Return-period horizontal bands (drawn first so they sit behind) ──
        bands = [
            (r5,  r10,   "rgba(249,255,0,0.25)",   "5–10 yr"),
            (r10, r20,   "rgba(225,149,6,0.25)",   "10–20 yr"),
            (r20, r50,   "rgba(255,0,0,0.25)",     "20–50 yr"),
            (r50, r_max, "rgba(176,0,255,0.25)",   "50+ yr"),
        ]
        for y0, y1, color, label in bands:
            fig.add_hrect(
                y0=y0,
                y1=y1,
                fillcolor=color,
                line_width=0,
                layer="below",
            )

        # ── Ensemble envelope (min → max shaded green) ────────────────────
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=min_values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="Min",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=max_values,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0,180,0,0.2)",
                name="Ensemble Range",
                hoverinfo="skip",
            )
        )

        # ── Ensemble mean ─────────────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=mean_values,
                mode="lines",
                name="Ensemble Mean",
                line=dict(color="#1f77b4", width=1.5),
            )
        )

        # ── HRES (deterministic high-res) ─────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=hres_dates,
                y=hres_values,
                mode="lines",
                name="HRES",
                line=dict(color="black", width=1.5),
            )
        )

        # ── Return-period reference lines ─────────────────────────────────
        rp_lines = [
            (r2,  "#888888", "2 yr"),
            (r5,  "#F9FF00", "5 yr"),
            (r10, "#e19506", "10 yr"),
            (r20, "#FF0000", "20 yr"),
            (r50, "#B000FF", "50 yr"),
        ]
        for val, color, label in rp_lines:
            fig.add_hline(
                y=val,
                line=dict(color=color, width=1, dash="dot"),
                annotation_text=label,
                annotation_position="top left",
                annotation_font=dict(size=9, color=color),
            )

        fig.update_layout(
            title=dict(
                text=f"Forecast at Date (Time Zone: UTC) {data['runDate']}, River ID: {self.comid}",
                x=0.5,
                xanchor="center",
            ),
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=50),
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(
                title="Date",
                type="date",
            ),
            yaxis=dict(
                title="Streamflow (m³/s)",
                range=[0, max(max_values) * 1.15],
            ),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": True, "responsive": True},
        }
