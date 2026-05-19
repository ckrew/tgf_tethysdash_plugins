import json

import plotly.graph_objects as go
from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin

from tethysdash_plugins._icimod import icimod_get

# Base RGB colors for each return period (interpolated toward white at low %)
_PERIODS = ["two", "ten", "twenty", "fifty", "max"]

_BASE_COLORS = {
    "two":    None,            # gray — fixed, no gradient
    "ten":    (249, 255, 0),   # yellow  #F9FF00
    "twenty": (225, 149, 6),   # orange  #e19506
    "fifty":  (255, 0,   0),   # red     #FF0000
    "max":    (176, 0,   255), # purple  #B000FF
}

_ROW_LABELS = {
    "two":    "Percent Exceedance (2-yr)",
    "ten":    "Percent Exceedance (10-yr)",
    "twenty": "Percent Exceedance (20-yr)",
    "fifty":  "Percent Exceedance (50-yr)",
    "max":    "Percent Exceedance (100-yr)",
}

_GRAY = "rgb(220,220,220)"
_WHITE = "rgb(255,255,255)"


def _interpolate(value_str: str, base_rgb: tuple) -> str:
    """Blend white → base_rgb proportionally to value (0–100)."""
    v = int(value_str) / 100.0
    r = int(255 + (base_rgb[0] - 255) * v)
    g = int(255 + (base_rgb[1] - 255) * v)
    b = int(255 + (base_rgb[2] - 255) * v)
    return f"rgb({r},{g},{b})"


class ExceedanceProbability(TethysDashPlugin):
    type = "plotly"
    name = "geoglows_exceedance_probability"
    label = "Geoglows Exceedance Probability"
    group = "TGF"
    description = "Daily exceedance probability table showing the % of ensemble members exceeding each return-period threshold, colour-coded by probability."
    tags = ["Geoglows", "Forecast", "Nepal"]
    args = {"comid": "text"}

    def run(self):
        self.send_update("Fetching exceedance probability data...")
        data = icimod_get(f"/apps/streamflownepal/forecastpercent/?comid={self.comid}")

        dates = data["percdates"]
        periods = _PERIODS

        # ── Build table columns ────────────────────────────────────────────
        # Column 0: row labels; Columns 1…n: one per date
        row_labels = [_ROW_LABELS[p] for p in periods]

        cell_values = [row_labels]          # first column = return period names
        cell_colors = [                     # first column = fixed muted base color
            [
                _GRAY if _BASE_COLORS[p] is None
                else _interpolate("30", _BASE_COLORS[p])  # subtle tint for label col
                for p in periods
            ]
        ]

        for i, date in enumerate(dates):
            col_vals = [data[p][i] for p in periods]
            col_colors = []
            for p, v in zip(periods, col_vals):
                base = _BASE_COLORS[p]
                col_colors.append(_GRAY if base is None else _interpolate(v, base))
            cell_values.append(col_vals)
            cell_colors.append(col_colors)

        # ── Plotly table ───────────────────────────────────────────────────
        n_date_cols = len(dates)
        fig = go.Figure(go.Table(
            columnwidth=[2] + [1] * n_date_cols,
            header=dict(
                values=["<b>Dates</b>"] + [f"<b>{d}</b>" for d in dates],
                fill_color="#2c3e50",
                font=dict(color="white", size=12),
                align="center",
                height=30,
                line=dict(color="gray", width=1),
            ),
            cells=dict(
                values=cell_values,
                fill_color=cell_colors,
                align="center",
                font=dict(size=12),
                height=28,
                line=dict(color="gray", width=1),
            ),
        ))

        fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10),
            template="plotly_white",
        )

        fig_dict = json.loads(fig.to_json())
        return {
            "data": fig_dict["data"],
            "layout": fig_dict["layout"],
            "config": {"displayModeBar": False, "responsive": True},
        }
