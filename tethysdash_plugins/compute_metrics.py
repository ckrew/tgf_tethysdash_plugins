from tethysapp.tethysdash.plugin_helpers import TethysDashPlugin
from scipy.stats import pearsonr, spearmanr
import numpy as np


def classify(value: float) -> str:
    """Rate a metric value: Very Good / Good / Acceptable / Weak."""
    if value >= 0.75:
        return " Very Good"
    if value >= 0.65:
        return " Good"
    if value >= 0.50:
        return " Acceptable"
    return " Weak"


def compute_metrics(obs: list, sim: list) -> dict:
    """Return a dict of all performance metrics."""

    def safe_div(a: float, b: float) -> float:
        return a / b if b != 0 else np.nan

    def pearson_r(o: list, s: list) -> float:
        if np.std(o) == 0 or np.std(s) == 0:
            return np.nan
        return pearsonr(o, s)[0]

    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    r = pearson_r(obs, sim)
    alpha = safe_div(np.std(sim), np.std(obs))
    beta = safe_div(np.mean(sim), np.mean(obs))
    cv_obs = safe_div(np.std(obs), np.mean(obs))
    cv_sim = safe_div(np.std(sim), np.mean(sim))

    nse = round(
        1 - safe_div(np.sum((obs - sim) ** 2), np.sum((obs - np.mean(obs)) ** 2)), 4
    )
    kge_2009 = round(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), 4)
    kge_2012 = round(
        1
        - np.sqrt((r - 1) ** 2 + (safe_div(cv_sim, cv_obs) - 1) ** 2 + (beta - 1) ** 2),
        4,
    )
    rmse = round(np.sqrt(np.mean((obs - sim) ** 2)), 4)
    mae = round(np.mean(np.abs(obs - sim)), 4)
    pbias = round(100 * np.sum(obs - sim) / np.sum(obs), 2)
    r_squared = round(r**2, 4)
    pearson_r_value = round(r, 4)
    spearman_r_value = round(spearmanr(obs, sim)[0], 4)

    return [
        {
            "Stat": "NSE",
            "Value": f"{nse} → {classify(nse)}",
        },
        {
            "Stat": "KGE (2009)",
            "Value": f"{kge_2009} → {classify(kge_2009)}",
        },
        {
            "Stat": "KGE (2012)",
            "Value": f"{kge_2012} → {classify(kge_2012)}",
        },
        {"Stat": "RMSE", "Value": f"{rmse} m³/s"},
        {"Stat": "MAE", "Value": f"{mae} m³/s"},
        {"Stat": "PBIAS", "Value": f"{pbias}%"},
        {"Stat": "R²", "Value": f"{r_squared}"},
        {"Stat": "Pearson R", "Value": f"{pearson_r_value}"},
        {"Stat": "Spearman R", "Value": f"{spearman_r_value}"},
    ]


class ComputeMetrics(TethysDashPlugin):
    type = "table"
    name = "geoglows_forecast_metrics"
    label = "Geoglows Forecast Metrics"
    group = "TGF"
    description = "Computes metrics for the latest geoglows forecasts for a specified COMID. Metrics include NSE, KGE, RMSE, MAE, PBIAS, R², Pearson R, and Spearman R."
    tags = ["Geoglows", "Bias", "Metrics", "Nepal"]
    args = {"comid": "text"}

    def run(self):
        return {
            "title": "Geoglows Forecast Metrics",
            "subtitle": self.comid,
            "data": compute_metrics(obs=[1, 2, 3], sim=[1.1, 1.9, 3.2]),
        }
