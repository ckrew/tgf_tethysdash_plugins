import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tethysapp.tethysdash.plugin_helpers import (
    LayerConfigurationBuilder,
    TethysDashPlugin,
)

from tethysdash_plugins._cache import STREAMFLOW_CACHE_DIR

_GEOJSON_PATH = os.path.join(
    os.path.dirname(__file__), "static", "NepalStations.geojson"
)


def _perf_color(kge12: float) -> str:
    if kge12 >= 0.5:
        return "#16a34a"
    if kge12 >= 0.3:
        return "#f59e0b"
    return "#ef4444"


def _compute(obs, sim) -> dict | None:
    from scipy.stats import spearmanr
    if np.std(obs) == 0 or np.std(sim) == 0:
        return None
    r = float(pearsonr(obs, sim)[0])
    alpha = float(np.std(sim) / np.std(obs))
    beta = float(np.mean(sim) / np.mean(obs))
    cv_obs = float(np.std(obs) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    cv_sim = float(np.std(sim) / np.mean(sim)) if np.mean(sim) != 0 else np.nan
    return {
        "nse":        round(float(1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)), 4),
        "kge09":      round(float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)), 4),
        "kge12":      round(float(1 - np.sqrt((r - 1) ** 2 + ((cv_sim / cv_obs) - 1) ** 2 + (beta - 1) ** 2)), 4),
        "rmse":       round(float(np.sqrt(np.mean((obs - sim) ** 2))), 4),
        "mae":        round(float(np.mean(np.abs(obs - sim))), 4),
        "pbias":      round(float(100 * np.sum(obs - sim) / np.sum(obs)), 2),
        "r2":         round(r ** 2, 4),
        "pearson_r":  round(r, 4),
        "spearman_r": round(float(spearmanr(obs, sim)[0]), 4),
    }


_STYLE_RULES = {
    "rules": [
        {
            "conditionField": "color",
            "conditionType": "=",
            "conditionValue": "#16a34a",
            "geometryType": "point",
            "name": "KGE ≥ 0.5 (Good)",
            "fill": "#16a34a",
            "shape": "circle",
            "size": "6",
        },
        {
            "conditionField": "color",
            "conditionType": "=",
            "conditionValue": "#f59e0b",
            "geometryType": "point",
            "name": "KGE 0.3–0.5 (Weak)",
            "fill": "#f59e0b",
            "shape": "circle",
            "size": "6",
        },
        {
            "conditionField": "color",
            "conditionType": "=",
            "conditionValue": "#ef4444",
            "geometryType": "point",
            "name": "KGE < 0.3 (Poor)",
            "fill": "#ef4444",
            "shape": "circle",
            "size": "6",
        },
        {
            "conditionField": "color",
            "conditionType": "=",
            "conditionValue": "#aaaaaa",
            "geometryType": "point",
            "name": "No Data",
            "fill": "#aaaaaa",
            "shape": "circle",
            "size": "6",
        },
    ],
    "default": {"point": {"stroke": "#000000", "strokeWidth": "1"}},
}


class NepalPerformanceLayer(TethysDashPlugin):
    type = "map_layer"
    name = "geoglows_nepal_performance_layer"
    label = "Nepal Station Performance"
    group = "TGF"
    description = "Nepal gauge stations colored by GEOGloWS model performance (KGE 2012, full history)."
    tags = ["Geoglows", "Bias", "Metrics", "Nepal", "map layer"]
    dynamic_map_layer = True
    args = {"start_date": "date", "end_date": "date"}

    def run(self):
        layer_name = self.label
        builder = LayerConfigurationBuilder(layer_name, "GeoJSON")
        builder.set_plugin_source(self.name, self.args)
        builder.set_style(_STYLE_RULES)
        builder.set_legend("default")
        builder.set_queryable(True)

        builder.add_attribute_alias("name", "Station", layer_name)
        builder.add_attribute_alias("River", "River", layer_name)
        builder.add_attribute_alias("state", "State", layer_name)
        builder.add_attribute_alias("elevation_m", "Elevation (m)", layer_name)
        builder.add_attribute_alias("nse",        "NSE",                layer_name)
        builder.add_attribute_alias("kge09",      "KGE (2009)",         layer_name)
        builder.add_attribute_alias("kge12",      "KGE (2012)",         layer_name)
        builder.add_attribute_alias("rmse",       "RMSE (m³/s)",        layer_name)
        builder.add_attribute_alias("mae",        "MAE (m³/s)",         layer_name)
        builder.add_attribute_alias("pbias",      "PBIAS (%)",          layer_name)
        builder.add_attribute_alias("r2",         "R²",                 layer_name)
        builder.add_attribute_alias("pearson_r",  "Pearson R",          layer_name)
        builder.add_attribute_alias("spearman_r", "Spearman R",         layer_name)
        builder.add_attribute_alias("n_days",     "Observation Days",   layer_name)
        builder.add_attribute_alias("samplingFeatureCode", "Station ID", layer_name)
        builder.add_attribute_variable(
            "samplingFeatureCode", "Selected Station", layer_name
        )

        for field in (
            "uid",
            "country_id",
            "samplingFeatureType",
            "elevationDatum",
            "matching_column",
            "matching_column_v2",
            "GEOGloWS_v2_vpu",
            "color",
            "COMID_v2",
            "description",
            "siteType",
        ):
            builder.omit_popup_attribute(field, layer_name)

        return builder.build()

    def fetch_features(self):
        def _to_naive(ts):
            t = pd.Timestamp(ts)
            return t.tz_localize(None) if t.tzinfo is None else t.tz_convert(None)

        start = _to_naive(self.start_date)
        end = _to_naive(self.end_date)

        self.send_update("Computing station performance...")
        with open(_GEOJSON_PATH) as f:
            collection = json.load(f)

        for feature in collection["features"]:
            sid = feature["properties"]["samplingFeatureCode"]
            path = os.path.join(STREAMFLOW_CACHE_DIR, f"{sid}.parquet")

            color = "#aaaaaa"
            m = {}
            n_days = None

            if os.path.exists(path):
                df = pd.read_parquet(path, engine="pyarrow").loc[start:end].dropna()
                if len(df) >= 30:
                    result = _compute(df["Q_obs"].values, df["Q_sim"].values)
                    if result:
                        m = result
                        n_days = len(df)
                        color = _perf_color(m["kge12"])

            feature["properties"]["color"]      = color
            feature["properties"]["n_days"]     = n_days
            feature["properties"]["nse"]        = m.get("nse")
            feature["properties"]["kge09"]      = m.get("kge09")
            feature["properties"]["kge12"]      = m.get("kge12")
            feature["properties"]["rmse"]       = m.get("rmse")
            feature["properties"]["mae"]        = m.get("mae")
            feature["properties"]["pbias"]      = m.get("pbias")
            feature["properties"]["r2"]         = m.get("r2")
            feature["properties"]["pearson_r"]  = m.get("pearson_r")
            feature["properties"]["spearman_r"] = m.get("spearman_r")

        collection["crs"] = {"type": "name", "properties": {"name": "EPSG:4326"}}
        self.send_update("Done.", percentage_complete=100)
        return collection
