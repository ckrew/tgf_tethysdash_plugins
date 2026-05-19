"""Shared data-loading utility for TGF TethysDash plugins.

Provides cached access to GEOGloWS retrospective streamflow and locally-
downloaded observed gauge data.  All plugins import from this module so that
the Drive folder is only downloaded once per process and Parquet files are
reused whenever available.
"""

import os

import gdown
import geoglows
import numpy as np
import pandas as pd
import requests

OBSERVED_FOLDER_ID = "1JJ8gQWUdpH-0QAIUdgXVHF5TEw2TyHcd"
OBSERVED_CACHE_DIR = f"/tmp/tgf_observed_{OBSERVED_FOLDER_ID}"
STREAMFLOW_CACHE_DIR = os.path.join(os.path.dirname(__file__), "streamflow_cache")


def _ensure_obs_cache() -> None:
    """Download the observed folder from Drive once; subsequent calls are no-ops."""
    if not os.path.isdir(OBSERVED_CACHE_DIR) or not os.listdir(OBSERVED_CACHE_DIR):
        os.makedirs(OBSERVED_CACHE_DIR, exist_ok=True)
        gdown.download_folder(
            id=OBSERVED_FOLDER_ID, output=OBSERVED_CACHE_DIR, quiet=True
        )


def _load_observed_csv(station_id: str) -> pd.DataFrame | None:
    """Find ``{station_id}_Q.csv`` in OBSERVED_CACHE_DIR.

    Returns a DataFrame with a tz-naive datetime index and a single column
    named ``Q_obs``, or *None* if no matching file exists.
    """
    _ensure_obs_cache()
    filename = f"{station_id}_Q"
    for fname in os.listdir(OBSERVED_CACHE_DIR):
        if os.path.splitext(fname)[0] == filename:
            df = pd.read_csv(
                os.path.join(OBSERVED_CACHE_DIR, fname),
                index_col=0,
                parse_dates=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce")
            df.columns = ["Q_obs"]
            df[df < 0] = 0
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df.dropna()
    return None


def load_station_data(
    station_id: str,
    comid: str,
    send_update=None,
) -> pd.DataFrame:
    """Return a merged DataFrame with columns Q_sim, Q_cor, Q_obs.

    Checks for a pre-built Parquet file at
    ``streamflow_cache/{station_id}.parquet`` first (fast path).  If the file
    does not exist, fetches the full GEOGloWS retrospective, loads the
    observed CSV, applies bias correction, and returns the left-joined result
    so that all retrospective dates are present (Q_obs is NaN outside the
    observation window).

    Args:
        station_id: Gauge identifier, e.g. ``"240"``.
        comid: GEOGloWS COMID as a string or int.
        send_update: Optional callable that accepts a single string message
            for progress reporting (e.g. ``self.send_update``).
    """

    def _notify(msg: str) -> None:
        if callable(send_update):
            send_update(msg)

    parquet_path = os.path.join(STREAMFLOW_CACHE_DIR, f"{station_id}.parquet")

    if os.path.exists(parquet_path):
        _notify("Loading data from local cache...")
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        return df

    # ── Cache miss: fetch from API ─────────────────────────────────────────
    _notify("Cache miss — fetching from GeoGLOWS API...")
    retro = requests.get(
        f"https://geoglows.ecmwf.int/api/v2/retrospectivedaily/{comid}?format=json",
        timeout=120,
    ).json()

    flow_key = next(k for k in retro if k != "datetime")
    sim_df = pd.DataFrame(
        {"Q_sim": retro[flow_key]},
        index=pd.to_datetime(retro["datetime"]),
    )
    sim_df[sim_df < 0] = 0
    sim_df.index = sim_df.index.tz_localize(None)

    _notify("Loading observed data...")
    obs_df = _load_observed_csv(station_id)

    _notify("Applying bias correction...")
    cor_df = geoglows.bias.correct_historical(sim_df, obs_df)
    cor_df.columns = ["Q_cor"]

    # Left join so all retrospective dates are preserved; Q_obs is NaN outside
    # the observation window.
    merged = sim_df.join(cor_df, how="left").join(obs_df, how="left")
    return merged[["Q_sim", "Q_cor", "Q_obs"]]
