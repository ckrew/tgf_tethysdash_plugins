import requests
from tethysapp.tethysdash.exceptions import VisualizationError

_BASE = "https://tethys.icimod.org"

_HEADERS = {
    "Referer": f"{_BASE}/apps/streamflownepal/",
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    ),
}


def icimod_get(path: str, timeout: int = 30) -> dict:
    url = f"{_BASE}{path}"
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)

    if "application/json" not in resp.headers.get("Content-Type", ""):
        raise VisualizationError(
            f"ICIMOD API returned non-JSON (HTTP {resp.status_code}) for {url}. "
            "The endpoint may be unavailable or the COMID is not recognised."
        )

    return resp.json()
