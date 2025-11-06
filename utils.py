# utils.py
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Price area -> (city, lat, lon)
# -----------------------------
PRICE_AREA_COORDS: Dict[str, Dict[str, float | str]] = {
    "NO1": {"city": "Oslo",         "latitude": 59.9139,  "longitude": 10.7522},
    "NO2": {"city": "Kristiansand", "latitude": 58.1467,  "longitude": 7.9956},
    "NO3": {"city": "Trondheim",    "latitude": 63.4305,  "longitude": 10.3951},
    "NO4": {"city": "Tromsø",       "latitude": 69.6492,  "longitude": 18.9553},
    "NO5": {"city": "Bergen",       "latitude": 60.39299, "longitude": 5.32415},
}


def get_area_coords(price_area: str) -> Tuple[float, float, str]:
    """
    Returns (lat, lon, city) for the given price area (NO1..NO5).
    Raises ValueError if the area is unknown.
    """
    pa = price_area.upper().strip()
    if pa not in PRICE_AREA_COORDS:
        raise ValueError(f"Unknown price area: {price_area}. Valid: {list(PRICE_AREA_COORDS.keys())}")
    info = PRICE_AREA_COORDS[pa]
    return float(info["latitude"]), float(info["longitude"]), str(info["city"])


# -----------------------------
# Open-Meteo (ERA5, historical)
# -----------------------------
DEFAULT_HOURLY: Tuple[str, ...] = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
)


@st.cache_data(show_spinner=False)
def download_open_meteo(
    price_area: Optional[str] = None,
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    start_date: str = "2019-01-01",
    end_date: str = "2019-12-31",
    hourly: Sequence[str] = DEFAULT_HOURLY,
    timezone: str = "Europe/Oslo",
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetches historical reanalysis data (ERA5) from the Open-Meteo Archive API for a given
    location (specified via price area or lat/lon) and the time range [start_date, end_date].

    Parameters
    ----------
    price_area : NO1..NO5 (optional). If set, overrides lat/lon.
    lat, lon   : coordinates (optional if price_area is set).
    start_date : 'YYYY-MM-DD'
    end_date   : 'YYYY-MM-DD'
    hourly     : list of variables (see DEFAULT_HOURLY).
    timezone   : e.g., 'Europe/Oslo' so the time is local Norwegian time.

    Returns
    -------
    DataFrame with columns ['time', <selected hourly variables>].
    """
    if price_area:
        lat, lon, _ = get_area_coords(price_area)
    if lat is None or lon is None:
        raise ValueError("Must specify either price_area or (lat, lon).")

    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly),
        "timezone": timezone,
    }

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if "hourly" not in payload or "time" not in payload["hourly"]:
        # Empty response / unknown format -> return an empty DF with correct column names
        cols = ["time"] + list(hourly)
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(payload["hourly"])
    # Ensure all requested columns exist (may be missing if the API had no data)
    for col in hourly:
        if col not in df.columns:
            df[col] = pd.NA

    # Parse time
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df[["time", *hourly]]


# -------------------------------------------------
# (Existing) CSV loader – keep as fallback
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_name: str = "/Users/a.h.sheikh/Desktop/IND320_Git_Job/IND320_1_Project/Innlevering og IPYNB/open-meteo-subset.csv") -> pd.DataFrame:
    """
    Loads a CSV if it exists in a few known locations; otherwise,
    safe demo data is generated so the app still works.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent / csv_name,                 # project root
        here.parent / "data" / csv_name,       # /data
        here.parent / "pages" / csv_name,      # /pages
        Path.cwd() / csv_name,                 # working directory in the cloud
        Path(csv_name),                        # relative
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Find and parse a time-like column if possible
            date_like = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])]
            if date_like:
                dcol = date_like[0]
                df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
                if df[dcol].notna().any():
                    df = df.sort_values(dcol)
                    df["month"] = df[dcol].dt.to_period("M").astype(str)
            else:
                df["month"] = "Unknown"
            return df

    # Fallback: generate demo data
    idx = pd.date_range("2021-01-01", periods=24 * 14, freq="H")
    s = pd.Series(range(len(idx)), dtype="float64")
    df = pd.DataFrame({
        "time": idx,
        "temperature": ((5.0 + 0.3 * s) % 10) + 2,
        "wind":       ((0.4 * s) % 8) + 1,
        "precip":     ((0.1 * s) % 5),
    })
    df["month"] = df["time"].dt.to_period("M").astype(str)
    return df


# -------------------------------------------------
# Small helpers for Streamlit pages (selection, state)
# -------------------------------------------------
def get_selected_price_area(default: str = "NO1") -> str:
    """
    Get price area from session state (set on page 2), with fallback.
    """
    pa = st.session_state.get("price_area", default)
    return str(pa).upper().strip()

# ---- Price area table helper (used by pages/3_Meteorology.py) ----
def get_price_area_table():
    import pandas as pd
    # Try to fetch from Mongo via utils_elhub (if available)
    try:
        from utils_elhub import list_price_areas
        areas = list_price_areas()
        # Normalize to a DataFrame with column name 'price_area'
        if isinstance(areas, (list, tuple, set)):
            return pd.DataFrame({"price_area": list(areas)})
        if hasattr(areas, "to_frame"):
            # e.g., a Series
            df = areas.to_frame(name="price_area")
            if "price_area" not in df.columns:
                df.columns = ["price_area"]
            return df
        if hasattr(areas, "columns"):
            df = areas
            if "price_area" not in df.columns:
                # try to guess the first column
                first = df.columns[0]
                df = df.rename(columns={first: "price_area"})[["price_area"]]
            return df
    except Exception:
        pass
    # Fallback: static list
    return pd.DataFrame({"price_area": ["NO1", "NO2", "NO3", "NO4", "NO5"]})
