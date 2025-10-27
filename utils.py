# utils.py
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Prisområde -> (by, lat, lon)
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
    Returnerer (lat, lon, city) for gitt prisområde (NO1..NO5).
    Kaster ValueError hvis ukjent område.
    """
    pa = price_area.upper().strip()
    if pa not in PRICE_AREA_COORDS:
        raise ValueError(f"Ukjent prisområde: {price_area}. Gyldig: {list(PRICE_AREA_COORDS.keys())}")
    info = PRICE_AREA_COORDS[pa]
    return float(info["latitude"]), float(info["longitude"]), str(info["city"])


# -----------------------------
# Open-Meteo (ERA5, historisk)
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
    Henter historiske reanalyse-data (ERA5) fra Open-Meteo Archive API for gitt lokasjon
    (angitt via prisområde eller lat/lon) og tidsrom [start_date, end_date].

    Parametre
    ---------
    price_area : NO1..NO5 (valgfritt). Hvis satt, overstyrer lat/lon.
    lat, lon   : koordinater (valgfritt hvis price_area settes).
    start_date : 'YYYY-MM-DD'
    end_date   : 'YYYY-MM-DD'
    hourly     : liste over variabler (se DEFAULT_HOURLY).
    timezone   : f.eks. 'Europe/Oslo' så tid blir lokal norsk tid.

    Returnerer
    ----------
    DataFrame med kolonnene ['time', <valgte hourly-variabler>].
    """
    if price_area:
        lat, lon, _ = get_area_coords(price_area)
    if lat is None or lon is None:
        raise ValueError("Må angi enten price_area eller (lat, lon).")

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
        # Tomt svar / ukjent format -> returnér tom DF med riktige kolonnenavn
        cols = ["time"] + list(hourly)
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(payload["hourly"])
    # Sørg for at alle ønskede kolonner finnes (kan mangle hvis API ikke hadde data)
    for col in hourly:
        if col not in df.columns:
            df[col] = pd.NA

    # Parse tid
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df[["time", *hourly]]


# -------------------------------------------------
# (Eksisterende) CSV-loader – behold som fallback
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_name: str = "/Users/a.h.sheikh/Desktop/IND320_Git_Job/IND320_1_Project/Innlevering og IPYNB/open-meteo-subset.csv") -> pd.DataFrame:
    """
    Laster en CSV hvis den finnes et par kjente steder; hvis ikke
    genereres trygge demo-data slik at appen fortsatt fungerer.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent / csv_name,                 # prosjektrot
        here.parent / "data" / csv_name,       # /data
        here.parent / "pages" / csv_name,      # /pages
        Path.cwd() / csv_name,                 # arbeidskatalog i skyen
        Path(csv_name),                        # relativt
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Finn og parse en tidskolonne om mulig
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

    # Fallback: generér demo-data
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
# Små helpers for Streamlit-sider (valg, state)
# -------------------------------------------------
def get_selected_price_area(default: str = "NO1") -> str:
    """
    Hent prisområde fra session state (satt på side 2), med fallback.
    """
    pa = st.session_state.get("price_area", default)
    return str(pa).upper().strip()

# ---- Price area table helper (used by pages/3_Meteorology.py) ----
def get_price_area_table():
    import pandas as pd
    # Prøv å hente fra Mongo via utils_elhub (om tilgjengelig)
    try:
        from utils_elhub import list_price_areas
        areas = list_price_areas()
        # Normaliser til DataFrame med kolonnenavn 'price_area'
        if isinstance(areas, (list, tuple, set)):
            return pd.DataFrame({"price_area": list(areas)})
        if hasattr(areas, "to_frame"):
            # f.eks. en Series
            df = areas.to_frame(name="price_area")
            if "price_area" not in df.columns:
                df.columns = ["price_area"]
            return df
        if hasattr(areas, "columns"):
            df = areas
            if "price_area" not in df.columns:
                # prøv å gjette første kolonne
                first = df.columns[0]
                df = df.rename(columns={first: "price_area"})[["price_area"]]
            return df
    except Exception:
        pass
    # Fallback: statisk liste
    return pd.DataFrame({"price_area": ["NO1", "NO2", "NO3", "NO4", "NO5"]})
