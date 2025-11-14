# pages/6_Snow_Drift.py

import datetime as dt
from math import pi

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Snow drift & wind rose",
    page_icon="❄️",
    layout="wide",
)

st.title("Snow drift calculation and wind rose (ERA5 / Open-Meteo)")

# -------------------------------------------------------------------
# 1. Get coordinate from map page
# -------------------------------------------------------------------
coord = st.session_state.get("map_coord", None)

if coord is None:
    st.warning(
        "No coordinate stored from the map page.\n\n"
        "Go to **'Map and Energy Statistics – Norwegian Price Areas (NO1–NO5)'** "
        "and click on the map first. Then come back here."
    )
    st.stop()

lat, lon = coord
st.info(f"Using coordinate from map page: **lat = {lat:.4f}**, **lon = {lon:.4f}**")

# -------------------------------------------------------------------
# 2. Helper: fetch ERA5 hourly data from Open-Meteo
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_era5_hourly(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly ERA5 reanalysis from Open-Meteo for given coordinate and date range.
    Variables: temperature, snowfall, wind speed & direction at 10m.
    """
    base_url = "https://archive-api.open-meteo.com/v1/era5"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "snowfall",
            "windspeed_10m",
            "winddirection_10m",
        ],
        "timezone": "Europe/Oslo",
    }

    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise RuntimeError("No 'hourly' field returned from Open-Meteo API.")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


# -------------------------------------------------------------------
# 3. Sector and snow drift helpers (simplified Tabler-style model)
# -------------------------------------------------------------------
def sector_index(direction_deg: float, n_sectors: int = 16) -> int:
    """
    Map wind direction in degrees [0, 360) to sector index 0..n_sectors-1.
    Sectors are 360/n_sectors degrees wide, centered on N, NNE, ...
    """
    width = 360.0 / n_sectors
    # Shift by half a sector so sector 0 is centered on North
    idx = int(((direction_deg + width / 2.0) % 360.0) // width)
    return idx


def compute_snow_drift_for_season(
    df: pd.DataFrame,
    v_threshold: float = 5.0,
    temp_threshold: float = 0.0,
    snow_threshold: float = 0.0,
    n_sectors: int = 16,
):
    """
    Simplified snow drift index for one snow season:
    - Use only hours where:
        * T <= temp_threshold (°C)
        * snowfall > snow_threshold (mm)
        * windspeed_10m > v_threshold (m/s)
    - Snow drift index ~ sum( (v - v_threshold)^3 ) over these hours
      (cubic relationship inspired by Tabler).
    - Sector totals: same index, but accumulated per wind direction sector.
    """
    cond = (
        (df["temperature_2m"] <= temp_threshold)
        & (df["snowfall"] > snow_threshold)
        & (df["windspeed_10m"] > v_threshold)
    )
    active = df[cond].copy()

    if active.empty:
        return 0.0, np.zeros(n_sectors), 0.0, 0

    v = active["windspeed_10m"].to_numpy()
    # Cubic relationship above threshold (arbitrary scaling factor 1.0)
    q_hourly = (v - v_threshold) ** 3

    drift_index = float(q_hourly.sum())
    total_snow = float(active["snowfall"].sum())
    n_hours = int(len(active))

    dirs = active["winddirection_10m"].to_numpy()
    sectors = np.zeros(n_sectors, dtype=float)

    for q, d in zip(q_hourly, dirs):
        s = sector_index(d, n_sectors=n_sectors)
        sectors[s] += q

    return drift_index, sectors, total_snow, n_hours


def plot_wind_rose(avg_sector_values: np.ndarray, title: str = "Snow drift wind rose"):
    """
    Plot a polar wind rose using average sector drift values.
    """
    n_sectors = len(avg_sector_values)
    # Angles at sector centers
    theta = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
    width = 2 * np.pi / n_sectors

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    bars = ax.bar(theta, avg_sector_values, width=width, bottom=0.0, alpha=0.8)

    # Align 0° with North and rotate clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Cardinal labels every 45° (N, NE, E, SE, S, SW, W, NW)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

    ax.set_title(title, va="bottom")
    return fig


# -------------------------------------------------------------------
# 4. UI: year range and parameters
# -------------------------------------------------------------------
st.subheader("Snow year definition and settings")

col_years, col_thresh = st.columns([2, 1])

with col_years:
    years = list(range(2000, 2025))
    start_year, end_year = st.select_slider(
        "Snow years (1 July – 30 June)",
        options=years,
        value=(2019, 2023),
        help="A snow year runs from 1 July of the selected year "
             "to 30 June of the following year.",
    )

with col_thresh:
    v_threshold = st.slider(
        "Wind speed threshold (m/s)",
        min_value=3.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Only hours with wind speed above this threshold "
             "are counted towards snow drift.",
    )

st.caption(
    "For each snow year, we fetch ERA5 reanalysis data for this coordinate, "
    "compute a simple snow drift index, and aggregate it by wind direction sector."
)

if start_year > end_year:
    st.error("Start year must be ≤ end year.")
    st.stop()

years_range = list(range(start_year, end_year + 1))

# -------------------------------------------------------------------
# 5. Compute snow drift per year
# -------------------------------------------------------------------
results = []
sector_sums = None
n_sectors = 16

with st.spinner("Downloading ERA5 data and computing snow drift..."):
    for y in years_range:
        # Snow year: 1 July y -> 30 June y+1
        start_date = dt.date(y, 7, 1)
        end_date = dt.date(y + 1, 6, 30)

        df = fetch_era5_hourly(
            lat=lat,
            lon=lon,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        drift_index, sectors, total_snow, n_hours = compute_snow_drift_for_season(
            df,
            v_threshold=v_threshold,
            temp_threshold=0.0,
            snow_threshold=0.0,
            n_sectors=n_sectors,
        )

        if sector_sums is None:
            sector_sums = sectors.copy()
        else:
            sector_sums += sectors

        results.append(
            {
                "snow_year": f"{y}/{y+1}",
                "drift_index": drift_index,
                "total_snow_mm": total_snow,
                "active_hours": n_hours,
            }
        )

if not results:
    st.warning("No years selected – nothing to compute.")
    st.stop()

df_results = pd.DataFrame(results)

# Avoid division by zero
if sector_sums is None or np.allclose(sector_sums, 0):
    avg_sector_values = np.zeros(n_sectors)
else:
    avg_sector_values = sector_sums / len(years_range)

# -------------------------------------------------------------------
# 6. Plots: snow drift per year + wind rose
# -------------------------------------------------------------------
st.subheader("Results")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Snow drift index per snow year**")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df_results["snow_year"], df_results["drift_index"], marker="o")
    ax1.set_xlabel("Snow year (July–June)")
    ax1.set_ylabel("Snow drift index (arbitrary units)")
    ax1.grid(alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig1, use_container_width=True)

    st.dataframe(df_results, use_container_width=True)

with col_right:
    st.markdown("**Wind rose for snow drift (average over selected years)**")
    fig2 = plot_wind_rose(avg_sector_values, title="Snow drift wind rose")
    st.pyplot(fig2, use_container_width=True)

st.caption(
    "This is a simplified snow drift model based on ERA5 hourly data from Open-Meteo. "
    "If your `Snow_drift.py` implements the full Tabler method, you can reuse those "
    "functions here and replace the simplified computation."
)
