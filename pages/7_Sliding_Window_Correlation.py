# pages/8_Sliding_Window_Correlation.py

import datetime as dt
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import download_open_meteo
from utils_elhub import get_client, list_price_areas

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Sliding Window Correlation – Meteorology vs Energy",
    page_icon="📈",
    layout="wide",
)

st.title("Meteorology and Energy – Sliding Window Correlation")

st.markdown(
    """
This page computes a **sliding window correlation** between:

- One **meteorological variable** (ERA5 / Open-Meteo, hourly), and  
- One **energy variable** (**production** or **consumption** from Elhub/MongoDB),

for a selected Norwegian price area (**NO1–NO5**).

You can adjust:

- **Lag (hours):** how much the energy series is shifted relative to the meteorology  
- **Window length (hours):** size of the rolling window used for correlation
"""
)

# -------------------------------------------------------------------
# 1. Select price area and variables
# -------------------------------------------------------------------

# Price areas from DB, fallback if DB not available
try:
    areas_from_db: List[str] = list_price_areas()
    if not areas_from_db:
        areas_from_db = ["NO1", "NO2", "NO3", "NO4", "NO5"]
except Exception:
    areas_from_db = ["NO1", "NO2", "NO3", "NO4", "NO5"]

default_area = st.session_state.get("price_area", areas_from_db[0])

col_sel1, col_sel2, col_sel3 = st.columns([1.2, 1.2, 1.8])

with col_sel1:
    price_area = st.selectbox(
        "Price area",
        options=areas_from_db,
        index=areas_from_db.index(default_area)
        if default_area in areas_from_db
        else 0,
        help="Norwegian Elspot price area.",
    )
    # Keep in session so other pages (map, snow drift) can reuse it
    st.session_state["price_area"] = price_area

with col_sel2:
    energy_mode = st.selectbox(
        "Energy variable",
        options=["Production (kWh)", "Consumption (kWh)"],
        help="Choose whether to use production or consumption from Elhub.",
    )

with col_sel3:
    st.markdown("**Year / time span**")
    st.caption("For now we use the full year **2021-01-01 → 2021-12-31** "
               "for both meteorology and energy.")
    year_start = 2021
    year_end = 2021

# -------------------------------------------------------------------
# 2. Sliders: lag and window length
# -------------------------------------------------------------------
st.subheader("Correlation settings")

col_lag, col_win = st.columns(2)

with col_lag:
    lag_hours = st.slider(
        "Lag (hours)",
        min_value=-72,
        max_value=72,
        value=0,
        step=1,
        help=(
            "Positive lag shifts the **energy** series forward in time.\n\n"
            "Example: lag = +6 means we correlate meteorology(t) with energy(t + 6h)."
        ),
    )

with col_win:
    window_hours = st.slider(
        "Window length (hours, rolling window)",
        min_value=24,
        max_value=24 * 30,
        value=24 * 7,
        step=24,
        help="Number of hours used in each rolling correlation window.",
    )

st.caption(
    "Correlation is computed as a rolling Pearson correlation over the selected window size.\n"
    "Both series are aligned to hourly resolution and restricted to the year 2021."
)

# -------------------------------------------------------------------
# 3. Data fetchers
# -------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def get_weather_era5(price_area: str, year_start: int, year_end: int) -> pd.DataFrame:
    """
    Download ERA5 reanalysis via Open-Meteo for the given price area and years.
    Uses utils.download_open_meteo, which maps price areas to representative cities.
    """
    start_date = dt.date(year_start, 1, 1).isoformat()
    end_date = dt.date(year_end, 12, 31).isoformat()

    df = download_open_meteo(
        price_area=price_area,
        start_date=start_date,
        end_date=end_date,
        hourly=None,  # will use DEFAULT_HOURLY inside utils
    )
    if "time" not in df.columns:
        raise RuntimeError("Weather DF does not contain 'time' column.")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    return df


@st.cache_data(show_spinner=True)
def get_energy_series(price_area: str, mode: str) -> pd.DataFrame:
    """
    Fetch hourly energy time series from MongoDB for year 2021:
      - Aggregate quantityKwh over all groups in the selected price area.
      - mode = 'production' or 'consumption'
    Returns DataFrame with columns: ['time', 'energy_kwh'].
    """
    client = get_client()
    db = client["elhub"]

    if mode == "production":
        coll_name = "production_2021_by_hour"
    else:
        coll_name = "consumption_2021_by_hour"

    coll = db[coll_name]

    start_dt = dt.datetime(2021, 1, 1)
    end_dt = dt.datetime(2022, 1, 1)

    pipeline = [
        {
            "$match": {
                "priceArea": price_area,
                "startTime": {"$gte": start_dt, "$lt": end_dt},
            }
        },
        {
            "$group": {
                "_id": "$startTime",
                "energy_kwh": {"$sum": "$quantityKwh"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "time": "$_id",
                "energy_kwh": 1,
            }
        },
        {"$sort": {"time": 1}},
    ]

    docs = list(coll.aggregate(pipeline))
    if not docs:
        return pd.DataFrame(columns=["time", "energy_kwh"])

    df = pd.DataFrame(docs)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    return df


# -------------------------------------------------------------------
# 4. Fetch data and set up variable selectors
# -------------------------------------------------------------------
with st.spinner("Fetching meteorology and energy time series..."):
    try:
        df_weather = get_weather_era5(price_area, year_start, year_end)
    except Exception as e:
        st.error(f"Failed to fetch weather data for {price_area}: {e}")
        st.stop()

    energy_mode_key = "production" if energy_mode.startswith("Production") else "consumption"
    try:
        df_energy = get_energy_series(price_area, energy_mode_key)
    except Exception as e:
        st.error(f"Failed to fetch energy data for {price_area}: {e}")
        st.stop()

if df_energy.empty:
    st.error(
        "No energy data found for this price area and year. "
        "Check that your MongoDB collections contain 2021 data."
    )
    st.stop()

# Meteorological variables – pick from available columns
candidate_meteo_vars = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "windspeed_10m",
    "winddirection_10m",
    "pressure_msl",
]

available_meteo_vars = [c for c in candidate_meteo_vars if c in df_weather.columns]

if not available_meteo_vars:
    st.error("Weather DataFrame does not contain any of the expected meteorological variables.")
    st.stop()

meteo_var = st.selectbox(
    "Meteorological variable",
    options=available_meteo_vars,
    help="Select the weather variable used in the correlation.",
)

st.markdown(
    f"Selected **price area:** `{price_area}`, "
    f"**meteorological variable:** `{meteo_var}`, "
    f"**energy:** `{energy_mode}`."
)

# -------------------------------------------------------------------
# 5. Build aligned and lagged series + sliding correlation
# -------------------------------------------------------------------
st.subheader("Sliding window correlation")

# Align to hourly and merge
df_w = df_weather[["time", meteo_var]].copy().rename(columns={meteo_var: "meteo"})
df_e = df_energy[["time", "energy_kwh"]].copy().rename(columns={"energy_kwh": "energy"})

df_w = df_w.set_index("time").sort_index().resample("1H").mean()
df_e = df_e.set_index("time").sort_index().resample("1H").sum()

df_merged = pd.concat([df_w, df_e], axis=1, join="inner").dropna()
if df_merged.empty:
    st.error("No overlapping hourly data between weather and energy series.")
    st.stop()

# Apply lag (shift energy)
if lag_hours != 0:
    df_merged["energy_lagged"] = df_merged["energy"].shift(lag_hours)
else:
    df_merged["energy_lagged"] = df_merged["energy"]

df_merged = df_merged.dropna()

if len(df_merged) < window_hours:
    st.error(
        f"Not enough data ({len(df_merged)} points) for a window of {window_hours} hours. "
        "Reduce the window length."
    )
    st.stop()

# Normalize for joint plotting (z-scores)
def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


df_merged["meteo_norm"] = zscore(df_merged["meteo"])
df_merged["energy_norm"] = zscore(df_merged["energy_lagged"])

# Sliding correlation
df_merged["corr"] = (
    df_merged["meteo_norm"]
    .rolling(window=window_hours, min_periods=max(10, window_hours // 4))
    .corr(df_merged["energy_norm"])
)

# -------------------------------------------------------------------
# 6. Plot with Plotly
# -------------------------------------------------------------------
col_ts, col_corr = st.columns(2)

with col_ts:
    st.markdown("**Normalized time series (meteorology vs energy)**")

    fig_ts = go.Figure()

    fig_ts.add_trace(
        go.Scatter(
            x=df_merged.index,
            y=df_merged["meteo_norm"],
            mode="lines",
            name=f"{meteo_var} (z-score)",
        )
    )

    fig_ts.add_trace(
        go.Scatter(
            x=df_merged.index,
            y=df_merged["energy_norm"],
            mode="lines",
            name=f"{energy_mode} (lagged {lag_hours}h, z-score)",
        )
    )

    fig_ts.update_layout(
        xaxis_title="Time",
        yaxis_title="Normalized value (z-score)",
        legend_title="Series",
        height=400,
    )

    st.plotly_chart(fig_ts, use_container_width=True)

with col_corr:
    st.markdown("**Sliding window correlation**")

    fig_corr = go.Figure()

    fig_corr.add_trace(
        go.Scatter(
            x=df_merged.index,
            y=df_merged["corr"],
            mode="lines",
            name="Sliding correlation",
        )
    )

    fig_corr.update_yaxes(range=[-1.05, 1.05])
    fig_corr.update_layout(
        xaxis_title="Time (window center)",
        yaxis_title=f"corr({meteo_var}, {energy_mode_key}, lag={lag_hours}h)",
        height=400,
    )

    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------------------------------------------
# 7. Data table / debug
# -------------------------------------------------------------------
with st.expander("Show correlation data (debug / inspection)"):
    st.write(
        "This table shows the merged, lagged and normalized data used for the plots "
        "(last rows first)."
    )
    st.dataframe(
        df_merged[["meteo", "energy", "energy_lagged", "meteo_norm", "energy_norm", "corr"]]
        .sort_index(ascending=False)
        .head(200),
        use_container_width=True,
    )
