# pages/7_Sliding_Window_Correlation.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from utils_elhub import get_client

st.set_page_config(
    page_title="Sliding Window Correlation",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Sliding Window Correlation – Meteorology vs Energy")

# -------------------------------------------------------------
# 1. Meteorology fetcher (Open-Meteo ERA5)
# -------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_weather(lat, lon, variable, start_date, end_date):
    """
    Fetches a single meteorological variable from Open-Meteo ERA5.
    VARIABLE *must* be a single string → we wrap it in a list.
    """
    url = "https://archive-api.open-meteo.com/v1/era5"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [variable],   # ← FIX: must be list – prevents join error
        "timezone": "Europe/Oslo",
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()
    if "hourly" not in data:
        raise RuntimeError(f"No 'hourly' returned from Open-Meteo for '{variable}'.")

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


# -------------------------------------------------------------
# 2. Energy fetcher (MongoDB)
# -------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_energy(area: str, variable: str, start_date: str, end_date: str):
    """
    Fetch energy production or consumption from MongoDB.
    variable = "production_2021_by_hour" or "consumption_2021_by_hour"
    """
    cli = get_client()
    db = cli["elhub"]
    coll = db[variable]

    match_stage = {
        "$match": {
            "priceArea": area,
            "startTime": {
                "$gte": pd.to_datetime(start_date),
                "$lt": pd.to_datetime(end_date),
            },
        }
    }

    project_stage = {
        "$project": {
            "_id": 0,
            "time": "$startTime",
            "quantityKwh": 1,
        }
    }

    docs = list(coll.aggregate([match_stage, project_stage]))

    if not docs:
        return pd.DataFrame(columns=["time", "quantityKwh"]).set_index("time")

    df = pd.DataFrame(docs)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


# -------------------------------------------------------------
# 3. Sliding Window Correlation
# -------------------------------------------------------------
def sliding_correlation(series_x, series_y, window, lag):
    """
    Computes correlation(Y vs X shifted) in sliding windows.
    """
    # Apply lag: positive = weather leads energy
    y_shifted = series_y.shift(lag)

    corr = y_shifted.rolling(window).corr(series_x)
    return corr


# -------------------------------------------------------------
# 4. UI Controls
# -------------------------------------------------------------
st.header("Choose Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    weather_var = st.selectbox(
        "Meteorological variable",
        ["temperature_2m", "windspeed_10m", "snowfall"],
        index=0,
    )

with col2:
    energy_var = st.selectbox(
        "Energy dataset",
        [
            ("Production (kWh)", "production_2021_by_hour"),
            ("Consumption (kWh)", "consumption_2021_by_hour"),
        ],
        format_func=lambda x: x[0],
    )

with col3:
    area = st.selectbox("Price Area", ["NO1", "NO2", "NO3", "NO4", "NO5"])

start_date = "2021-01-01"
end_date = "2021-12-31"

st.markdown(
    f"""
    **Date interval:**  
    Weather + energy data fetched for: **{start_date} → {end_date}**
    """
)

col_lag, col_win = st.columns(2)

with col_lag:
    lag = st.slider(
        "Lag (hours)",
        min_value=-72,
        max_value=72,
        value=0,
        step=1,
        help="Positive: weather leads energy. Negative: energy leads weather.",
    )

with col_win:
    window = st.slider(
        "Window length (hours)",
        min_value=24,
        max_value=500,
        value=168,
        step=24,
        help="Rolling window size in hours.",
    )


# -------------------------------------------------------------
# 5. Fetch data
# -------------------------------------------------------------
with st.spinner("Fetching weather + energy data..."):
    weather = fetch_weather(
        lat=59.91,  # Oslo default (NO1)
        lon=10.75,
        variable=weather_var,
        start_date=start_date,
        end_date=end_date,
    )

    df_energy = fetch_energy(
        area=area,
        variable=energy_var[1],
        start_date=start_date,
        end_date=end_date,
    )

# Merge
df = weather.join(df_energy, how="inner")
df.rename(columns={"quantityKwh": "energy"}, inplace=True)

if df.empty:
    st.error("No overlapping timestamps found. Check database or API.")
    st.stop()

# -------------------------------------------------------------
# 6. Compute correlation
# -------------------------------------------------------------
corr = sliding_correlation(
    df[weather_var], df["energy"], window=window, lag=lag
)

df_corr = pd.DataFrame({"corr": corr})


# -------------------------------------------------------------
# 7. Plot result
# -------------------------------------------------------------
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_corr.index,
        y=df_corr["corr"],
        mode="lines",
        name="Correlation",
    )
)

fig.update_layout(
    title=f"Sliding Window Correlation ({weather_var} vs {area} energy) – Window={window}h, Lag={lag}h",
    xaxis_title="Time",
    yaxis_title="Correlation",
    yaxis=dict(range=[-1, 1]),
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Raw merged data")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Correlation series")
st.dataframe(df_corr.dropna(), use_container_width=True)
