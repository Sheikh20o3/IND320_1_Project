# pages/7_Sliding_Window_Correlation.py
import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from utils_elhub import get_client, list_price_areas

# Page config
st.set_page_config(
    page_title="Sliding Window Correlation",
    page_icon="📈",
    layout="wide",
)

st.title("Sliding Window Correlation – Meteorology vs. Energy")

# Approximate coordinates per Norwegian price area
PRICEAREA_COORDS = {
    "NO1": (59.9139, 10.7522),   # Oslo
    "NO2": (58.1467, 7.9956),    # Kristiansand-ish
    "NO3": (63.4305, 10.3951),   # Trondheim
    "NO4": (69.6492, 18.9553),   # Tromsø
    "NO5": (60.39299, 5.32415),  # Bergen
}


@st.cache_data(show_spinner=True)
def fetch_era5_hourly(lat: float, lon: float, year: int) -> pd.DataFrame:
    """
    Fetch hourly ERA5 reanalysis from Open-Meteo for one full year.
    Keeps a small set of relevant variables.
    """
    base_url = "https://archive-api.open-meteo.com/v1/era5"

    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)

    hourly_vars = [
        "temperature_2m",
        "windspeed_10m",
        "snowfall",
        "precipitation",
    ]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(hourly_vars),
        "timezone": "Europe/Oslo",
    }

    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise RuntimeError("No 'hourly' field returned from Open-Meteo API.")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)

    # Parse time and drop timezone so it matches Elhub (tz-naive)
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    else:
        df["time"] = df["time"].dt.tz_localize(None)

    return df


@st.cache_data(show_spinner=True)
def fetch_elhub_series(
    price_area: str,
    dataset: str,
    year: int,
) -> pd.DataFrame:
    """
    Fetch hourly energy series from MongoDB for one year and one price area.

    dataset: "Production" or "Consumption"
    Returns DataFrame with columns: time, energy_kwh
    """
    cli = get_client()
    db = cli["elhub"]

    # Her antar vi at dataene fra del 2 ligger i:
    #   elhub.production_2021_by_hour
    #   elhub.consumption_2021_by_hour
    # med feltnavn: pricearea, starttime, quantitykwh
    if dataset == "Production":
        coll_name = "production_2021_by_hour"
    else:
        coll_name = "consumption_2021_by_hour"

    coll = db[coll_name]

    start_dt = dt.datetime(year, 1, 1)
    end_dt = dt.datetime(year + 1, 1, 1)

    # OBS: lowercase feltnavn, matcher Mongo
    match = {
        "pricearea": price_area,
        "starttime": {"$gte": start_dt, "$lt": end_dt},
    }

    pipeline = [
        {"$match": match},
        {"$sort": {"starttime": 1}},
        {
            "$project": {
                "_id": 0,
                "time": "$starttime",
                "energy_kwh": "$quantitykwh",
            }
        },
    ]

    docs = list(coll.aggregate(pipeline))
    df = pd.DataFrame(docs)

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"])

    # Drop timezone hvis den finnes, så vi matcher ERA5
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    else:
        df["time"] = df["time"].dt.tz_localize(None)

    return df





def compute_sliding_correlation(
    df: pd.DataFrame,
    lag_hours: int,
    window_hours: int,
    meteo_col: str,
    energy_col: str,
) -> pd.DataFrame:
    """
    Given a DataFrame with columns ['time', meteo_col, energy_col],
    compute rolling correlation between meteo and energy with:
      - energy shifted by lag_hours
      - rolling window = window_hours (in hours)
    Returns DataFrame with columns: ['time', 'corr', meteo_col, energy_col_shifted]
    """
    df = df.sort_values("time").copy()
    df.set_index("time", inplace=True)

    x = df[meteo_col].astype(float)

    # Positive lag means energy lags behind meteorology
    y = df[energy_col].astype(float).shift(lag_hours)

    corr = x.rolling(window=window_hours, min_periods=window_hours // 2).corr(y)

    out = pd.DataFrame(
        {
            "time": corr.index,
            "corr": corr.values,
            meteo_col: x.values,
            f"{energy_col}_shifted": y.values,
        }
    ).dropna(subset=["corr"])

    return out


# ---------------- UI controls ---------------- #

# Price area selection (from DB if possible)
try:
    areas_from_db = list_price_areas()
    area_options = areas_from_db or ["NO1", "NO2", "NO3", "NO4", "NO5"]
except Exception:
    area_options = ["NO1", "NO2", "NO3", "NO4", "NO5"]

default_area = st.session_state.get("price_area", area_options[0])

col_top1, col_top2, col_top3 = st.columns(3)

with col_top1:
    price_area = st.selectbox(
        "Price area",
        options=area_options,
        index=area_options.index(default_area)
        if default_area in area_options
        else 0,
    )

with col_top2:
    year = st.selectbox("Year", options=[2021, 2022, 2023, 2024], index=0)

with col_top3:
    # Selector for energy production vs consumption (oppgavetekst)
    dataset = st.radio(
        "Energy series (Production vs Consumption)",
        ["Production", "Consumption"],
        horizontal=True,
    )

st.caption(
    "We correlate an hourly meteorological variable from ERA5 (Open-Meteo) with "
    "hourly Elhub energy data (production or consumption) for the same price area and year."
)

# Selector for meteorological property (oppgavetekst)
METEO_LABELS = {
    "temperature_2m": "Temperature 2m (°C)",
    "windspeed_10m": "Wind speed 10m (m/s)",
    "precipitation": "Precipitation (mm)",
    "snowfall": "Snowfall (cm of snow water equivalent)",
}

meteo_key = st.selectbox(
    "Meteorological property",
    options=list(METEO_LABELS.keys()),
    format_func=lambda k: METEO_LABELS[k],
    index=0,
)

# Lag & window length
col_ctrl1, col_ctrl2 = st.columns(2)

with col_ctrl1:
    lag_hours = st.slider(
        "Lag (hours, positive = energy lags behind meteorology)",
        min_value=-120,
        max_value=120,
        value=0,
        step=1,
    )

with col_ctrl2:
    window_days = st.slider(
        "Window length (days)",
        min_value=3,
        max_value=60,
        value=14,
        step=1,
    )

window_hours = window_days * 24

# ---------------- Fetch & align data ---------------- #

if price_area not in PRICEAREA_COORDS:
    st.error(f"No coordinates defined for price area {price_area}.")
    st.stop()

lat, lon = PRICEAREA_COORDS[price_area]

with st.spinner("Downloading ERA5 data and Elhub data..."):
    try:
        df_met = fetch_era5_hourly(lat, lon, year)
    except Exception as e:
        st.error(f"Failed to fetch weather data for {price_area}: {e}")
        st.stop()

    df_energy = fetch_elhub_series(price_area, dataset, year)

if df_energy.empty:
    st.error(
        f"No {dataset.lower()} data found in MongoDB for {price_area} in {year}. "
        "Check that the corresponding collection is loaded."
    )
    st.stop()

if meteo_key not in df_met.columns:
    st.error(
        f"Meteorological variable '{meteo_key}' not present in ERA5 data. "
        "Check the API parameters."
    )
    st.stop()

# Keep only required columns and align on hourly timestamps
df_met_small = df_met[["time", meteo_key]].dropna()
df_energy_small = df_energy[["time", "energy_kwh"]].dropna()

# Ensure both are tz-naive and sorted
df_met_small["time"] = pd.to_datetime(df_met_small["time"])
if df_met_small["time"].dt.tz is not None:
    df_met_small["time"] = df_met_small["time"].dt.tz_localize(None)
df_met_small = df_met_small.sort_values("time")

df_energy_small["time"] = pd.to_datetime(df_energy_small["time"])
if df_energy_small["time"].dt.tz is not None:
    df_energy_small["time"] = df_energy_small["time"].dt.tz_localize(None)
df_energy_small = df_energy_small.sort_values("time")

idx_met = df_met_small["time"]
idx_eng = df_energy_small["time"]

common_idx = idx_met[idx_met.isin(idx_eng)]

if common_idx.empty:
    st.error(
        "No overlapping timestamps found between ERA5 data and "
        f"{dataset.lower()} data.\n\n"
        "Check that both datasets cover the same year and that timestamps "
        "are hourly and aligned."
    )

    with st.expander("Debug info"):
        st.write("ERA5 time range:", str(idx_met.min()), "→", str(idx_met.max()))
        st.write("ERA5 rows:", len(idx_met))
        st.write(f"{dataset} time range:", str(idx_eng.min()), "→", str(idx_eng.max()))
        st.write(f"{dataset} rows:", len(idx_eng))

    st.stop()

# Restrict both to common timestamps
df_met_aligned = df_met_small[df_met_small["time"].isin(common_idx)].copy()
df_energy_aligned = df_energy_small[df_energy_small["time"].isin(common_idx)].copy()

# Merge to one frame
df_merged = pd.merge(
    df_met_aligned,
    df_energy_aligned,
    on="time",
    how="inner",
)

if df_merged.empty:
    st.error("Merged DataFrame is empty after alignment – nothing to correlate.")
    st.stop()

# Compute sliding correlation
df_corr = compute_sliding_correlation(
    df=df_merged,
    lag_hours=lag_hours,
    window_hours=window_hours,
    meteo_col=meteo_key,
    energy_col="energy_kwh",
)

if df_corr.empty:
    st.warning(
        "Correlation series is empty after rolling calculation. "
        "Try a shorter window or smaller lag."
    )
    st.stop()

# ---------------- Plots ---------------- #

st.subheader("Time series and sliding window correlation")

col_plot1, col_plot2 = st.columns([2, 1])

with col_plot1:
    st.markdown(
        f"**Hourly {METEO_LABELS[meteo_key]} and "
        f"{dataset.lower()} energy (shifted by {lag_hours} h)**"
    )

    fig_ts = go.Figure()

    fig_ts.add_trace(
        go.Scatter(
            x=df_corr["time"],
            y=df_corr[meteo_key],
            mode="lines",
            name=METEO_LABELS[meteo_key],
        )
    )

    fig_ts.add_trace(
        go.Scatter(
            x=df_corr["time"],
            y=df_corr["energy_kwh_shifted"],
            mode="lines",
            name=f"{dataset} (shifted)",
            yaxis="y2",
        )
    )

    fig_ts.update_layout(
        xaxis_title="Time",
        yaxis=dict(
            title=METEO_LABELS[meteo_key],
            side="left",
        ),
        yaxis2=dict(
            title=f"{dataset} energy (kWh)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=450,
    )

    st.plotly_chart(fig_ts, use_container_width=True)

with col_plot2:
    st.markdown(
        f"**Sliding window correlation** "
        f"({window_days} d window, lag = {lag_hours} h)"
    )

    fig_corr = go.Figure()
    fig_corr.add_trace(
        go.Scatter(
            x=df_corr["time"],
            y=df_corr["corr"],
            mode="lines",
            name="Correlation",
        )
    )
    fig_corr.update_layout(
        xaxis_title="Time",
        yaxis_title="Correlation coefficient",
        height=450,
        yaxis=dict(range=[-1, 1]),
    )

    st.plotly_chart(fig_corr, use_container_width=True)

with st.expander("Data used for correlation (head)"):
    st.write("Merged and aligned data:")
    st.dataframe(df_merged.head(), use_container_width=True)
    st.write("Correlation series (head):")
    st.dataframe(df_corr.head(), use_container_width=True)
