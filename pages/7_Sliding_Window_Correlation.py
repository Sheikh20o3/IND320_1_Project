# pages/7_Sliding_Window_Correlation.py
import datetime as dt
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

from utils_elhub import get_client, list_price_areas

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Sliding Window Correlation",
    page_icon="📈",
    layout="wide",
)

st.title("Sliding Window Correlation – Meteorology vs. Energy")

# -------------------------------------------------------------------
# Constants and helpers
# -------------------------------------------------------------------

# Approximate coordinates per Norwegian price area
PRICEAREA_COORDS = {
    "NO1": (59.9139, 10.7522),   # Oslo
    "NO2": (58.1467, 7.9956),    # Kristiansand-ish
    "NO3": (63.4305, 10.3951),   # Trondheim
    "NO4": (69.6492, 18.9553),   # Tromsø
    "NO5": (60.39299, 5.32415),  # Bergen
}

# Project root: one level up from pages/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# CSV path for consumption (only source for Consumption)
CONSUMPTION_CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "Ass4_Rapporter",
    "elhub_consumption_2021_2024_all_areas.csv",
)

def _ensure_naive_oslo(series: pd.Series) -> pd.Series:
    """
    Convert a Series to datetime64[ns] and ensure it's tz-naive in Europe/Oslo.
    Works for strings, tz-aware datetimes and tz-naive datetimes.
    """
    series = pd.to_datetime(series, errors="coerce")
    # If tz-aware, convert to Europe/Oslo and drop tz
    tz = getattr(series.dt, "tz", None)
    if tz is not None:
        series = series.dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    return series


# -------------------------------------------------------------------
# ERA5 (Open-Meteo) loader
# -------------------------------------------------------------------
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

    # Normalise time: tz-naive in Europe/Oslo
    df["time"] = _ensure_naive_oslo(df["time"])
    df = df.dropna(subset=["time"])

    return df


# -------------------------------------------------------------------
# Consumption from CSV (ONLY source for Consumption)
# -------------------------------------------------------------------
def _load_consumption_from_csv(price_area: str, year: int) -> pd.DataFrame:
    """
    Load hourly consumption series from the local CSV file for one price area
    and one year. This is the ONLY source for Consumption.
    Returns DataFrame with columns: ['time', 'energy_kwh'].
    """
    if not os.path.exists(CONSUMPTION_CSV_PATH):
        st.error(
            "Consumption CSV file not found.\n\n"
            f"Expected path: '{CONSUMPTION_CSV_PATH}'.\n"
            "Check that the file is in the repo under 'Ass4_Rapporter/'."
        )
        return pd.DataFrame()

    df = pd.read_csv(CONSUMPTION_CSV_PATH)

    # Map lower-case -> original column names
    cols = {c.lower(): c for c in df.columns}

    area_col = cols.get("pricearea") or cols.get("price_area") or cols.get("area")
    time_col = (
        cols.get("starttime")
        or cols.get("time")
        or cols.get("timestamp")
        or cols.get("datetime")
    )
    qty_col = (
        cols.get("quantitykwh")
        or cols.get("quantity_kwh")
        or cols.get("kwh")
        or cols.get("energy_kwh")
        or cols.get("value")
    )

    if not (area_col and time_col and qty_col):
        st.error(
            "Could not auto-detect column names in the consumption CSV file.\n\n"
            f"Columns: {list(df.columns)}"
        )
        return pd.DataFrame()

    # Filter by price area
    df = df[df[area_col] == price_area].copy()
    if df.empty:
        st.warning(
            f"No consumption data in CSV for price area {price_area} "
            f"(any year)."
        )
        return pd.DataFrame()

    # Normalize time to tz-naive Europe/Oslo and drop rows without valid time
    df["time"] = _ensure_naive_oslo(df[time_col])
    df = df.dropna(subset=["time"])

    # Filter by year AFTER timezone normalization
    start_dt = pd.Timestamp(year=year, month=1, day=1)
    end_dt = pd.Timestamp(year=year + 1, month=1, day=1)

    df = df[(df["time"] >= start_dt) & (df["time"] < end_dt)]
    if df.empty:
        st.warning(
            f"No consumption data found in CSV for {price_area} in {year}."
        )
        return pd.DataFrame()

    # Ensure numeric kWh
    df["energy_kwh"] = pd.to_numeric(df[qty_col], errors="coerce")
    df = df.dropna(subset=["energy_kwh"])
    df = df.sort_values("time")

    return df[["time", "energy_kwh"]]


# -------------------------------------------------------------------
# Production from MongoDB (as before, with robust detection)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def _load_production_from_mongo(price_area: str, year: int) -> pd.DataFrame:
    """
    Load hourly production series from MongoDB for one price area and one year.
    Returns DataFrame with columns: ['time', 'energy_kwh'].
    """
    cli = get_client()
    all_db_names = cli.list_database_names()

    keyword = "prod"
    preferred_collections = [
        "production_2021_by_hour",
        "production_2021_2024",
        "production",
        "elhub_production",
    ]

    chosen_db = None
    chosen_coll = None

    # Prefer DB named "elhub"
    if "elhub" in all_db_names:
        db = cli["elhub"]
        existing = set(db.list_collection_names())
        for name in preferred_collections:
            if name in existing:
                chosen_db = db
                chosen_coll = db[name]
                break

    # If not found, scan all DBs for a collection containing "prod"
    if chosen_coll is None:
        for db_name in all_db_names:
            db = cli[db_name]
            for coll_name in db.list_collection_names():
                if keyword in coll_name.lower():
                    chosen_db = db
                    chosen_coll = db[coll_name]
                    break
            if chosen_coll is not None:
                break

    if chosen_coll is None:
        st.error(
            "No MongoDB collection found for production "
            f"(searched for keyword '{keyword}')."
        )
        return pd.DataFrame()

    st.info(
        f"Using MongoDB collection '{chosen_db.name}.{chosen_coll.name}' "
        "for production."
    )

    # Introspect one sample document to detect field names
    sample = chosen_coll.find_one()
    if not sample:
        st.error(
            f"Collection '{chosen_db.name}.{chosen_coll.name}' is empty – "
            "no production data."
        )
        return pd.DataFrame()

    keys = {k.lower(): k for k in sample.keys()}

    def pick(possible_names):
        for cand in possible_names:
            lc = cand.lower()
            if lc in keys:
                return keys[lc]
        return None

    area_field = pick(["pricearea", "price_area", "area"])
    time_field = pick(["starttime", "start_time", "time", "timestamp", "datetime"])
    qty_field = pick(["quantitykwh", "quantity_kwh", "kwh", "energy_kwh", "value"])

    if not (area_field and time_field and qty_field):
        st.error(
            "Could not auto-detect field names in MongoDB production document.\n\n"
            f"Sample keys: {list(sample.keys())}"
        )
        return pd.DataFrame()

    # Match only by price area; filter by year in pandas after normalising time
    match = {area_field: price_area}

    pipeline = [
        {"$match": match},
        {"$sort": {time_field: 1}},
        {
            "$project": {
                "_id": 0,
                "time": f"${time_field}",
                "energy_kwh": f"${qty_field}",
            }
        },
    ]

    try:
        docs = list(chosen_coll.aggregate(pipeline))
    except Exception as e:
        st.error(f"MongoDB aggregation for production failed: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    if df.empty:
        st.warning(
            f"No production data in MongoDB for {price_area} "
            f"(any year)."
        )
        return pd.DataFrame()

    # Normalize time to tz-naive Europe/Oslo
    df["time"] = _ensure_naive_oslo(df["time"])
    df = df.dropna(subset=["time"])

    # Filter by year
    start_dt = pd.Timestamp(year=year, month=1, day=1)
    end_dt = pd.Timestamp(year=year + 1, month=1, day=1)

    df = df[(df["time"] >= start_dt) & (df["time"] < endDt)]
    if df.empty:
        st.warning(
            f"No production data found in MongoDB for {price_area} in {year}."
        )
        return pd.DataFrame()

    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    df = df.dropna(subset=["energy_kwh"])
    df = df.sort_values("time")

    return df[["time", "energy_kwh"]]


# -------------------------------------------------------------------
# Public energy fetcher (used by the app)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_elhub_series(
    price_area: str,
    dataset: str,
    year: int,
) -> pd.DataFrame:
    """
    Fetch hourly energy series for one year and one price area.

    dataset: "Production" or "Consumption"

    - Consumption: ONLY from CSV file.
    - Production: from MongoDB (as before).
    Returns DataFrame with columns: ['time', 'energy_kwh'].
    """
    if dataset == "Consumption":
        return _load_consumption_from_csv(price_area, year)

    # Production
    return _load_production_from_mongo(price_area, year)


# -------------------------------------------------------------------
# Sliding correlation
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# UI controls
# -------------------------------------------------------------------

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
    dataset = st.radio(
        "Energy series (Production vs Consumption)",
        ["Production", "Consumption"],
        horizontal=True,
    )

st.caption(
    "We correlate an hourly meteorological variable from ERA5 (Open-Meteo) with "
    "hourly Elhub energy data (production or consumption) for the same price area and year."
)

# Meteorological property selector
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

# -------------------------------------------------------------------
# Fetch & align data
# -------------------------------------------------------------------

if price_area not in PRICEAREA_COORDS:
    st.error(f"No coordinates defined for price area {price_area}.")
    st.stop()

lat, lon = PRICEAREA_COORDS[price_area]

with st.spinner("Downloading ERA5 data and energy data..."):
    try:
        df_met = fetch_era5_hourly(lat, lon, year)
    except Exception as e:
        st.error(f"Failed to fetch weather data for {price_area}: {e}")
        st.stop()

    df_energy = fetch_elhub_series(price_area, dataset, year)

if df_energy.empty:
    st.error(
        f"No {dataset.lower()} data available for {price_area} in {year}. "
        "Check that MongoDB/CSV is correctly loaded."
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

# Make sure times are datetime and naive (they should already be, but be safe)
df_met_small["time"] = pd.to_datetime(df_met_small["time"], errors="coerce")
df_energy_small["time"] = pd.to_datetime(df_energy_small["time"], errors="coerce")

df_met_small = df_met_small.dropna(subset=["time"]).sort_values("time")
df_energy_small = df_energy_small.dropna(subset=["time"]).sort_values("time")

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

# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

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
