# pages/8_SARIMAX_Forecasting.py

import datetime as dt
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils_elhub import list_price_areas

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SARIMAX Forecasting",
    page_icon="🔮",
    layout="wide",
)

st.title("SARIMAX Forecasting – Energy Production & Consumption")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONSUMPTION_CSV = os.path.join(
    PROJECT_ROOT, "Ass4_Rapporter", "elhub_consumption_2021_2024_all_areas.csv"
)

PRODUCTION_CSV_2021 = os.path.join(
    PROJECT_ROOT, "Ass4_Rapporter", "elhub_production_2021_all_areas.csv"
)
PRODUCTION_CSV_2022_2024 = os.path.join(
    PROJECT_ROOT, "Ass4_Rapporter", "elhub_production_2022_2024_all_areas.csv"
)

PRICEAREA_COORDS: dict[str, Tuple[float, float]] = {
    "NO1": (59.9139, 10.7522),   # Oslo
    "NO2": (58.1467, 7.9956),    # Kristiansand-ish
    "NO3": (63.4305, 10.3951),   # Trondheim
    "NO4": (69.6492, 18.9553),   # Tromsø
    "NO5": (60.39299, 5.32415),  # Bergen
}

METEO_LABELS = {
    "temperature_2m": "Temperature 2m (°C)",
    "windspeed_10m": "Wind speed 10m (m/s)",
    "precipitation": "Precipitation (mm)",
    "snowfall": "Snowfall (cm of snow water equivalent)",
}


# ---------------------------------------------------------------------------
# Helper: timezone handling
# ---------------------------------------------------------------------------

def _to_naive_oslo(series: pd.Series) -> pd.Series:
    """
    Convert a Series with datetime-like / ISO strings to tz-naive
    Europe/Oslo datetimes. Safe on strings, tz-aware og allerede-naiv.
    """
    s = pd.to_datetime(series, errors="coerce")
    # Etter to_datetime er .dt alltid tilgjengelig (DatetimeIndex/Series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    else:
        s = s.dt.tz_localize(None)
    return s


# ---------------------------------------------------------------------------
# ERA5 / Open-Meteo
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def fetch_era5_hourly(lat: float, lon: float, year: int) -> pd.DataFrame:
    """
    Fetch hourly ERA5 reanalysis from Open-Meteo for one full year.
    Returns tz-naive Europe/Oslo timestamps in 'time'.
    """
    base_url = "https://archive-api.open-meteo.com/v1/era5"

    start_date = dt.date(year, 1, 1)
    end_date = dt.date(year, 12, 31)

    hourly_vars = list(METEO_LABELS.keys())

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
    df["time"] = _to_naive_oslo(df["time"])

    return df


# ---------------------------------------------------------------------------
# Energy series loaders (CSV for both Production & Consumption)
# ---------------------------------------------------------------------------

def _load_energy_csv_generic(
    paths: List[str],
    price_area: str,
    year: int,
    kind: str,
) -> pd.DataFrame:
    """
    Generic CSV loader:
      * paths: list of CSV paths to concat
      * expects columns: priceArea, startTime, quantityKwh
      * returns ['time', 'energy_kwh'] for given price_area and year
    """
    existing_paths = [p for p in paths if os.path.exists(p)]

    if not existing_paths:
        st.error(
            f"No CSV files found for {kind.lower()}.\n\n"
            f"Tried: {paths}"
        )
        return pd.DataFrame()

    dfs = []
    for path in existing_paths:
        df_i = pd.read_csv(
            path,
            usecols=["priceArea", "startTime", "quantityKwh"],
            low_memory=False,
        )
        dfs.append(df_i)

    df = pd.concat(dfs, ignore_index=True)

    # Filter on price area
    df = df[df["priceArea"] == price_area].copy()
    if df.empty:
        st.warning(
            f"No {kind.lower()} rows in CSV for price area {price_area}."
        )
        return pd.DataFrame()

    # Parse time & year
    df["time"] = _to_naive_oslo(df["startTime"])
    df = df.dropna(subset=["time"])

    start_dt = dt.datetime(year, 1, 1)
    end_dt = dt.datetime(year + 1, 1, 1)

    df = df[(df["time"] >= start_dt) & (df["time"] < end_dt)]
    if df.empty:
        st.warning(
            f"No {kind.lower()} data in CSV for {price_area} in {year}."
        )
        return pd.DataFrame()

    df["energy_kwh"] = pd.to_numeric(df["quantityKwh"], errors="coerce")
    df = df.dropna(subset=["energy_kwh"])

    return df[["time", "energy_kwh"]].sort_values("time")


@st.cache_data(show_spinner=True)
def load_consumption_series(price_area: str, year: int) -> pd.DataFrame:
    """Consumption from single CSV file."""
    return _load_energy_csv_generic(
        [CONSUMPTION_CSV],
        price_area=price_area,
        year=year,
        kind="consumption",
    )


@st.cache_data(show_spinner=True)
def load_production_series(price_area: str, year: int) -> pd.DataFrame:
    """Production from 2021 + 2022–2024 CSV files."""
    return _load_energy_csv_generic(
        [PRODUCTION_CSV_2021, PRODUCTION_CSV_2022_2024],
        price_area=price_area,
        year=year,
        kind="production",
    )


@st.cache_data(show_spinner=True)
def fetch_energy_series(price_area: str, dataset: str, year: int) -> pd.DataFrame:
    """
    Wrapper for energy series.
      dataset: "Production" or "Consumption"
    """
    if dataset == "Consumption":
        return load_consumption_series(price_area, year)
    else:
        return load_production_series(price_area, year)


# ---------------------------------------------------------------------------
# SARIMAX forecasting
# ---------------------------------------------------------------------------

def run_sarimax_forecast(
    df_all: pd.DataFrame,
    exog_cols: List[str],
    train_start: dt.datetime,
    train_end: dt.datetime,
    horizon_hours: int,
    order: tuple,
    seasonal_order: tuple,
    trend: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run SARIMAX with given parameters.

    df_all: DataFrame with columns ['time', 'energy_kwh', (exog...)]
    exog_cols: list of column names to use as exogenous
    Returns:
      df_train, df_future, df_fc
    """
    df_all = df_all.sort_values("time").copy()

    # Split into train / future
    mask_train = (df_all["time"] >= train_start) & (df_all["time"] <= train_end)
    df_train = df_all.loc[mask_train].copy()
    df_future = df_all.loc[df_all["time"] > train_end].copy()

    if df_train.empty:
        raise ValueError("Training set is empty – adjust training period.")

    if df_future.empty:
        raise ValueError(
            "No future data after training end – reduce training end date "
            "or check that the year contains data."
        )

    # Limit forecast horizon to available future rows
    steps = min(horizon_hours, len(df_future))
    if steps <= 0:
        raise ValueError("Forecast horizon is zero – check horizon or period.")

    df_future = df_future.iloc[:steps].copy()

    y_train = df_train["energy_kwh"]
    y_future_actual = df_future["energy_kwh"]

    if exog_cols:
        exog_train = df_train[exog_cols]
        exog_future = df_future[exog_cols]
    else:
        exog_train = None
        exog_future = None

    # Build & fit model
    model = SARIMAX(
        endog=y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False)

    # Dynamic multi-step forecast into the future
    fc_res = results.get_forecast(steps=steps, exog=exog_future)
    fc_mean = fc_res.predicted_mean
    ci = fc_res.conf_int(alpha=0.05)

    # Conf-int column names (e.g. 'lower energy_kwh', 'upper energy_kwh')
    lower_col = ci.columns[0]
    upper_col = ci.columns[1]

    df_fc = pd.DataFrame(
        {
            "time": df_future["time"].values,
            "forecast": fc_mean.values,
            "lower": ci[lower_col].values,
            "upper": ci[upper_col].values,
            "actual": y_future_actual.values,
        }
    )

    return df_train, df_future, df_fc


# ---------------------------------------------------------------------------
# UI controls
# ---------------------------------------------------------------------------

# Price areas (prefer DB list if available)
try:
    areas_from_db = list_price_areas()
    area_options = areas_from_db or ["NO1", "NO2", "NO3", "NO4", "NO5"]
except Exception:
    area_options = ["NO1", "NO2", "NO3", "NO4", "NO5"]

col_top1, col_top2, col_top3 = st.columns(3)

with col_top1:
    price_area = st.selectbox("Price area", options=area_options, index=0)

with col_top2:
    year = st.selectbox("Year", options=[2021, 2022, 2023, 2024], index=0)

with col_top3:
    dataset = st.radio(
        "Energy series",
        ["Production", "Consumption"],
        horizontal=True,
    )

st.caption(
    "Forecast hourly energy (production or consumption) using a SARIMAX model "
    "with optional meteorological exogenous variables."
)

# Meteorological exogenous variables
exog_keys = st.multiselect(
    "Exogenous variables (ERA5)",
    options=list(METEO_LABELS.keys()),
    format_func=lambda k: METEO_LABELS[k],
    default=["temperature_2m"],
)

# Training period and horizon
st.subheader("Training period & forecast horizon")

col_time1, col_time2, col_time3 = st.columns(3)

with col_time1:
    train_start_date = st.date_input(
        "Training start date",
        value=dt.date(year, 1, 1),
        min_value=dt.date(year, 1, 1),
        max_value=dt.date(year, 12, 31),
    )

with col_time2:
    train_end_date = st.date_input(
        "Training end date",
        value=dt.date(year, 9, 30),
        min_value=dt.date(year, 1, 1),
        max_value=dt.date(year, 12, 31),
    )

with col_time3:
    horizon_days = st.slider(
        "Forecast horizon (days, hourly resolution)",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
    )

horizon_hours = horizon_days * 24

# SARIMAX parameters
st.subheader("SARIMAX parameters")

col_ord1, col_ord2, col_ord3 = st.columns(3)
with col_ord1:
    p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1, step=1)
with col_ord2:
    d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=0, step=1)
with col_ord3:
    q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1, step=1)

col_sord1, col_sord2, col_sord3, col_sord4 = st.columns(4)
with col_sord1:
    P = st.number_input("Seasonal AR (P)", min_value=0, max_value=3, value=1, step=1)
with col_sord2:
    D = st.number_input("Seasonal diff (D)", min_value=0, max_value=2, value=0, step=1)
with col_sord3:
    Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=3, value=1, step=1)
with col_sord4:
    m = st.selectbox(
        "Seasonal period m",
        options=[24, 24 * 7],
        index=0,
        format_func=lambda v: f"{v} (daily)" if v == 24 else f"{v} (weekly)",
    )

trend = st.selectbox(
    "Trend",
    options=["n", "c", "t", "ct"],
    index=1,  # default 'c' (constant)
    format_func=lambda t: {
        "n": "n (no trend)",
        "c": "c (constant)",
        "t": "t (linear trend)",
        "ct": "ct (constant + trend)",
    }[t],
)

order = (int(p), int(d), int(q))
seasonal_order = (int(P), int(D), int(Q), int(m))

st.markdown("---")

run_button = st.button("Run SARIMAX forecast", type="primary")

# ---------------------------------------------------------------------------
# Data fetching & forecasting
# ---------------------------------------------------------------------------

if run_button:
    if train_start_date > train_end_date:
        st.error("Training start date must be before training end date.")
        st.stop()

    if price_area not in PRICEAREA_COORDS:
        st.error(f"No coordinates defined for price area {price_area}.")
        st.stop()

    # Convert training dates to datetimes
    train_start_dt = dt.datetime.combine(train_start_date, dt.time(0, 0))
    # Use end-of-day 23:00 for training end
    train_end_dt = dt.datetime.combine(train_end_date, dt.time(23, 0))

    lat, lon = PRICEAREA_COORDS[price_area]

    with st.spinner("Fetching ERA5 and energy series..."):
        try:
            df_met = fetch_era5_hourly(lat, lon, year)
        except Exception as e:
            st.error(f"Failed to fetch ERA5 data: {e}")
            st.stop()

        df_energy = fetch_energy_series(price_area, dataset, year)

    if df_energy.empty:
        st.error(
            f"No {dataset.lower()} data available for {price_area} in {year}. "
            "Check that CSV files exist and contain this area/year."
        )
        st.stop()

    # Align and merge exogenous variables (if any selected)
    df_energy = df_energy.copy()
    df_energy["time"] = _to_naive_oslo(df_energy["time"])

    if exog_keys:
        missing_cols = [c for c in exog_keys if c not in df_met.columns]
        if missing_cols:
            st.error(
                f"Selected exogenous variables not found in ERA5 data: {missing_cols}"
            )
            st.stop()

        df_met_small = df_met[["time"] + exog_keys].copy()
        df_met_small["time"] = _to_naive_oslo(df_met_small["time"])

        df_all = pd.merge(
            df_energy,
            df_met_small,
            on="time",
            how="inner",
        )
    else:
        df_all = df_energy.copy()

    # Drop any NA in target or selected exog
    cols_required = ["time", "energy_kwh"] + exog_keys
    df_all = df_all.dropna(subset=cols_required)

    if df_all.empty:
        st.error(
            "Merged dataset is empty after aligning energy and meteorology. "
            "Try another year, price area or exogenous selection."
        )
        st.stop()

    try:
        df_train, df_future, df_fc = run_sarimax_forecast(
            df_all=df_all,
            exog_cols=exog_keys,
            train_start=train_start_dt,
            train_end=train_end_dt,
            horizon_hours=horizon_hours,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
        )
    except Exception as e:
        st.error(f"SARIMAX fitting/forecasting failed: {e}")
        st.stop()

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    st.subheader("Forecast results")

    # Main time series plot
    fig_ts = go.Figure()

    # Training data
    fig_ts.add_trace(
        go.Scatter(
            x=df_train["time"],
            y=df_train["energy_kwh"],
            mode="lines",
            name="Train (observed)",
        )
    )

    # Actual future
    fig_ts.add_trace(
        go.Scatter(
            x=df_future["time"],
            y=df_future["energy_kwh"],
            mode="lines",
            name="Future (actual)",
        )
    )

    # Forecast mean
    fig_ts.add_trace(
        go.Scatter(
            x=df_fc["time"],
            y=df_fc["forecast"],
            mode="lines",
            name="Forecast",
        )
    )

    # Confidence interval band
    fig_ts.add_trace(
        go.Scatter(
            x=pd.concat([df_fc["time"], df_fc["time"][::-1]]),
            y=pd.concat([df_fc["upper"], df_fc["lower"][::-1]]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            name="95% CI",
            opacity=0.2,
        )
    )

    fig_ts.update_layout(
        xaxis_title="Time",
        yaxis_title=f"{dataset} energy (kWh)",
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        shapes=[
            dict(
                type="line",
                x0=train_end_dt,
                x1=train_end_dt,
                y0=min(df_all["energy_kwh"]),
                y1=max(df_all["energy_kwh"]),
                line=dict(dash="dash"),
            )
        ],
    )

    st.plotly_chart(fig_ts, use_container_width=True)

    # Show head of data
    with st.expander("Data preview (train, future, forecast)"):
        st.write("Training data:")
        st.dataframe(df_train.head(), use_container_width=True)
        st.write("Future data (actual):")
        st.dataframe(df_future.head(), use_container_width=True)
        st.write("Forecast (incl. CI & actual):")
        st.dataframe(df_fc.head(), use_container_width=True)
