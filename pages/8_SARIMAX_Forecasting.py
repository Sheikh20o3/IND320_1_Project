import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils_elhub import list_price_areas

# ---------------- Streamlit page config ---------------- #

st.set_page_config(
    page_title="SARIMAX Forecasting",
    page_icon="🔮",
    layout="wide",
)

st.title("SARIMAX Forecasting – Energy Production & Consumption")

st.caption(
    "Forecast hourly energy production or consumption using SARIMAX, "
    "optionally with meteorological variables as exogenous regressors."
)

# ---------------- Helper constants ---------------- #

# Approximate coordinates per Norwegian price area
PRICEAREA_COORDS = {
    "NO1": (59.9139, 10.7522),   # Oslo
    "NO2": (58.1467, 7.9956),    # Kristiansand-ish
    "NO3": (63.4305, 10.3951),   # Trondheim
    "NO4": (69.6492, 18.9553),   # Tromsø
    "NO5": (60.39299, 5.32415),  # Bergen
}

# Labels for ERA5 meteorological variables
METEO_LABELS = {
    "temperature_2m": "Temperature 2m (°C)",
    "windspeed_10m": "Wind speed 10m (m/s)",
    "precipitation": "Precipitation (mm)",
    "snowfall": "Snowfall (cm snow water equivalent)",
}

# Paths to CSV-files in the repo
REPO_ROOT = Path(__file__).resolve().parent.parent
ASS4_DIR = REPO_ROOT / "Ass4_Rapporter"

CONSUMPTION_CSV = ASS4_DIR / "elhub_consumption_2021_2024_all_areas.csv"
PRODUCTION_CSV_2021 = ASS4_DIR / "elhub_production_2021_all_areas.csv"
PRODUCTION_CSV_2022_2024 = ASS4_DIR / "elhub_production_2022_2024_all_areas.csv"

# Vi begrenser treningsdatasettet for å holde SARIMAX kjapp
MAX_TRAIN_POINTS = 1500  # ~ 62 dager med timesdata


# ---------------- Time handling helpers ---------------- #

def _to_naive_oslo(series: pd.Series) -> pd.Series:
    """
    Convert a Series with timestamps (strings or datetime) to tz-naive
    timestamps in Europe/Oslo.
    Works for both tz-aware and tz-naive input.
    """
    dt_utc = pd.to_datetime(series, errors="coerce", utc=True)
    dt_local = dt_utc.dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    return dt_local


# ---------------- Data loading: ERA5 (Open-Meteo) ---------------- #

@st.cache_data(show_spinner=True)
def fetch_era5_hourly(lat: float, lon: float, year: int) -> pd.DataFrame:
    """
    Fetch hourly ERA5 reanalysis from Open-Meteo for one full year.
    Returns DataFrame with columns: time, temperature_2m, windspeed_10m,
    snowfall, precipitation.
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

    df["time"] = pd.to_datetime(df["time"])
    if getattr(df["time"].dt, "tz", None) is not None:
        df["time"] = df["time"].dt.tz_convert("Europe/Oslo").dt.tz_localize(None)
    else:
        df["time"] = df["time"].dt.tz_localize(None)

    return df


# ---------------- Data loading: Elhub CSV ---------------- #

def _load_energy_csv_generic(
    paths,
    price_area: str,
    year: int,
    area_col: str = "priceArea",
    time_col: str = "startTime",
    value_col: str = "quantityKwh",
) -> pd.DataFrame:
    """
    Generic loader for Elhub CSV-files (both production & consumption).

    - Reads one or more CSV paths.
    - Filters on price area and year.
    - Converts timestamps to tz-naive Europe/Oslo.
    - Aggregates to one row per hour: 'time', 'energy_kwh'.

    Note: Series used for SARIMAX is the TOTAL hourly kWh for the given
    price area and year, aggregated over all groups present in the CSV.
    """
    frames = []

    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        df = pd.read_csv(path)

        missing_cols = [c for c in (area_col, time_col, value_col) if c not in df.columns]
        if missing_cols:
            continue

        df = df[df[area_col] == price_area].copy()
        if df.empty:
            continue

        # Time-håndtering
        df["time"] = _to_naive_oslo(df[time_col])
        df = df.dropna(subset=["time"])

        # Begrens til valgt år
        start_dt = pd.Timestamp(year=year, month=1, day=1)
        end_dt = pd.Timestamp(year=year + 1, month=1, day=1)
        df = df[(df["time"] >= start_dt) & (df["time"] < end_dt)]
        if df.empty:
            continue

        # Numerisk energikolonne
        df["energy_kwh"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["energy_kwh"])

        frames.append(df[["time", "energy_kwh"]])

    if not frames:
        return pd.DataFrame(columns=["time", "energy_kwh"])

    out = pd.concat(frames, ignore_index=True)

    # Aggreger til én rad per time (summert over grupper)
    out = (
        out.groupby("time", as_index=False)["energy_kwh"]
        .sum()
        .sort_values("time")
        .reset_index(drop=True)
    )

    return out


@st.cache_data(show_spinner=True)
def load_consumption_series(price_area: str, year: int) -> pd.DataFrame:
    """Load hourly consumption for one price area & year from CSV."""
    if not CONSUMPTION_CSV.exists():
        st.error(
            "Consumption-CSV-fil ikke funnet.\n\n"
            f"Forventet sti: '{CONSUMPTION_CSV}'."
        )
        return pd.DataFrame(columns=["time", "energy_kwh"])

    df = _load_energy_csv_generic(
        paths=[CONSUMPTION_CSV],
        price_area=price_area,
        year=year,
        area_col="priceArea",
        time_col="startTime",
        value_col="quantityKwh",
    )

    if df.empty:
        st.warning(
            f"Ingen consumption-data for {price_area} i {year} "
            f"i CSV-filen '{CONSUMPTION_CSV.name}'."
        )

    return df


@st.cache_data(show_spinner=True)
def load_production_series(price_area: str, year: int) -> pd.DataFrame:
    """Load hourly production for one price area & year from CSV."""
    paths = []
    if PRODUCTION_CSV_2021.exists():
        paths.append(PRODUCTION_CSV_2021)
    if PRODUCTION_CSV_2022_2024.exists():
        paths.append(PRODUCTION_CSV_2022_2024)

    if not paths:
        st.error(
            "Production-CSV-filer ikke funnet.\n\n"
            f"Forventet minst én av:\n"
            f"- {PRODUCTION_CSV_2021}\n"
            f"- {PRODUCTION_CSV_2022_2024}"
        )
        return pd.DataFrame(columns=["time", "energy_kwh"])

    df = _load_energy_csv_generic(
        paths=paths,
        price_area=price_area,
        year=year,
        area_col="priceArea",
        time_col="startTime",
        value_col="quantityKwh",
    )

    if df.empty:
        st.warning(
            f"Ingen production-data for {price_area} i {year} "
            f"i production-CSV-filene."
        )

    return df


@st.cache_data(show_spinner=True)
def fetch_energy_series(
    price_area: str,
    dataset: str,
    year: int,
) -> pd.DataFrame:
    """
    Wrapper brukt av siden.

    dataset: "Production" eller "Consumption"
    Returnerer DataFrame med ['time', 'energy_kwh'].
    """
    if dataset == "Consumption":
        return load_consumption_series(price_area, year)
    else:
        return load_production_series(price_area, year)


# ---------------- SARIMAX helper ---------------- #

def prepare_endog_and_exog(
    df_energy: pd.DataFrame,
    df_met: pd.DataFrame | None,
    exog_keys: list[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    forecast_hours: int,
    max_train_points: int = MAX_TRAIN_POINTS,
):
    """
    Align energy & meteorology, subset train period, and build forecast index.
    Returns (y_train, y_future_index, exog_train, exog_forecast).

    Vi begrenser treningsdatasettet til maks `max_train_points` siste observasjoner
    for å gjøre SARIMAX raskere (spesielt på Streamlit Cloud).
    """
    # Energy til timeindeks
    df_energy = df_energy.copy()
    df_energy["time"] = pd.to_datetime(df_energy["time"])
    df_energy = df_energy.sort_values("time")
    df_energy = df_energy.set_index("time")

    y_all = df_energy["energy_kwh"].asfreq("H")
    y_all = y_all.interpolate(limit_direction="both")

    # Treningsperiode (inklusiv)
    mask_train = (y_all.index >= train_start) & (y_all.index <= train_end)
    y_train = y_all.loc[mask_train]

    if y_train.empty:
        raise ValueError("Treningsperioden er tom – juster datoene.")

    # Forecast-horisont
    forecast_start = train_end + pd.Timedelta(hours=1)
    forecast_index = pd.date_range(
        start=forecast_start,
        periods=forecast_hours,
        freq="H",
    )

    exog_train = None
    exog_forecast = None

    if exog_keys and df_met is not None:
        df_met = df_met.copy()
        df_met["time"] = pd.to_datetime(df_met["time"])
        df_met = df_met.set_index("time").sort_index()

        df_met = df_met[exog_keys]

        exog_all = df_met.reindex(y_all.index.union(forecast_index))
        exog_all = exog_all.interpolate(limit_direction="both")

        exog_train = exog_all.loc[y_train.index]
        exog_forecast = exog_all.loc[forecast_index]

    # BEGRENS TRENINGSLENGDE: bare siste max_train_points observasjoner
    if len(y_train) > max_train_points:
        y_train = y_train.iloc[-max_train_points:]
        if exog_train is not None:
            exog_train = exog_train.loc[y_train.index]

    return y_train, forecast_index, exog_train, exog_forecast


def run_sarimax(
    y_train: pd.Series,
    forecast_steps: int,
    exog_train=None,
    exog_forecast=None,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24),
):
    """
    Fit SARIMAX and return (results, forecast_mean, lower_ci, upper_ci).

    Robust oppsett:
    - trend='c' for å ha intercept (unngå at serien kollapser mot 0).
    - ingen simple_differencing (statsmodels håndterer differensiering).
    - fallback til enklere ikke-sesongmodell hvis første forsøk feiler.
    """
    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        results = model.fit(
            method="lbfgs",
            maxiter=150,
            disp=False,
        )

    except Exception as e:
        st.warning(
            f"Initial SARIMAX configuration failed ({e}). "
            "Falling back to a simpler non-seasonal ARIMA(1,1,1) with intercept."
        )

        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        results = model.fit(
            method="lbfgs",
            maxiter=150,
            disp=False,
        )

    if exog_forecast is not None:
        forecast_res = results.get_forecast(steps=forecast_steps, exog=exog_forecast)
    else:
        forecast_res = results.get_forecast(steps=forecast_steps)

    forecast_mean = forecast_res.predicted_mean
    ci = forecast_res.conf_int()
    lower_ci = ci.iloc[:, 0]
    upper_ci = ci.iloc[:, 1]

    return results, forecast_mean, lower_ci, upper_ci


# ---------------- UI Controls ---------------- #

default_areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
try:
    areas_from_db = list_price_areas()
    area_options = areas_from_db or default_areas
except Exception:
    area_options = default_areas

col_top1, col_top2, col_top3 = st.columns(3)

with col_top1:
    price_area = st.selectbox(
        "Price area",
        options=area_options,
        index=0,
    )

with col_top2:
    dataset = st.radio(
        "Dataset",
        options=["Production", "Consumption"],
        horizontal=True,
    )

with col_top3:
    year = st.selectbox(
        "Year",
        options=[2021, 2022, 2023, 2024],
        index=0,
    )

st.markdown(
    f"*Note: The training dataset is automatically limited to the last "
    f"{MAX_TRAIN_POINTS} hours in the chosen training period to keep the model responsive.*"
)
st.markdown("---")

col_param1, col_param2 = st.columns(2)

with col_param1:
    st.subheader("ARIMA parameters")

    p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1, step=1)
    d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=1, step=1)
    q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1, step=1)

with col_param2:
    st.subheader("Seasonal parameters")

    P = st.number_input("Seasonal AR (P)", min_value=0, max_value=3, value=1, step=1)
    D = st.number_input("Seasonal differencing (D)", min_value=0, max_value=2, value=1, step=1)
    Q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=3, value=1, step=1)
    m = st.selectbox(
        "Seasonal period (m)",
        options=[24, 24 * 7],
        index=0,
        format_func=lambda x: f"{x} hours ({'daily' if x == 24 else 'weekly'})",
    )

st.markdown("---")

col_train, col_forecast = st.columns(2)

with col_train:
    st.subheader("Training period (within selected year)")

    train_start_date = st.date_input("Training start date", dt.date(year, 1, 1))
    train_end_date = st.date_input("Training end date (inclusive)", dt.date(year, 12, 31))

with col_forecast:
    st.subheader("Forecast horizon")

    forecast_days = st.slider(
        "Forecast horizon (days ahead)",
        min_value=1,
        max_value=60,
        value=14,
        step=1,
    )
    forecast_hours = forecast_days * 24

st.markdown("---")

st.subheader("Exogenous variables (optional)")

exog_keys = st.multiselect(
    "Select meteorological variables to include as exogenous regressors",
    options=list(METEO_LABELS.keys()),
    format_func=lambda k: METEO_LABELS[k],
)

st.markdown("---")

# ---------------- Load data ---------------- #

with st.spinner("Loading Elhub energy data..."):
    df_energy = fetch_energy_series(price_area, dataset, year)

if df_energy.empty:
    st.error(
        f"No {dataset.lower()} data available for {price_area} in {year}. "
        "Check that the corresponding CSV files are present in the repo."
    )
    st.stop()

st.info(
    f"Modelling hourly **{dataset.lower()} energy (kWh)** for "
    f"**price area {price_area}** in **{year}**, aggregated over all "
    "production/consumption groups present in the CSV data."
)

min_time = pd.to_datetime(df_energy["time"]).min()
max_time = pd.to_datetime(df_energy["time"]).max()

train_start = pd.Timestamp.combine(train_start_date, dt.time(0, 0))
train_end = pd.Timestamp.combine(train_end_date, dt.time(23, 0))

if train_start < min_time:
    train_start = min_time
if train_end > max_time:
    train_end = max_time
if train_end <= train_start:
    st.error("Training end must be after training start.")
    st.stop()

forecast_start = train_end + pd.Timedelta(hours=1)
max_forecast_end = max_time

available_hours = int((max_forecast_end - forecast_start) / pd.Timedelta(hours=1)) + 1
if available_hours <= 0:
    st.error(
        "No data available after the training period in the selected year – "
        "cannot define a forecast horizon. Try ending the training period earlier."
    )
    st.stop()

if forecast_hours > available_hours:
    st.warning(
        f"Requested forecast horizon of {forecast_hours} hours exceeds available "
        f"data ({available_hours} hours). Using {available_hours} hours instead."
    )
    forecast_hours = available_hours

# ---------------- Fetch meteorology if needed ---------------- #

df_met = None
if exog_keys:
    if price_area not in PRICEAREA_COORDS:
        st.error(f"No coordinates defined for price area {price_area}. Cannot fetch ERA5.")
        st.stop()

    lat, lon = PRICEAREA_COORDS[price_area]

    with st.spinner("Downloading ERA5 meteorological data from Open-Meteo..."):
        try:
            df_met = fetch_era5_hourly(lat, lon, year)
        except Exception as e:
            st.error(f"Failed to fetch ERA5 data: {e}")
            df_met = None

# ---------------- Run SARIMAX ---------------- #

run_button = st.button("Run SARIMAX forecast")

if run_button:
    try:
        y_train, forecast_index, exog_train, exog_forecast = prepare_endog_and_exog(
            df_energy=df_energy,
            df_met=df_met,
            exog_keys=exog_keys,
            train_start=train_start,
            train_end=train_end,
            forecast_hours=forecast_hours,
            max_train_points=MAX_TRAIN_POINTS,
        )

        order = (int(p), int(d), int(q))
        seasonal_order = (int(P), int(D), int(Q), int(m))

        with st.spinner("Fitting SARIMAX model..."):
            results, forecast_mean, lower_ci, upper_ci = run_sarimax(
                y_train=y_train,
                forecast_steps=len(forecast_index),
                exog_train=exog_train,
                exog_forecast=exog_forecast,
                order=order,
                seasonal_order=seasonal_order,
            )

    except Exception as e:
        st.error(f"Failed to fit SARIMAX model or produce forecast: {e}")
        st.stop()

    # ---------------- Plot results ---------------- #

    st.subheader("Forecast results")

    df_energy_idx = df_energy.copy()
    df_energy_idx["time"] = pd.to_datetime(df_energy_idx["time"])
    df_energy_idx = df_energy_idx.set_index("time").sort_index()
    y_all = df_energy_idx["energy_kwh"].asfreq("H")
    y_all = y_all.interpolate(limit_direction="both")

    y_train_full = y_all[(y_all.index >= train_start) & (y_all.index <= train_end)]
    y_actual_future = y_all.reindex(forecast_index)

    fig = go.Figure()

    # Training data
    fig.add_trace(
        go.Scatter(
            x=y_train_full.index,
            y=y_train_full.values,
            mode="lines",
            name="Training data",
        )
    )

    # Actual future values (if present)
    if not y_actual_future.isna().all():
        fig.add_trace(
            go.Scatter(
                x=y_actual_future.index,
                y=y_actual_future.values,
                mode="lines",
                name="Actual (future)",
            )
        )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast_mean.values,
            mode="lines",
            name="Forecast",
        )
    )

    # Confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=upper_ci.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=lower_ci.values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="95% confidence interval",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=f"{dataset} energy (kWh)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Forecast values (head)"):
        df_out = pd.DataFrame(
            {
                "time": forecast_index,
                "forecast": forecast_mean.values,
                "lower_95": lower_ci.values,
                "upper_95": upper_ci.values,
            }
        )
        st.dataframe(df_out.head(), use_container_width=True)
