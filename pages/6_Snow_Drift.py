# pages/6_Snow_Drift.py
import datetime as dt
import calendar
from math import pi  # kan evt. fjernes, men skader ikke

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Snow drift & wind rose",
    page_icon="❄️",
    layout="wide",
)

st.title("Snow drift calculation and wind rose (ERA5 / Open-Meteo)")

# ------------------------------------------------------------
# 1. Get coordinate from map page
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 2. Helper: fetch ERA5 hourly data from Open-Meteo
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_era5_hourly(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly ERA5 reanalysis from Open-Meteo for given coordinate and date range.
    Variables: temperature, snowfall, wind speed & direction at 10 m.
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

# ------------------------------------------------------------
# 3. Tabler-baserte hjelpefunksjoner (Qt i kg/m)
# ------------------------------------------------------------
def compute_Qupot(hourly_wind_speeds, dt: float = 3600.0) -> float:
    """
    Potential wind-driven snow transport Qupot [kg/m], Tabler (2003):

        Qupot = sum((u^3.8) * dt) / 233847

    hvor u er vindhastighet [m/s] og dt er tidssteg [s].
    """
    hourly_wind_speeds = [float(u) for u in hourly_wind_speeds]
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847.0
    return float(total)


def sector_index(direction_deg: float, n_sectors: int = 16) -> int:
    """
    Map wind direction in degrees [0, 360) to sector index 0..n_sectors-1.
    Sectors are 360/n_sectors degrees wide, centered on the main compass
    directions (N, NNE, NE, ...).
    """
    width = 360.0 / n_sectors
    # Center bins by half a sector (e.g. 11.25° for 16 sectors)
    return int(((direction_deg + width / 2.0) % 360.0) // width)


def compute_sector_transport(
    hourly_wind_speeds,
    hourly_wind_dirs,
    dt: float = 3600.0,
    n_sectors: int = 16,
):
    """
    Compute cumulative transport per sector [kg/m] using Tabler's u^3.8 formula.
    Samme logikk som compute_sector_transport i Snow_drift.py.
    """
    sectors = [0.0] * n_sectors
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        u = float(u)
        d = float(d)
        idx = sector_index(d, n_sectors=n_sectors)
        sectors[idx] += ((u ** 3.8) * dt) / 233847.0
    return sectors


def compute_snow_transport(
    T: float,
    F: float,
    theta: float,
    Swe: float,
    hourly_wind_speeds,
    dt: float = 3600.0,
):
    """
    Compute Tabler (2003) snow transport components.

    Parameters
    ----------
    T : float
        Maximum transport distance [m].
    F : float
        Fetch distance [m].
    theta : float
        Relocation coefficient [-].
    Swe : float
        Total snowfall water equivalent [mm].
    hourly_wind_speeds : list of float
        Hourly wind speeds [m/s].
    dt : float
        Time step [s].

    Returns
    -------
    dict
        Qupot (kg/m): Potential wind-driven transport.
        Qspot (kg/m): Snowfall-limited transport.
        Srwe (mm): Relocated water equivalent.
        Qinf (kg/m): Controlling transport value.
        Qt (kg/m): Mean annual snow transport.
        Control: "Snowfall controlled" or "Wind controlled".
    """
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe  # [kg/m]
    Srwe = theta * Swe     # [mm]

    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"

    Qt = Qinf * (1 - 0.14 ** (F / T))

    return {
        "Qupot (kg/m)": float(Qupot),
        "Qspot (kg/m)": float(Qspot),
        "Srwe (mm)": float(Srwe),
        "Qinf (kg/m)": float(Qinf),
        "Qt (kg/m)": float(Qt),
        "Control": control,
    }

# ------------------------------------------------------------
# 4. Sesongvis Qt og sektorer
# ------------------------------------------------------------
def compute_snow_drift_for_season(
    df: pd.DataFrame,
    v_threshold: float = 5.0,
    temp_threshold: float = 1.0,
    snow_threshold: float = 0.0,
    n_sectors: int = 16,
    T: float = 3000.0,
    F: float = 30000.0,
    theta: float = 0.5,
):
    """
    Beregn Tabler-basert snøtransport for én snøsesong:

    - Swe_hourly = snowfall når T <= temp_threshold.
    - Swe_total = sum(Swe_hourly) [mm].
    - Vind under v_threshold settes til 0 for Qupot og sektorer.
    - Returnerer Qt [kg/m], sektorbidrag [kg/m], total Swe [mm]
      og antall "aktive" timer.
    """
    if df.empty:
        return 0.0, np.zeros(n_sectors), 0.0, 0

    df = df.copy()
    df["Swe_hourly"] = np.where(df["temperature_2m"] <= temp_threshold,
                                df["snowfall"], 0.0)
    Swe_total = float(df["Swe_hourly"].sum())

    ws = df["windspeed_10m"].to_numpy(dtype=float)
    wdir = df["winddirection_10m"].to_numpy(dtype=float)

    # Bruk vindterskel i Qupot og sektorer
    ws_eff = np.where(ws > v_threshold, ws, 0.0)

    transport = compute_snow_transport(
        T=T,
        F=F,
        theta=theta,
        Swe=Swe_total,
        hourly_wind_speeds=ws_eff.tolist(),
        dt=3600.0,
    )
    Qt = transport["Qt (kg/m)"]

    # Sektorvis fordeling [kg/m] med samme effektive vind
    sectors = np.array(
        compute_sector_transport(
            hourly_wind_speeds=ws_eff.tolist(),
            hourly_wind_dirs=wdir.tolist(),
            dt=3600.0,
            n_sectors=n_sectors,
        ),
        dtype=float,
    )

    # Aktive timer: snø + vind over terskel
    active_mask = (df["Swe_hourly"] > snow_threshold) & (ws > v_threshold)
    n_hours = int(active_mask.sum())

    return float(Qt), sectors, Swe_total, n_hours


def make_wind_rose_figure(avg_sector_values: np.ndarray, title: str = "Snow transport wind rose (Qt)"):
    """
    Plot vindrose i Plotly basert på Qt [kg/m] per sektor.
    """
    n_sectors = len(avg_sector_values)
    # 16 kompassretninger
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'][:n_sectors]
    theta = np.linspace(0, 360, n_sectors, endpoint=False)

    fig = go.Figure(
        data=go.Barpolar(
            r=avg_sector_values,
            theta=theta,
            thetaunit="degrees",
            text=directions,
            hovertemplate="Dir: %{text}<br>Qt: %{r:.1f} kg/m<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                rotation=90,  # 0° ved nord
                tickmode="array",
                tickvals=theta,
                ticktext=directions,
            ),
            radialaxis=dict(
                title="Snow transport Qt [kg/m]",
                showline=True,
                linewidth=1,
            ),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

# ------------------------------------------------------------
# 5. UI: snøår og terskler
# ------------------------------------------------------------
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
    "For each snow year, ERA5 reanalysis data is fetched for this coordinate, "
    "and Tabler (2003) snow transport Qt [kg/m] is computed and aggregated by "
    "wind direction sector."
)

if start_year > end_year:
    st.error("Start year must be ≤ end year.")
    st.stop()

years_range = list(range(start_year, end_year + 1))

# ------------------------------------------------------------
# 6. Beregn Qt per år
# ------------------------------------------------------------
results = []
sector_sums = None
n_sectors = 16

with st.spinner("Downloading ERA5 data and computing snow transport (Qt)..."):
    for y in years_range:
        # Snøår: 1. juli y -> 30. juni y+1
        start_date = dt.date(y, 7, 1)
        end_date = dt.date(y + 1, 6, 30)

        df_season = fetch_era5_hourly(
            lat=lat,
            lon=lon,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        Qt, sectors, total_snow, n_hours = compute_snow_drift_for_season(
            df_season,
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
                "Qt_kg_per_m": Qt,
                "total_snow_mm": total_snow,
                "active_hours": n_hours,
            }
        )

if not results:
    st.warning("No years selected – nothing to compute.")
    st.stop()

df_results = pd.DataFrame(results)

if sector_sums is None or np.allclose(sector_sums, 0):
    avg_sector_values = np.zeros(n_sectors)
else:
    avg_sector_values = sector_sums / len(years_range)

# ------------------------------------------------------------
# 7. Plot: Qt per år + vindrose (Plotly)
# ------------------------------------------------------------
st.subheader("Results")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Snow transport Qt per snow year**")

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=df_results["snow_year"],
            y=df_results["Qt_kg_per_m"],
            mode="lines+markers",
            name="Qt (kg/m)",
        )
    )
    fig1.update_layout(
        xaxis_title="Snow year (July–June)",
        yaxis_title="Snow transport Qt [kg/m]",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.dataframe(df_results, use_container_width=True)

with col_right:
    st.markdown("**Wind rose for snow transport (average over selected years)**")
    fig2 = make_wind_rose_figure(avg_sector_values, title="Snow transport wind rose (Qt)")
    st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Qt is the mean annual snow transport per unit length of fence [kg/m], "
    "here estimated from ERA5 via Open-Meteo using a Tabler (2003)-style model."
)

# ------------------------------------------------------------
# 8. BONUS: Monthly snow transport (Qt) + Plotly
# ------------------------------------------------------------
st.header("📅 Monthly Snow Transport (Bonus)")

MONTH_LABELS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                "Jan", "Feb", "Mar", "Apr", "May", "Jun"]


def compute_monthly_snowtransport_for_year(
    snow_year_start: int,
    lat: float,
    lon: float,
    v_threshold: float,
    n_sectors: int,
) -> pd.DataFrame:
    """
    Compute monthly Tabler snow transport Qt for a snow year:
    1 July snow_year_start  – 30 June snow_year_start+1.

    Returns DataFrame with columns:
        snow_year, month_index (0..11), month_label, Qt_kg_per_m
    """
    records = []

    # (month_index, (year, month))
    month_years = list(enumerate(
        [(snow_year_start, m) for m in range(7, 13)] +
        [(snow_year_start + 1, m) for m in range(1, 7)]
    ))

    for idx, (y, m) in month_years:
        last_day = calendar.monthrange(y, m)[1]
        start_date = dt.date(y, m, 1)
        end_date = dt.date(y, m, last_day)

        df_m = fetch_era5_hourly(
            lat=lat,
            lon=lon,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        Qt_m, _, _, _ = compute_snow_drift_for_season(
            df_m,
            v_threshold=v_threshold,
            temp_threshold=0.0,
            snow_threshold=0.0,
            n_sectors=n_sectors,
        )

        records.append(
            {
                "snow_year": f"{snow_year_start}/{snow_year_start+1}",
                "month_index": idx,           # 0..11
                "month_label": MONTH_LABELS[idx],
                "Qt_kg_per_m": Qt_m,
            }
        )

    return pd.DataFrame(records)


# Beregn månedlig Qt for alle valgte snøår
monthly_frames = []
with st.spinner("Computing monthly snow transport (bonus)..."):
    for y in years_range:
        df_m = compute_monthly_snowtransport_for_year(
            snow_year_start=y,
            lat=lat,
            lon=lon,
            v_threshold=v_threshold,
            n_sectors=n_sectors,
        )
        monthly_frames.append(df_m)

df_monthly = pd.concat(monthly_frames, ignore_index=True)

st.subheader("Yearly vs monthly snow transport Qt")

fig3 = go.Figure()
for snow_year in df_monthly["snow_year"].unique():
    sub = df_monthly[df_monthly["snow_year"] == snow_year]
    fig3.add_trace(
        go.Scatter(
            x=sub["month_label"],
            y=sub["Qt_kg_per_m"],
            mode="lines+markers",
            name=snow_year,
        )
    )

fig3.update_layout(
    xaxis_title="Month in snow year (Jul–Jun)",
    yaxis_title="Monthly snow transport Qt [kg/m]",
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig3, use_container_width=True)

st.write("**Monthly snow transport (table):**")
st.dataframe(df_monthly, use_container_width=True)
