# pages/2_Data_Table.py (for example)

import streamlit as st
import pandas as pd
from utils import download_open_meteo, get_selected_price_area

st.set_page_config(page_title="Data Table", page_icon="📄", layout="wide")
st.title("Data Table (Open-Meteo)")

# --- Shared price area (from global selector) ---
PA = get_selected_price_area()

# --- Year and variables ---
YEARS = list(range(2019, 2025))
YEAR = st.selectbox("Year", YEARS, index=YEARS.index(2021))

VARS5 = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "surface_pressure",
]

@st.cache_data(show_spinner=True)
def _load(pa: str, year: int, vars5: list[str]) -> pd.DataFrame:
    """Download one year of hourly ERA5 (Open-Meteo) data for a price area."""
    return download_open_meteo(
        price_area=pa,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        hourly=tuple(vars5),
        timezone="Europe/Oslo",
    )

df = _load(PA, YEAR, VARS5)

if df.empty:
    st.warning("No data returned from Open-Meteo for this selection.")
    st.stop()

missing = [c for c in VARS5 if c not in df.columns]
if missing:
    st.error(f"Missing columns from API: {missing}. Please check the Open-Meteo call.")
    st.stop()

# --- Time processing and month selection ---
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).copy()
df["month"] = df["time"].dt.to_period("M").astype(str)

months = sorted(df["month"].unique())
if not months:
    st.warning("Did not find any months in the dataset.")
    st.stop()

month = st.selectbox("Choose month", months, index=0)
small = df[df["month"] == month][["time"] + VARS5].copy()

# --- Build table with LineChartColumn (interactive sparkline) ---
rows = [
    {
        "Metric": col,
        "Trend": small[col].astype(float).tolist(),
    }
    for col in VARS5
]
table = pd.DataFrame(rows)

spark_cfg = st.column_config.LineChartColumn(
    label="Trend",
    help=f"Hourly time series for {month} in {PA}, year {YEAR}",
)

st.caption(f"Active price area: **{PA}**, year: **{YEAR}**")

st.dataframe(
    table,
    column_config={
        "Metric": st.column_config.TextColumn("Metric"),
        "Trend": spark_cfg,
    },
    hide_index=True,
    use_container_width=True,
)
