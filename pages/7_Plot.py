# pages/5_Plot.py

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import download_open_meteo, get_selected_price_area

st.set_page_config(page_title="Data plot (Plotly)", page_icon="📈", layout="wide")
st.title("Data plot (Open-Meteo)")

# --- Valgt prisområde (delt med andre sider) ---
PA = get_selected_price_area()
YEAR = 2021

VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
    "surface_pressure",
]

@st.cache_data(show_spinner=True)
def _load(pa: str, year: int, vars_: list[str]) -> pd.DataFrame:
    return download_open_meteo(
        price_area=pa,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        hourly=tuple(vars_),
    )

df = _load(PA, YEAR, VARS)
if df.empty:
    st.warning("No data.")
    st.stop()

# --- Tidshåndtering ---
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).copy()

# 🔧 VIKTIG: konverter til ekte Python-datetime
min_time_ts = df["time"].min()
max_time_ts = df["time"].max()

min_time = min_time_ts.to_pydatetime()
max_time = max_time_ts.to_pydatetime()

# Slider for valg av tidsperiode
start_time, end_time = st.slider(
    "Select time window",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time),
    format="YYYY-MM-DD",
)

df_window = df[(df["time"] >= start_time) & (df["time"] <= end_time)].copy()
if df_window.empty:
    st.warning("No data in the selected time window.")
    st.stop()

# --- Valg av serier ---
num_cols = VARS
default_sel = num_cols[:2]
ycols = st.multiselect("Choose series", options=num_cols, default=default_sel)

if not ycols:
    st.info("Please select at least one series.")
    st.stop()

# --- Plot med Plotly ---
melt = df_window[["time"] + ycols].melt(
    id_vars="time",
    var_name="series",
    value_name="value",
)

fig = px.line(
    melt,
    x="time",
    y="value",
    color="series",
    title=f"Time series – {PA} ({start_time.date()} to {end_time.date()})",
)

st.plotly_chart(fig, use_container_width=True)
