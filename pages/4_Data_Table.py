import streamlit as st
import pandas as pd
from utils import download_open_meteo, get_selected_price_area

st.set_page_config(page_title="Data Table", page_icon="📄", layout="wide")
st.title("Data Table (Open-Meteo, 2021)")

PA = get_selected_price_area()
YEAR = 2021
VARS5 = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m", "surface_pressure"]

@st.cache_data(show_spinner=True)
def _load(pa: str, year: int, vars5: list[str]) -> pd.DataFrame:
    return download_open_meteo(
        price_area=pa,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        hourly=tuple(vars5),
        timezone="Europe/Oslo",
    )

df = _load(PA, YEAR, VARS5)
if df.empty:
    st.warning("Ingen data fra Open-Meteo.")
    st.stop()

missing = [c for c in VARS5 if c not in df.columns]
if missing:
    st.error(f"Mangler kolonner fra API: {missing}. Sjekk kall til Open-Meteo.")
    st.stop()

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"]).copy()
df["month"] = df["time"].dt.to_period("M").astype(str)

months = sorted(df["month"].unique())
if not months:
    st.warning("Fant ingen måneder i datasettet.")
    st.stop()

month = st.selectbox("Velg måned", months, index=0)
small = df[df["month"] == month][["time"] + VARS5].copy()

rows = [{"Metric": col, "Trend": small[col].astype(float).tolist()} for col in VARS5]
table = pd.DataFrame(rows)

spark_cfg = st.column_config.LineChartColumn(
    label="Trend",
    help=f"Tidsserie for {month} i {PA}",
)

st.dataframe(
    table,
    column_config={
        "Metric": st.column_config.TextColumn("Metric"),
        "Trend": spark_cfg,
    },
    hide_index=True,
    use_container_width=True,
)
