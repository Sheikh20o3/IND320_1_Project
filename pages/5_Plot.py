import streamlit as st
import pandas as pd
import plotly.express as px
from utils import download_open_meteo, get_selected_price_area

st.set_page_config(page_title="Data plot (Plotly)", page_icon="📈", layout="wide")
st.title("Data plot (Open-Meteo)")

PA = get_selected_price_area()
YEAR = 2021
VARS = ["temperature_2m","precipitation","wind_speed_10m","relative_humidity_2m","surface_pressure"]

@st.cache_data(show_spinner=True)
def _load(pa, year, vars_):
    return download_open_meteo(price_area=pa,
                               start_date=f"{year}-01-01",
                               end_date=f"{year}-12-31",
                               hourly=tuple(vars_))

df = _load(PA, YEAR, VARS)
if df.empty:
    st.warning("No data.")
    st.stop()

df["time"] = pd.to_datetime(df["time"])

num_cols = VARS
default_sel = num_cols[:2]
ycols = st.multiselect("Choose series", options=num_cols, default=default_sel)

melt = df[["time"] + ycols].melt(id_vars="time", var_name="series", value_name="value")
fig = px.line(melt, x="time", y="value", color="series", title=f"Time series – {PA}")
st.plotly_chart(fig, use_container_width=True)
