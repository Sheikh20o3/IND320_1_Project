# streamlit_app.py
import streamlit as st
from utils import get_selected_price_area, load_data

st.set_page_config(page_title="Introduction", page_icon="🌤️", layout="wide")

st.title("Introduction")
st.caption(
    "This app combines Elhub energy data with ERA5 / Open-Meteo weather data. "
    "Use the pages in the sidebar to explore production, meteorology, snow drift, "
    "correlations and SARIMAX forecasting."
)

area = get_selected_price_area()
st.info(f"Selected price area: {area}")

df = load_data()
st.subheader("Quick look at the data")
st.dataframe(df.head(), use_container_width=True)
