# streamlit_app.py
import streamlit as st
from utils import get_selected_price_area, load_data

st.set_page_config(page_title="Introduction", page_icon="🌤️", layout="wide")

st.title("Introduction")
st.caption(
    "This app combines Elhub energy data with ERA5 / Open-Meteo weather data. "
    "Use the menu on the left to explore production, meteorology, snow drift, "
    "correlations and SARIMAX forecasting."
)

# Shows selected price area (chosen on the Elhub page)
area = get_selected_price_area()
st.info(f"Selected price area: {area}")

# Sidebar navigation in desired order
st.sidebar.header("Navigation")

# Link to this front page (optional, men greit å ha med)
st.sidebar.page_link("streamlit_app.py", label="Introduction")

st.sidebar.page_link("pages/2_Elhub_Production.py", label="Elhub Production")
st.sidebar.page_link("pages/3_Meteorology.py", label="Meteorology")
st.sidebar.page_link("pages/4_Data_Table.py", label="Data Table")
st.sidebar.page_link("pages/5_Map_PriceAreas.py", label="Map PriceAreas")
st.sidebar.page_link("pages/6_Snow_Drift.py", label="Snow Drift")
st.sidebar.page_link("pages/5_Plot.py", label="Plot")
st.sidebar.page_link("pages/6_Analysis.py", label="Analysis")
st.sidebar.page_link("pages/7_About.py", label="About")
st.sidebar.page_link(
    "pages/7_Sliding_Window_Correlation.py",
    label="Sliding Window Correlation",
)
st.sidebar.page_link(
    "pages/8_SARIMAX_Forecasting.py",
    label="SARIMAX Forecasting",
)

# Quick preview of data
df = load_data()
st.subheader("Quick look at the data")
st.dataframe(df.head(), use_container_width=True)
