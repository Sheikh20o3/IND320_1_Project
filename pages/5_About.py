
import streamlit as st # Imports the Streamlit library.

st.set_page_config(page_title="About", page_icon="ℹ️") # Configures the browser tab title 
st.title("About this app") # Displays the main title 

st.markdown("""
**In short:** A small, multi-page Streamlit app for exploring the `open-meteo-subset.csv` dataset.

**What you can do**
- Browse the raw data table.
- See per-row sparklines for the first month (using `st.column_config.LineChartColumn`).
- Build interactive time-series plots with **Plotly** — select a single column or all columns, and filter by month range.

**How it works**
- Reads a **local CSV** and caches it with `st.cache_data` for speed.
- Pages: Home, Data, Plot, About.
- Dependencies (see `requirements.txt`): `streamlit`, `pandas`, `numpy`, `plotly`.
""")