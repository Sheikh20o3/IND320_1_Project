# 7_about.py — IND320 Assignment 3

import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="About / Project Details (Part 3)", page_icon="ℹ️")
st.title("About this Project (IND320, Part 3)")

# --- 1. Application Scope and Features ---
st.header("1. Application Scope and Features")

st.markdown("""
This Streamlit app is the **online companion for IND320 – Assignment 3**.  
It brings together **Elhub production data (2021)** and **meteorological reanalysis (ERA5 via Open-Meteo)** and adds the required analyses:

- A **global Price Area selector** (moved early in the navigation) controls which area is used across pages.
- **Meteorology (Open-Meteo, ERA5)**:
  - API-based download (no CSV import) driven by the selected price area and **year = 2021** for the app.
  - Variables include at minimum 2 m temperature and precipitation (extendable).
- **Production (Elhub 2021)** from **MongoDB Atlas** for interactive plots (pie/time series).
- **New analysis pages**:
  - **STL decomposition** (on Elhub production) with configurable parameters.
  - **Spectrogram** (on Elhub production) with window length/overlap controls.
  - **Outlier/SPC** on temperature using DCT high-pass to create **SATV** and robust control limits.
  - **Anomaly/LOF** on precipitation with configurable outlier proportion (default 1%).
""")

st.info(
    "Notebook requirement: In the Jupyter Notebook, the Open-Meteo function is "
    "applied to **Bergen for year 2019**. In the Streamlit app, the download uses the selected area with **year 2021**."
)

# --- 2. Data Pipeline and Technology ---
st.header("2. Data Pipeline and Technology")

st.subheader("Meteorology: Open-Meteo ERA5 API")
st.markdown("""
- API docs: https://open-meteo.com/en/docs  
- The app calls a reusable function (e.g., `download_open_meteo(lon, lat, year)`) to fetch ERA5 reanalysis.  
- **In this app:** the **price area selector** provides the area → city/coords mapping → API download for **2021**.  
- **In the Notebook:** the same function is used for **Bergen, 2019** to satisfy the assignment requirement.
""")

st.subheader("Elhub Production Data (2021) via MongoDB")
st.markdown("""
- Hourly production (2021), originally retrieved in Part 2 and loaded to **MongoDB Atlas**.  
- The app uses helper functions in `utils_elhub.py` (e.g., `fetch_pie_df`, `fetch_line_df`) to query:
  - **Pie chart**: total annual production per group for selected price area.
  - **Time series**: monthly breakdown per production group for comparisons.
""")

st.subheader("Analytical Methods and Parameters")
st.markdown("""
- **STL decomposition (LOESS)** on production:
  - Parameters with sensible defaults: price area, production group, period length, seasonal smoother, trend smoother, **robust** (True/False).
  - Wrapped in a function that returns the plot.
- **Spectrogram** on production:
  - Parameters: price area, production group, window length, window overlap.
  - Wrapped in a function that returns the plot.
- **Outlier/SPC on temperature**:
  - **DCT high-pass** → **SATV** (seasonally adjusted temperature variation).
  - Robust limits via median/MAD; parameters: frequency cut-off and k-sigma.
  - Plots inliers/outliers (SATV is used for limits only; not plotted directly).
- **Anomaly/LOF on precipitation**:
  - Parameter: contamination (default **1%**).  
  - Returns plot + summary of detected anomalies.
""")

# --- 3. Page Structure and Navigation ---
st.header("3. Page Structure and Navigation")

st.markdown("""
To align with Assignment 3 requirements, pages were reordered and extended:

