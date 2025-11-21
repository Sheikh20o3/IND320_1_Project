# streamlit_app.py
import streamlit as st
from utils import get_selected_price_area, load_data

st.set_page_config(page_title="Introduction", page_icon="🌤️", layout="wide")

st.title("Introduction")

st.caption(
    "Interactive exploration of Norwegian electricity production/consumption "
    "and meteorology using Elhub and ERA5 / Open-Meteo data."
)

# --- Intro text about the project/app ---
st.markdown(
    """
This Streamlit application is the main user interface for our IND320 project,
where we combine **energy data from Elhub** with **meteorological reanalysis
from ERA5 via the Open-Meteo API**.

The backend work is done in Jupyter Notebooks using **Spark, Cassandra and MongoDB**:
Elhub API data (hourly production and consumption for price areas NO1–NO5,
years **2021–2024**) are downloaded, cleaned and stored in the databases.
The app then reads the curated data and presents them through a set of
interactive analysis pages.

### Data sources

- **Elhub API / CSV**  
  - Hourly **production per group and price area**  
  - Hourly **consumption per group and price area**  
  - Years **2021–2024**, aggregated and stored in Cassandra and MongoDB

- **ERA5 / Open-Meteo**  
  - Hourly temperature, wind, precipitation, snowfall, etc.  
  - Either for a **selected price area** or for **user-clicked coordinates**  
  - Used for correlation analysis, snow drift modelling and forecasting

### Overview of the pages (left menu)

- **Elhub Production**  
  Explore hourly electricity production by price area and production group,
  using interactive Plotly figures.

- **Meteorology**  
  Analyse ERA5 weather series with **STL decomposition** and **spectrograms**
  for a chosen price area and year.

- **Data Table**  
  Compact Open-Meteo overview where each variable is shown as a small
  **sparkline** (LineChartColumn) for a selected month.

- **Map PriceAreas**  
  Folium/GeoJSON map of price areas NO1–NO5.  
  Price areas are coloured by **mean production/consumption** over a selected
  time interval, and any **clicked coordinate is stored** for use on the
  Snow Drift page.

- **Snow Drift**  
  Uses the stored coordinate to compute a **snow-drift index** per
  *snow year* (1 July – 30 June) and a corresponding **wind rose**.
  Also includes a bonus view with **monthly snow drift**.

- **Plot**  
  General Open-Meteo time-series viewer with interactive Plotly lines for
  selected meteorological variables.

- **Analysis**  
  “Advanced analysis” page:
  - Robust **SPC-style outlier detection** for temperature (SATV + bounds)
  - **Local Outlier Factor (LOF)** anomaly detection for precipitation

- **Sliding Window Correlation**  
  Correlates a selected meteorological variable with **production or
  consumption** for a chosen price area and year.  
  Both **lag** and **window length** are user-controlled, and the page shows
  the raw series together with the **sliding correlation**.

- **SARIMAX Forecasting**  
  Forecasts hourly **production or consumption** with a configurable
  **SARIMAX model**:
  - User-selectable ARIMA and seasonal orders
  - Training period and forecast horizon
  - Optional **exogenous weather variables** (ERA5) as regressors
  - Plot with forecast and **confidence intervals**

- **About**  
  Summary of the project context, technology stack and how the different
  pages fit together.

Use the sidebar to move between these pages and experiment with the
controls (price area, year, variables, lags, windows, thresholds, etc.).
Many of the findings and interpretations are documented in the accompanying
Jupyter Notebook log.
"""
)

# --- Selected price area info ---
area = get_selected_price_area()
st.info(f"Selected price area (shared across pages): **{area}**")

# --- Quick data preview from utility loader ---
df = load_data()
st.subheader("Quick look at one of the curated datasets")
st.dataframe(df.head(), use_container_width=True)
