import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="About / Project Details", page_icon="ℹ️")
st.title("About this Project (IND320, Part 3)")
st.caption("Building on Part 2: Time Series Analysis and Anomaly Detection")

# --- 1. Application and Functionality ---
st.header("1. Application Scope and Features")

st.markdown(f"""
This Streamlit application has been significantly expanded to incorporate advanced **Time Series Analysis** and **Anomaly/Outlier Detection** techniques, using both **Elhub electricity production data (2021)** and **meteorological reanalysis data (2021)**.

### New Features in the App (Pages 'Time Series' and 'Weather Analysis')

* **Time Series Analysis (Page 'new A')**:
    * **Seasonal-Trend decomposition using LOESS (STL)**: Decomposes Elhub production data into seasonal, trend, and residual components.
    * **Spectrogram**: Visualizes the frequency components of the Elhub production time series.
* **Weather Analysis (Page 'new B')**:
    * **Statistical Process Control (SPC)**: Detects outliers in **temperature** data using High-Pass Filtering (DCT) and robust statistical boundaries.
    * **Local Outlier Factor (LOF)**: Identifies anomalies in **precipitation** data.

The main interactive page, **Elhub Production** (`4_Elhub_Production.py`), still allows users to **Select Price Area** (e.g., NO1, NO2, NO5), which now drives the analysis on the new pages 'new A' and 'new B' for the respective locations.
""")

# --- 2. Data Flow and Technology Stack ---
st.header("2. Data Pipeline and Technology")

st.subheader("Data Sources")
st.markdown("""
The project now integrates two primary data sources:

1.  **Elhub API (Reused from Part 2)**: Hourly electricity production data for all Norwegian price areas for **2021**. Data is stored in **MongoDB Atlas**.
2.  **Open-Meteo API (New)**: Historical **ERA5 reanalysis data** for key weather properties (e.g., temperature, precipitation) for the year **2021**. Data is downloaded **live** within the Streamlit app based on the selected price area's representative city (Oslo, Bergen, Kristiansand, Trondheim, or Tromsø).
""")

st.subheader("Database Integration and Live API Access")
st.markdown("""
* **Elhub Data**: The pipeline from Part 2 (Jupyter Notebook -> Spark/Cassandra -> **MongoDB Atlas**) is reused for the production data.
* **Weather Data**: The Streamlit app now directly calls the **Open-Meteo API** (via a new API wrapper function) to retrieve the meteorological data, replacing the previous local CSV import. This ensures the app uses the latest required methodology.
""")

st.subheader("Code Structure Updates")
st.markdown("""
Key files and their roles in this part of the project:
* `project_part_3.ipynb`: The main development and documentation platform, containing functions for all new analyses: API wrapper, SPC/DCT, LOF, STL decomposition, and Spectrogram.
* `utils_elhub.py`: Continues to hold the MongoDB connection logic and now includes the **Open-Meteo API wrapper function**.
* **Streamlit Pages**:
    * The page structure has been reorganized: `1, 4, new A, 2, 3, new B, 5`.
    * Two new pages contain the core analysis: **'new A' (Time Series)** and **'new B' (Weather Analysis)**, both using `st.tabs()` to present multiple plots and statistics.
""")

st.subheader("Log and AI Usage")
st.markdown("""
A **300-500 word log** detailing the compulsory work (Jupyter Notebook and Streamlit experience) has been included in the `project_part_3.ipynb` file, along with a brief description of **AI usage** during development.
""")

# --- 3. Git Log (Repository History) ---
st.header("3. Git Repository Log")

st.info("The log below shows the last 20 commits, executed on a temporary branch for safety before final merge.")

try:
    # Attempt to run git log command to display commit history
    result = subprocess.check_output(
        ["git", "log", "--pretty=format:%h|%an|%ad|%s", "--date=short", "-n", "20"],
        text=True
    )

    # Parse output into a DataFrame for clean display
    rows = [line.split("|") for line in result.split("\n") if line]
    df_log = pd.DataFrame(rows, columns=["Commit Hash", "Author", "Date", "Message"])
    
    st.dataframe(df_log, use_container_width=True)

except FileNotFoundError:
    st.warning("Could not run `git log`. This script needs to be executed within a Git repository to display history.")
except Exception as e:
    st.error(f"An error occurred while fetching Git log: {e}")


# --- 4. Contact Information ---
st.header("4. Contact Information")
st.markdown(f"Should you have any questions, comments, or feedback regarding this project, please contact me via email:")
st.markdown(f"**Email:** [abdul.haadi.sheikh@nmbu.no](mailto:abdul.haadi.sheikh@nmbu.no)")
