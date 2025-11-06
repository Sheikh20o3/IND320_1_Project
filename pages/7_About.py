# 7_about.py

import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="About / Project Details", page_icon="ℹ️")
st.title("About this Project (IND320, Part 2)")

# --- 1. Application and Functionality ---
st.header("1. Application Scope and Features")

st.markdown(f"""
This Streamlit application visualizes **hourly electricity production data for Norway for the entire year 2021**, retrieved from the **Elhub API**.

The main interactive page, **Elhub Production** (`4_Elhub_Production.py`), allows users to:

* **Select Price Area** (e.g., NO1, NO2, NO5) using radio buttons.
* View a **Pie Chart** displaying the total annual production broken down by production group for the selected area.
* Filter on **Production Groups** and **Month** to display a detailed **Time-Series Line Plot** comparing group production over that month.
""")

# --- 2. Data Flow and Technology Stack ---
st.header("2. Data Pipeline and Technology")

st.subheader("Data Source: Elhub API")
st.markdown("""
All data is sourced from the **Elhub API** (`https://api.elhub.no/`), specifically using the endpoint:
* `PRODUCTION_PER_GROUP_MBA_HOUR`

The API was queried to fetch hourly production for all price areas across the entirety of **2021**.
""")

st.subheader("Database Integration (Cassandra & MongoDB)")
st.markdown("""
The data pipeline involves multiple technologies to fulfill the requirements:
1.  **Jupyter Notebook** (`elhub_2021_pipeline.ipynb`) retrieves raw data.
2.  Data is initially loaded into the local **Cassandra** database using **Spark** integration.
3.  The curated subset (`priceArea`, `productionGroup`, `startTime`, `quantityKwh`) is then read via Spark and loaded into **MongoDB Atlas** (the remote database accessible by the Streamlit app).
4.  This Streamlit app connects directly to **MongoDB** via `utils_elhub.py` to fetch the necessary data for visualization.
""")

st.subheader("Code Structure")
st.markdown("""
The project is structured across several key files:
* `elhub_2021_pipeline.ipynb`: Contains the core logic for data extraction, Spark processing (to Cassandra), plotting, and final loading into MongoDB.
* `load_to_mongo.py`: A standalone script executed to push the Spark-processed data from Cassandra into MongoDB.
* `utils_elhub.py`: Contains crucial helper functions, including the **MongoDB connection logic** and data fetching functions (`fetch_pie_df`, `fetch_line_df`) used by the Streamlit pages.
* `4_Elhub_Production.py`: The main Streamlit page that handles user input and renders the required visualizations using data functions from `utils_elhub.py`.
""")

# --- 3. Git Log (Repository History) ---
st.header("3. Git Repository Log")

st.info("The log below shows the last 20 commits for review, demonstrating version control history.")

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



Dette er min about page som forklarer hva oppgaven går ut på, endre denne til å passe oppgaven. Engelsk
