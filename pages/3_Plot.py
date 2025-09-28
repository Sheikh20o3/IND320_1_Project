import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data

st.set_page_config(page_title="Plot", page_icon="📈", layout="wide") # Sets the page configuration to use a wide layout for the chart.
st.title("Data plot (Plotly)") # Displays the main header 

# Load data (cached in utils.load_data)
df = load_data() # Loads the full dataset, 

# Find a date/time-like column
date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])] # Finds column names that indicate time series data.
date_col = date_cols[0] if date_cols else None
if date_col is None: # Checks if a primary time column was successfully identified.
    st.error("No date/time column found - cannot build a time series.") # Displays an error if the X-axis data is missing.
    st.stop() # Halts execution since the core plot cannot be generated.

# Ensure datetime dtype
if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce") # Converts the time column to the required datetime object type.

# Numeric columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])] # Filters the columns to find all available numeric data for plotting.
if not num_cols:
    st.error("No numeric columns found to plot.") # Displays an error if there are no Y-axis.
    st.stop() # Stops execution.

# Select columns to plot
choice = st.selectbox("Select column(s) to plot", ["All columns"] + num_cols, index=0) # Creates an interactive dropdown menu for column selection.

# Select month range 
if "month" in df.columns and df["month"].notna().any(): # Checks for the pre-calculated 'month' column 
    months = sorted(df["month"].dropna().unique().tolist())
    default_month = months[0]
    start, end = st.select_slider("Select month(s)", options=months, value=(default_month, default_month)) # Creates an interactive slider 
    mask = (df["month"] >= start) & (df["month"] <= end) 
    pdf = df.loc[mask].copy() 
    subtitle = f"Months: {start} to {end}"
else:
    pdf = df.copy() # Uses the entire dataset if no 'month' column exists.
    subtitle = "All rows"

# Build Plotly figure
if choice == "All columns": # Executes the logic for plotting multiple series.
    long_df = pdf[[date_col] + num_cols].melt(id_vars=date_col, var_name="column", value_name="value") # Pivots the table from wide to long format, which Plotly requires for multi-line plots.
    fig = px.line(long_df, x=date_col, y="value", color="column", # Creates the Plotly line chart with multiple colored lines.
                  title=f"Time series - {subtitle}",
                  labels={date_col: "Date", "value": "Value", "column": "Column"})
else: # Executes the logic for plotting a single series.
    fig = px.line(pdf, x=date_col, y=choice, # Creates a standard Plotly line chart for the selected column.
                  title=f"Time series - {choice} - {subtitle}",
                  labels={date_col: "Date", choice: "Value"})

st.plotly_chart(fig, use_container_width=True) # Renders the Plotly figure within the Streamlit app, adjusting to the container width. allowing for fast reloading on subsequent accesses.