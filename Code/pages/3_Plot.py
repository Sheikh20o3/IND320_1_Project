import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data

st.set_page_config(page_title="Plot", page_icon="📈", layout="wide")
st.title("Data plot (Plotly)")

# Load data (cached in utils.load_data)
df = load_data()

# Find a date/time-like column
date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])]
date_col = date_cols[0] if date_cols else None
if date_col is None:
    st.error("No date/time column found - cannot build a time series.")
    st.stop()

# Ensure datetime dtype
if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Numeric columns
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.error("No numeric columns found to plot.")
    st.stop()

# Select column(s) to plot
choice = st.selectbox("Select column(s) to plot", ["All columns"] + num_cols, index=0)

# Select month range (default: first month)
if "month" in df.columns and df["month"].notna().any():
    months = sorted(df["month"].dropna().unique().tolist())
    default_month = months[0]
    start, end = st.select_slider("Select month(s)", options=months, value=(default_month, default_month))
    mask = (df["month"] >= start) & (df["month"] <= end)
    pdf = df.loc[mask].copy()
    subtitle = f"Months: {start} to {end}"
else:
    pdf = df.copy()
    subtitle = "All rows"

# Build Plotly figure
if choice == "All columns":
    long_df = pdf[[date_col] + num_cols].melt(id_vars=date_col, var_name="column", value_name="value")
    fig = px.line(long_df, x=date_col, y="value", color="column",
                  title=f"Time series - {subtitle}",
                  labels={date_col: "Date", "value": "Value", "column": "Column"})
else:
    fig = px.line(pdf, x=date_col, y=choice,
                  title=f"Time series - {choice} - {subtitle}",
                  labels={date_col: "Date", choice: "Value"})

st.plotly_chart(fig, use_container_width=True)
