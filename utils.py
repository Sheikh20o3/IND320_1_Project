import os
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)  # Cache the function's output; suppress spinner while caching
def load_data():
    path = os.path.join(os.path.dirname(__file__), "/Users/a.h.sheikh/Desktop/IND320_Git_Job/IND320_1_Project/open-meteo-subset.csv")  # Build path to CSV in the same directory
    df = pd.read_csv(path)  # Load CSV into a DataFrame

    # Try to find a date/time-like column by scanning column names
    date_like = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])]
    if date_like:
        dcol = date_like[0]  # Use the first matching column
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")  # Parse to datetime; invalid values become NaT
        if df[dcol].notna().any():  # Proceed only if at least one valid datetime exists
            df = df.sort_values(dcol)  # Sort rows by the datetime column
            df["month"] = df[dcol].dt.to_period("M").astype(str)  # Derive a month label like '2025-10'
    else:
        df["month"] = "Unknown"  # Fallback when no date/time-like column is found

    return df  # Return the prepared DataFrame
