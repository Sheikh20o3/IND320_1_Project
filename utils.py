import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_data(path: str = "/Users/a.h.sheikh/Desktop/IND320_Git_Job/IND320_1_Project/CSV/open-meteo-subset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # prøv å finne en dato-/tid-kolonne
    date_like = [c for c in df.columns if any(k in c.lower() for k in ["date","time","datetime","timestamp"])]
    if date_like:
        dcol = date_like[0]
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        if df[dcol].notna().any():
            df = df.sort_values(dcol)
            df["month"] = df[dcol].dt.to_period("M").astype(str)
    else:
        df["month"] = "Unknown"
    return df
