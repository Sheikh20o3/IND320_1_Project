import os
import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data(show_spinner=False)  # Cache the function's output; suppress spinner while caching

@st.cache_data(show_spinner=False)
def load_data(csv_name: str = "open-meteo-subset.csv") -> pd.DataFrame:
    here = Path(__file__).resolve()
    candidates = [
        here.parent / csv_name,                 # prosjektrot
        here.parent / "data" / csv_name,       # /data
        here.parent / "pages" / csv_name,      # /pages
        Path.cwd() / csv_name,                 # arbeidskatalog i skyen
        Path(csv_name),                        # relativt
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Finn og parse en tidskolonne om mulig
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

    # Fallback: generér trygge demo-data så appen fungerer i skyen
    idx = pd.date_range("2021-01-01", periods=24*14, freq="H")
    s = pd.Series(range(len(idx)), dtype="float64")
    df = pd.DataFrame({
        "time": idx,
        "temperature": ((5.0 + 0.3 * s) % 10) + 2,
        "wind":       ((0.4 * s) % 8) + 1,
        "precip":     ((0.1 * s) % 5),
    })
    df["month"] = df["time"].dt.to_period("M").astype(str)
    return df
