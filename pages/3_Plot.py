# pages/3_Plot.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from utils import load_data  # ← bruk felles loader

st.set_page_config(page_title="Data plot (Plotly)", page_icon="📈", layout="wide")
st.title("Data plot (Plotly)")
df = load_data()

def _find_csv(name: str):
    # Search for the CSV in repo root, within the pages/ folder, and in the current working directory
    candidates = [
        Path(__file__).resolve().parents[1] / name,  # repo root
        Path(__file__).resolve().parent / name,      # pages/
        Path.cwd() / name,                           # CWD
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = _find_csv("open-meteo-subset.csv")
    if csv_path:
        return pd.read_csv(csv_path)

    # Fallback: small demo dataset so the page also works in the cloud without the CSV
    dates = pd.date_range("2021-01-01", periods=48, freq="H")
    s = pd.Series(range(48), dtype="float64")
    df = pd.DataFrame({
        "time": dates,
        "temperature": ((5.0 + 0.3 * s) % 10) + 2,  # fake data
        "wind":       ((0.4 * s) % 8) + 1,
        "precip":     ((0.1 * s) % 5),
        "month": dates.month,
        "day":   dates.day,
        "hour":  dates.hour,
    })
    st.info("Fant ikke 'open-meteo-subset.csv' i repoet. Viser demo-data som fallback.")
    return df

df = load_data()


@st.cache_data
def load_data():
    """
    Try to find open-meteo-subset.csv in the repo.
    If not found — generate safe demo data so the page never crashes in the cloud.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "open-meteo-subset.csv",      # project root
        here.parent / "open-meteo-subset.csv",             # /pages
        Path.cwd() / "open-meteo-subset.csv",              # execution directory
        Path("open-meteo-subset.csv"),                     # relative fallback
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df, f"Lest fra: {p}"

    # Fallback: generate a small demo time series
    idx = pd.date_range("2021-01-01", periods=24*7, freq="H")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "time": idx,
        "temp": 5 + rng.normal(0, 1, size=len(idx)).cumsum() / 10,
        "wind": rng.normal(0, 1, size=len(idx)).cumsum() / 10,
    })
    return df, "Demo-data (CSV ikke funnet i repo)"

df, source = load_data()
st.caption(source)

# Find the time column and numeric columns
date_cols = [c for c in df.columns
             if pd.api.types.is_datetime64_any_dtype(df[c])
             or any(k in c.lower() for k in ["time", "date", "datetime", "timestamp"])]

if date_cols:
    tcol = date_cols[0]
    if not pd.api.types.is_datetime64_any_dtype(df[tcol]):
        # Try to parse to datetime if needed
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

if not num_cols:
    st.error("Fant ingen numeriske kolonner å plotte.")
    st.stop()

# Select series for plotting
default_sel = num_cols[:2] if len(num_cols) >= 2 else [num_cols[0]]
ycols = st.multiselect("Velg måleserier", options=num_cols, default=default_sel)

if date_cols:
    tcol = date_cols[0]
    st.info(f"Tidskolonne: **{tcol}**")
    melt = df[[tcol] + ycols].melt(id_vars=tcol, var_name="series", value_name="value")
    fig = px.line(melt, x=tcol, y="value", color="series", title="Tidsserie")
    # Streamlit shows a deprecation warning for use_container_width on dataframes.
    # For Plotly, it is fine to keep it for now:
    st.plotly_chart(fig, use_container_width=True)
else:
    # Without a time column: show a histogram of the first selected column
    col = ycols[0]
    fig = px.histogram(df, x=col, nbins=50, title=f"Histogram av {col}")
    st.plotly_chart(fig, use_container_width=True)
