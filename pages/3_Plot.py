# pages/3_Plot.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data  # felles loader som også søker i /data

st.set_page_config(page_title="Data plot (Plotly)", page_icon="📈", layout="wide")
st.title("Data plot (Plotly)")

def _make_demo_df() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=48, freq="H")
    s = pd.Series(range(48), dtype="float64")
    return pd.DataFrame(
        {
            "time": dates,
            "temperature": ((5.0 + 0.3 * s) % 10) + 2,  # fake data
            "wind": ((0.4 * s) % 8) + 1,
            "precip": ((0.1 * s) % 5),
        }
    )

# ---- Hent data (primært fra utils, med trygg fallback) ----
try:
    df = load_data()  # forventer å finne data/open-meteo-subset.csv
    source = "Lest via utils.load_data()"
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Ingen rader returnert")
except Exception:
    st.info("Fant ikke 'open-meteo-subset.csv' via utils. Viser demo-data som fallback.")
    df = _make_demo_df()
    source = "Demo-data (CSV ikke funnet)"

st.caption(source)

# ---- Finn tidskolonne og numeriske kolonner ----
date_cols = [
    c
    for c in df.columns
    if pd.api.types.is_datetime64_any_dtype(df[c])
    or any(k in c.lower() for k in ["time", "date", "datetime", "timestamp"])
]

if date_cols:
    tcol = date_cols[0]
    if not pd.api.types.is_datetime64_any_dtype(df[tcol]):
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

if not num_cols:
    st.error("Fant ingen numeriske kolonner å plotte.")
    st.stop()

# ---- Velg serier og plott ----
default_sel = num_cols[:2] if len(num_cols) >= 2 else [num_cols[0]]
ycols = st.multiselect("Velg måleserier", options=num_cols, default=default_sel)

if date_cols:
    tcol = date_cols[0]
    st.info(f"Tidskolonne: **{tcol}**")
    melt = df[[tcol] + ycols].melt(id_vars=tcol, var_name="series", value_name="value")
    fig = px.line(melt, x=tcol, y="value", color="series", title="Tidsserie")
    st.plotly_chart(fig, use_container_width=True)
else:
    col = ycols[0]
    fig = px.histogram(df, x=col, nbins=50, title=f"Histogram av {col}")
    st.plotly_chart(fig, use_container_width=True)
