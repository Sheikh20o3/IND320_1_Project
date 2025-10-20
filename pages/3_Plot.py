# pages/3_Plot.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Data plot (Plotly)", page_icon="📈", layout="wide")
st.title("Data plot (Plotly)")

@st.cache_data
def load_data():
    """
    Prøv å finne open-meteo-subset.csv i repoet.
    Hvis ikke – lag trygge demo-data, så siden aldri krasjer i skyen.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "open-meteo-subset.csv",      # prosjektrot
        here.parent / "open-meteo-subset.csv",             # /pages
        Path.cwd() / "open-meteo-subset.csv",              # kjøremappe
        Path("open-meteo-subset.csv"),                     # relativt fallback
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df, f"Lest fra: {p}"

    # Fallback: generér en liten demo-dataserie
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

# Finn tidskolonne og numeriske kolonner
date_cols = [c for c in df.columns
             if pd.api.types.is_datetime64_any_dtype(df[c])
             or any(k in c.lower() for k in ["time", "date", "datetime", "timestamp"])]

if date_cols:
    tcol = date_cols[0]
    if not pd.api.types.is_datetime64_any_dtype(df[tcol]):
        # Prøv å parse til datetime om nødvendig
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

if not num_cols:
    st.error("Fant ingen numeriske kolonner å plotte.")
    st.stop()

# Velg serier
default_sel = num_cols[:2] if len(num_cols) >= 2 else [num_cols[0]]
ycols = st.multiselect("Velg måleserier", options=num_cols, default=default_sel)

if date_cols:
    tcol = date_cols[0]
    st.info(f"Tidskolonne: **{tcol}**")
    melt = df[[tcol] + ycols].melt(id_vars=tcol, var_name="series", value_name="value")
    fig = px.line(melt, x=tcol, y="value", color="series", title="Tidsserie")
    # Streamlit varsler om deprecations for use_container_width på dataframe.
    # For plotly går det fint å beholde enn så lenge:
    st.plotly_chart(fig, use_container_width=True)
else:
    # Uten tid: vis histogram av første valgte kolonne
    col = ycols[0]
    fig = px.histogram(df, x=col, nbins=50, title=f"Histogram av {col}")
    st.plotly_chart(fig, use_container_width=True)
