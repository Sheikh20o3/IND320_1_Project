# pages/5_Plot.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Data plot (Plotly)", page_icon="📈", layout="wide")
st.title("Data plot (Plotly)")

def _find_csv(name: str) -> Path | None:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / name,                       # project root
        here.parent / name,                              # /pages
        here.parent.parent / "Innlevering og IPYNB" / name,  # old location
        Path.cwd() / name,                               # CWD
        Path(name),                                      # relative
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, str]:
    csv_path = _find_csv("open-meteo-subset.csv")
    if csv_path:
        df = pd.read_csv(csv_path)
        return df, f"Lest fra: {csv_path}"
    # Fallback: liten demo-serie
    idx = pd.date_range("2021-01-01", periods=24 * 7, freq="H")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "time": idx,
        "temperature": 8 + np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 0.2, len(idx)),
        "wind": 2 + rng.normal(0, 0.5, len(idx)),
        "precip": np.clip(rng.normal(0.2, 0.1, len(idx)), 0, None),
    })
    return df, "Demo-data (CSV ikke funnet)"

df, source = load_data()
st.caption(source)

# Finn tids- og numeriske kolonner
date_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
if date_cols:
    tcol = date_cols[0]
    if not pd.api.types.is_datetime64_any_dtype(df[tcol]):
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")

num_cols = list(df.select_dtypes(include="number").columns)

if not num_cols:
    st.error("Fant ingen numeriske kolonner å plotte.")
    st.stop()

# Velg serier
default_sel = num_cols[:2] if len(num_cols) >= 2 else [num_cols[0]]
ycols = st.multiselect("Velg måleserier", options=num_cols, default=default_sel)

# Plott
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
