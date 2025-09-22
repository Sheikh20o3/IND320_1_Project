import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_data

st.set_page_config(page_title="Plot", page_icon="📈", layout="wide")
st.title("Plott av data (Plotly)")

# Last data (med caching i utils.load_data)
df = load_data()

# Finn dato-/tid-kolonne
date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])]
date_col = date_cols[0] if date_cols else None
if date_col is None:
    st.error("Fant ingen dato-/tid-kolonne – kan ikke lage tidsserie.")
    st.stop()

# Sørg for datetime-type
if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Numeriske kolonner
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.error("Fant ingen numeriske kolonner å plotte.")
    st.stop()

# Velg kolonne(r)
choice = st.selectbox("Velg kolonne(r) å plotte", ["All columns"] + num_cols, index=0)

# Velg månedsspenn (default: første måned)
if "month" in df.columns and df["month"].notna().any():
    months = sorted(df["month"].dropna().unique().tolist())
    start, end = st.select_slider("Velg måned(er)", options=months, value=(months[0], months[0]))
    mask = (df["month"] >= start) & (df["month"] <= end)
    pdf = df.loc[mask].copy()
    subtitle = f"Måneder: {start} → {end}"
else:
    pdf = df.copy()
    subtitle = "Alle rader"

# Plotly-figur
if choice == "All columns":
    long_df = pdf[[date_col] + num_cols].melt(id_vars=date_col, var_name="variable", value_name="value")
    fig = px.line(
        long_df, x=date_col, y="value", color="variable",
        title=f"Tidsserie – {subtitle}",
        labels={date_col: "Dato", "value": "Verdi", "variable": "Kolonne"},
    )
else:
    fig = px.line(
        pdf, x=date_col, y=choice,
        title=f"Tidsserie – {choice} – {subtitle}",
        labels={date_col: "Dato", choice: "Verdi"},
    )

st.plotly_chart(fig, use_container_width=True)