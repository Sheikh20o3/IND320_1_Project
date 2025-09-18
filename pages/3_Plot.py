import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data

st.set_page_config(page_title="Plot", page_icon="📈", layout="wide")
st.title("Plott av data")

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

# Velg kolonne(r) via selectbox
choice = st.selectbox("Velg kolonne(r) å plotte", ["All columns"] + num_cols, index=0)

# Velg månedsspenn via select_slider (default: første måned)
if "month" in df.columns and df["month"].notna().any():
    months = sorted(df["month"].dropna().unique().tolist())
    start, end = st.select_slider(
        "Velg måned(er)",
        options=months,
        value=(months[0], months[0])  # default: første måned
    )
    mask = (df["month"] >= start) & (df["month"] <= end)
    pdf = df.loc[mask].copy()
else:
    st.info("Fant ikke 'month'-kolonne – viser alle rader.")
    pdf = df.copy()

# Lag plott
fig, ax = plt.subplots()
ax.grid(True, alpha=0.3)
ax.set_title("Tidsserie")
ax.set_xlabel("Dato")
ax.set_ylabel("Verdi")

if choice == "All columns":
    for c in num_cols:
        ax.plot(pdf[date_col], pdf[c], label=c)
    ax.legend(loc="best", frameon=False)
else:
    ax.plot(pdf[date_col], pdf[choice], label=choice)
    ax.legend(loc="best", frameon=False)

st.pyplot(fig, clear_figure=True)
