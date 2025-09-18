import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data

st.set_page_config(page_title="Plot", page_icon="📈", layout="wide")
st.title("Plott av data")

df = load_data()

# Identifiser dato/tid og numeriske kolonner
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
date_col = date_cols[0] if date_cols else None
if date_col is None:
    st.error("Fant ingen dato-/tid-kolonne – kan ikke lage tidsserie.")
    st.stop()

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.error("Fant ingen numeriske kolonner å plotte.")
    st.stop()

# Velg kolonne(r)
choice = st.selectbox("Velg kolonne(r) å plotte", ["All columns"] + num_cols, index=0)

# Velg månedsspenn med select_slider
months = df["month"].dropna().unique().tolist() if "month" in df.columns else []
months.sort()
if months:
    start, end = st.select_slider(
        "Velg måned(er)",
        optionimport stream     value=(months[0], months[0])  # default: første måned
    )
    mask = (df["month"] >= start) & (df["month"] <= end)
    pdf = df.loc[mask].copy()
else:
    pdf = df.copy()

fig = plt.figure()
ax = plt.gca()
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
