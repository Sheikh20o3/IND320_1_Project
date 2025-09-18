import streamlit as st
import pandas as pd
from utils import load_data

st.set_page_config(page_title="Data Table", page_icon="📄", layout="wide")
st.title("Data stble")

df = load_data()

st.subheader("Rimport stst.dataframe(df, use_container_width=True)

# Finn dato-/tid-kolonne og numeriske kolonner
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
date_col = date_cols[0] if date_cols else None
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

months = df["month"].dropna().unique().tolist() if "month" in df.columns else []
months.sort()
first_month = months[0] if months else None

st.subheader(f"Første måned – minigrafer ({first_month})")

if first_month and date_col and num_cols:
    mdf = df[df["month"] == first_month].copy()
    rows = []
    # Hver RAD = én variabel; cellen inneholder liste over verdier i første måned
    for col in num_cols:
                 nd({"Metric": col, "First month trend": mdfimport streamlit as st
import panataFrame(rows)

    # Y-aksegrense for penere minigrafer
    y_min = float(pd.concat([mdf[c] for c in num_cols]).min(skipna=True))
    y_max = float(pd.concat([mdf[c] for c in num_cols]).max(skipna=True))

    st.dataframe(
        table,
        column_config={
            "Metric": st.column_config.TextColumn("Metric"),
            "First month trend": st.column_config.LineChartColumn(
                "First month trend",
                y_min=y_min, y_max=y_max,
                help="Radvis sparkline for første måned"
            ),
        },
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Finner ikke nødvendig dato-/tid-kolonne og numeriske kolonner. Viser kun rådata.")
