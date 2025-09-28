import streamlit as st
import pandas as pd
from utils import load_data

st.set_page_config(page_title="Data Table", page_icon="📄", layout="wide")
st.title("Data Table")

df = load_data()

st.subheader("Raw data")
st.dataframe(df, use_container_width=True)

# Find datetime-like and numeric columns
date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])]
date_col = date_cols[0] if date_cols else None
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# Show row-wise sparklines (LineChartColumn) for the first month if available
if "month" in df.columns and date_col and num_cols:
    months = sorted([m for m in df["month"].dropna().unique()])
    if months:
        first_month = months[0]
        mdf = df[df["month"] == first_month].copy()

        rows = [{"Metric": col, "First month trend": mdf[col].tolist()} for col in num_cols]
        table = pd.DataFrame(rows)

        y_min = float(pd.concat([mdf[c] for c in num_cols]).min(skipna=True))
        y_max = float(pd.concat([mdf[c] for c in num_cols]).max(skipna=True))

        st.subheader(f"First month — sparklines ({first_month})")
        st.dataframe(
            table,
            column_config={
                "Metric": st.column_config.TextColumn("Metric"),
                "First month trend": st.column_config.LineChartColumn(
                    "First month trend",
                    y_min=y_min, y_max=y_max,
                    help="Row-wise sparkline for the first month",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No month found to display sparklines.")
else:
    st.info("A date/time column and numeric columns are required to build sparklines.")
