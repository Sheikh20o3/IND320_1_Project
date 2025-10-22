import streamlit as st
import pandas as pd
from utils import load_data

st.set_page_config(page_title="Data Table", page_icon="📄", layout="wide")
st.title("Data Table")

df = load_data()

st.subheader("Raw data")
st.dataframe(df, use_container_width=True)

# Find datetime-like and numeric columns by scanning names and dtypes
date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])]
date_col = date_cols[0] if date_cols else None
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# Row-wise sparklines
if "month" in df.columns and date_col and num_cols:
    months = sorted([m for m in df["month"].dropna().unique()])
    if months:
        first_month = months[0]
        mdf = df[df["month"] == first_month].copy()

        # Prepare the table: one row per numeric column
        rows = [{"Metric": col, "First month trend": mdf[col].tolist()} for col in num_cols]
        table = pd.DataFrame(rows)

        st.subheader(f"First month — sparklines ({first_month})")

        # IMPORTANT: Do not set y_min/y_max -> Streamlit will scale independently per row for LineChartColumn
        # This avoids a shared y-axis that can flatten smaller-variance series into near-straight lines.
        spark_cfg = st.column_config.LineChartColumn(
            "First month trend",
            help="Row-wise sparkline for the first month (independent y-scale per row)",
        )

        st.dataframe(
            table,
            column_config={
                "Metric": st.column_config.TextColumn("Metric"),  # Label for the metric/column name
                "First month trend": spark_cfg,                   # Sparkline showing the month’s sequence
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No month found to display sparklines.")
else:
    # Guard clause: we need a detected date/time column, at least one numeric column, and a 'month' column
    st.info("A date/time column and numeric columns are required to build sparklines.")
