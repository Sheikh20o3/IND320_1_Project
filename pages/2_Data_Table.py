import streamlit as st 
import pandas as pd 
from utils import load_data 

st.set_page_config(page_title="Data Table", page_icon="📄", layout="wide") # Sets the page title and uses a wide layout.
st.title("Data Table") # Displays the main header.

df = load_data() # Loads the DataFrame

st.subheader("Raw data") # Displays a subheading.
st.dataframe(df, use_container_width=True) # Displays the full raw DataFrame in an interactive Streamlit table.

# Find datetime-like and numeric columns
date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])] # Identifies columns suitable for time series analysis.
date_col = date_cols[0] if date_cols else None
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# Show row-wise sparklines
if "month" in df.columns and date_col and num_cols: # Checks if month, date, numeric data are met.
    months = sorted([m for m in df["month"].dropna().unique()])
    if months:
        first_month = months[0]
        mdf = df[df["month"] == first_month].copy() # Filters the data to include only the first month.

        # Prepare data for the sparkline table: each numeric column becomes a row.
        rows = [{"Metric": col, "First month trend": mdf[col].tolist()} for col in num_cols]
        table = pd.DataFrame(rows)

        # Calculate global min/max across all numeric columns in the first month.
        y_min = float(pd.concat([mdf[c] for c in num_cols]).min(skipna=True))
        y_max = float(pd.concat([mdf[c] for c in num_cols]).max(skipna=True))

        st.subheader(f"First month — sparklines ({first_month})") # Displays a subheading for the sparkline table.
        st.dataframe(
            table,
            column_config={
                "Metric": st.column_config.TextColumn("Metric"), # Configures the 'metric' column as standard text.
                "First month trend": st.column_config.LineChartColumn( 
                    "First month trend",
                    y_min=y_min, y_max=y_max, # Ensures all sparklines share the same y-axis scale for comparison.
                    help="Row-wise sparkline for the first month",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No month found to display sparklines.") # Fallback if month column exists but is empty.
else:
    st.info("A date/time column and numeric columns are required to build sparklines.") # Fallback if necessary data types are missing.