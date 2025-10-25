# pages/4_Elhub_Production.py
import calendar
import streamlit as st
import pandas as pd
import plotly.express as px



from utils_elhub import get_client

try:
    from utils_elhub import get_client, uri_preview
except Exception:
    from utils_elhub import get_client
    def uri_preview():
        return "uri_preview() ikke tilgjengelig i denne builden"

try:
    _ = get_client()
    st.success("MongoDB ping OK")
except Exception as e:
    st.error(f"Mongo-tilkobling feilet: {e}")




from utils_elhub import (
    list_price_areas,
    list_groups,
    fetch_pie_df,
    fetch_line_df,
)

st.title("Elhub Production – 2021")  # Page title


# (Duplicate imports below are kept intentionally; not modifying code structure)
import streamlit as st
from utils_elhub import get_client

try:
    from utils_elhub import get_client, uri_preview
except Exception:
    from utils_elhub import get_client
    def uri_preview():
        return "uri_preview() ikke tilgjengelig i denne builden"


# Test the MongoDB connection and show status in the UI
with st.status("Testing MongoDB connection", expanded=False):
    try:
        cli = get_client()
        cnt = cli["elhub"]["production_2021_by_hour"].count_documents({})
        st.success(f"Connected to MongoDB ✅  Documentation: {cnt:,}")
    except Exception as e:
        st.error("Can not se that it is connected to MongoDB.")
        st.exception(e)
        st.stop()


# --- UI: price area selection (radio in left column) ---
areas = list_price_areas()
if not areas:
    st.error("Didn't find any priceArea in MongoDB. Check that you have loaded data into elhub.production_2021_by_hour.")
    st.stop()

left, right = st.columns(2)

with left:
    area = st.radio("Select the price area", areas, index=0, horizontal=True)

    # Fetch data for the pie chart
    pie_df = fetch_pie_df(area).copy()

    # Failsafe: some earlier variants may have 'totalKwh' instead of 'quantityKwh'
    if "totalKwh" in pie_df.columns and "quantityKwh" not in pie_df.columns:
        pie_df.rename(columns={"totalKwh": "quantityKwh"}, inplace=True)

    if pie_df.empty:
        st.info(f"No data for {area}.")
    else:
        # Important: 'values' must match an existing column in the DataFrame
        fig = px.pie(
            pie_df,
            values="quantityKwh",
            names="productionGroup",
            hole=0.0,
            title=f"Total production 2021 – {area}",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

with right:
    # Group selection as "pills" (default: all selected)
    all_groups = list_groups(price_area=area)
    # st.pills exists in newer Streamlit versions — fall back to multiselect if unavailable
    try:
        selected_groups = st.pills("Select production groups:", options=all_groups, default=all_groups)
        if selected_groups is None:
            selected_groups = all_groups
    except Exception:
        selected_groups = st.multiselect("Select production groups:", options=all_groups, default=all_groups)

    # Select month
    month_names = [calendar.month_name[m] for m in range(1, 13)]
    mlabel = st.selectbox("Choose month:", options=month_names, index=0)
    month = month_names.index(mlabel) + 1

    # Fetch time series for the selected month and groups
    line_df = fetch_line_df(area, selected_groups, month, year=2021).copy()

    if line_df.empty:
        st.info(f"No data for {area} in {mlabel}.")
    else:
        # Plot: one line per production group
        fig2 = px.line(
            line_df,
            x="startTime",
            y="quantityKwh",
            color="productionGroup",
            title=f"Timeproduksjon – {area}, {mlabel} 2021",
        )
        fig2.update_layout(xaxis_title="Tid (UTC)", yaxis_title="kWh")
        st.plotly_chart(fig2, use_container_width=True)

# Documentation section (translated to English)
with st.expander("Sources and method"):
    st.markdown(
        """
- **Source:** Elhub API – dataset `PRODUCTION_PER_GROUP_MBA_HOUR` (2021).
- Raw data is fetched, normalized, and stored in **Cassandra**.
- Then extracted with **Spark** to the columns: `priceArea`, `productionGroup`, `startTime`, `quantityKwh`.
- The same data is loaded into **MongoDB** (`elhub.production_2021_by_hour`) and displayed here.
- Pie shows **total** per group for the selected price area (all of 2021).
- Line diagram shows **hours** for the selected month, price area, and groups.
"""
    )
