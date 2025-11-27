# pages/4_Elhub_Production.py

import calendar
import streamlit as st
import plotly.express as px

from utils_elhub import (
    get_client,
    list_price_areas,
    list_groups,
    fetch_pie_df,
    fetch_line_df,
)

# uri_preview is optional in some builds
try:
    from utils_elhub import uri_preview
except Exception:
    def uri_preview():
        return "uri_preview() not available in this build"


# --- Page title ---
st.title("Elhub Production – 2021")


# --- MongoDB connection status ---
with st.status("Testing MongoDB connection", expanded=False):
    try:
        cli = get_client()
        cnt = cli["elhub"]["production_2021_by_hour"].count_documents({})
        st.success(f"Connected to MongoDB ✅  Documents: {cnt:,}")
    except Exception as e:
        st.error("Cannot see that it is connected to MongoDB.")
        st.exception(e)
        st.stop()


# --- Price area selection (shared via session_state) ---
areas = list_price_areas()
if not areas:
    st.error(
        "Didn't find any priceArea in MongoDB. "
        "Check that you have loaded data into elhub.production_2021_by_hour."
    )
    st.stop()

default_area = st.session_state.get("price_area", areas[0])

st.subheader("Filter: price area, groups and month")
area = st.radio(
    "Select the price area",
    areas,
    index=(areas.index(default_area) if default_area in areas else 0),
    horizontal=True,
)

st.session_state["price_area"] = area
st.caption(f"Active price area: **{area}** (available to other pages)")


# --- Layout: left = pie (annual total), right = time series (monthly) ---
left, right = st.columns(2)

# -------------------- LEFT: PIE CHART (TOTAL 2021) --------------------
with left:
    st.markdown("#### Total production per group (2021)")

    # Fetch data for the pie chart
    pie_df = fetch_pie_df(area).copy()

    # Failsafe: earlier variants used 'totalKwh' instead of 'quantityKwh'
    if "totalKwh" in pie_df.columns and "quantityKwh" not in pie_df.columns:
        pie_df.rename(columns={"totalKwh": "quantityKwh"}, inplace=True)

    if pie_df.empty:
        st.info(f"No data for {area}.")
    else:
        fig = px.pie(
            pie_df,
            values="quantityKwh",
            names="productionGroup",
            hole=0.0,
            title=f"Total production 2021 – {area}",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)


# -------------------- RIGHT: LINE PLOT (HOURLY, MONTH) --------------------
with right:
    st.markdown("#### Hourly production by group (selected month)")

    # Group selection as "pills" (default: all selected)
    all_groups = list_groups(price_area=area)

    # st.pills exists in newer Streamlit versions — fall back to multiselect if unavailable
    try:
        selected_groups = st.pills(
            "Select production groups:", options=all_groups, default=all_groups
        )
        if selected_groups is None:
            selected_groups = all_groups
    except Exception:
        selected_groups = st.multiselect(
            "Select production groups:", options=all_groups, default=all_groups
        )

    # Select month
    month_names = [calendar.month_name[m] for m in range(1, 13)]
    mlabel = st.selectbox("Choose month:", options=month_names, index=0)
    month = month_names.index(mlabel) + 1

    # Fetch time series for the selected month and groups
    line_df = fetch_line_df(area, selected_groups, month, year=2021).copy()

    if line_df.empty:
        st.info(f"No data for {area} in {mlabel}.")
    else:
        fig2 = px.line(
            line_df,
            x="startTime",
            y="quantityKwh",
            color="productionGroup",
            title=f"Hourly production – {area}, {mlabel} 2021",
        )
        fig2.update_layout(xaxis_title="Time (UTC)", yaxis_title="kWh")
        st.plotly_chart(fig2, use_container_width=True)


