# pages/4_Elhub_Production.py
import calendar
import streamlit as st
import pandas as pd
import plotly.express as px

from utils_elhub import (
    list_price_areas,
    list_groups,
    fetch_pie_df,
    fetch_line_df,
)

st.title("Elhub Production – 2021")

# --- UI: prisområdevalg (radio i venstre kolonne) ---
areas = list_price_areas()
if not areas:
    st.error("Fant ingen priceArea i MongoDB. Sjekk at du har lastet inn data til elhub.production_2021_by_hour.")
    st.stop()

left, right = st.columns(2)

with left:
    area = st.radio("Velg prisområde:", areas, index=0, horizontal=True)

    # Hent data for pie
    pie_df = fetch_pie_df(area).copy()

    # Failsafe: noen eldre varianter kan ha 'totalKwh'
    if "totalKwh" in pie_df.columns and "quantityKwh" not in pie_df.columns:
        pie_df.rename(columns={"totalKwh": "quantityKwh"}, inplace=True)

    if pie_df.empty:
        st.info(f"Ingen data for {area}.")
    else:
        # Viktig: values må matche kolonnenavnet i DataFrame
        fig = px.pie(
            pie_df,
            values="quantityKwh",
            names="productionGroup",
            hole=0.0,
            title=f"Total produksjon 2021 – {area}",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

with right:
    # Groups-pills (default: alle valgt)
    all_groups = list_groups(price_area=area)
    # st.pills finnes i nye Streamlit-versjoner – fallback til multiselect hvis ikke
    try:
        selected_groups = st.pills("Velg produksjonsgrupper:", options=all_groups, default=all_groups)
        if selected_groups is None:
            selected_groups = all_groups
    except Exception:
        selected_groups = st.multiselect("Velg produksjonsgrupper:", options=all_groups, default=all_groups)

    # Velg måned
    month_names = [calendar.month_name[m] for m in range(1, 13)]
    mlabel = st.selectbox("Velg måned:", options=month_names, index=0)
    month = month_names.index(mlabel) + 1

    # Hent timeserie for valgt måned
    line_df = fetch_line_df(area, selected_groups, month, year=2021).copy()

    if line_df.empty:
        st.info(f"Ingen data for {area} i {mlabel}.")
    else:
        # Plot: egen linje per gruppe
        fig2 = px.line(
            line_df,
            x="startTime",
            y="quantityKwh",
            color="productionGroup",
            title=f"Timeproduksjon – {area}, {mlabel} 2021",
        )
        fig2.update_layout(xaxis_title="Tid (UTC)", yaxis_title="kWh")
        st.plotly_chart(fig2, use_container_width=True)

# Dokumentasjon under
with st.expander("Kilde og metode"):
    st.markdown(
        """
- **Kilde:** Elhub API – dataset `PRODUCTION_PER_GROUP_MBA_HOUR` (2021).
- Rådata hentes, normaliseres og lagres i **Cassandra**.  
- Deretter ekstraheres med **Spark** til kolonnene: `priceArea`, `productionGroup`, `startTime`, `quantityKwh`.  
- De samme dataene lastes inn i **MongoDB** (`elhub.production_2021_by_hour`) og vises her.  
- Pie viser **total** per gruppe for valgt prisområde (hele 2021).  
- Linjediagram viser **timer** for valgt måned, prisområde og grupper.
"""
    )
