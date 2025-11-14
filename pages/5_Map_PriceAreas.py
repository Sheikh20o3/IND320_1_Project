import json
import folium
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from streamlit_folium import st_folium

from utils_elhub import (
    list_price_areas,
    list_groups,
    fetch_mean_for_period  # <-- du lager denne i utils_elhub
)

st.set_page_config(page_title="Map – Price Areas", page_icon="🗺️", layout="wide")
st.title("Map of Norwegian Price Areas (NO1–NO5)")

# 1. Load GeoJSON
GEOJSON_PATH = "data/elspot_areas.geojson"

with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson = json.load(f)

# 2. User selects: production/consumption + group + date range
mode = st.radio("Dataset", ["Production", "Consumption"], horizontal=True)

# Available price areas:
areas = list_price_areas()

# Choose group:
group = st.selectbox("Energy group", list_groups(price_area=areas[0]), index=0)

# Time interval (days):
colA, colB = st.columns(2)
with colA:
    start_date = st.date_input("Start date", value=date(2021, 1, 1))
with colB:
    end_date = st.date_input("End date", value=date(2021, 1, 15))

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# 3. Fetch mean value per price area (Mongo)
# Expect DataFrame with: priceArea, meanValue
df_stats = fetch_mean_for_period(
    mode=mode.lower(),          # "production" or "consumption"
    group=group,
    start=start_date,
    end=end_date
)

if df_stats.empty:
    st.warning("No data returned for the selected interval.")
    st.stop()

# Map from priceArea → meanValue
value_map = df_stats.set_index("priceArea")["meanValue"].to_dict()

st.info(f"Fetched mean values for {mode.lower()} in group **{group}**")

# 4. Build map
center = [64.5, 11.0]
m = folium.Map(location=center, zoom_start=4.7, tiles="CartoDB positron")

# Default selected area from session_state
selected_area = st.session_state.get("price_area", "NO1")

def style_function(feature):
    area_code = feature["properties"].get("priceArea")

    # Outline if selected area
    if area_code == selected_area:
        outline_color = "red"
        weight = 4
    else:
        outline_color = "black"
        weight = 1

    # Choropleth fill:
    val = value_map.get(area_code)
    if val is None:
        fill_color = "#cccccc"
    else:
        # Normalize manually for simple color mapping
        vmax = max(value_map.values())
        vmin = min(value_map.values())
        norm = (val - vmin) / (vmax - vmin + 0.00001)

        # Blue → red scale
        r = int(255 * norm)
        b = int(255 * (1 - norm))
        fill_color = f"rgba({r},0,{b},0.45)"  # transparent

    return {
        "fillColor": fill_color,
        "color": outline_color,
        "weight": weight,
        "fillOpacity": 0.45,
    }

folium.GeoJson(
    geojson,
    name="Elspot Areas",
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=["priceArea"], aliases=["Area:"]),
).add_to(m)

# 5. Click handling
returned = st_folium(m, height=600, width="100%", returned_objects=["last_clicked"])

if returned and returned.get("last_clicked"):
    lat = returned["last_clicked"]["lat"]
    lon = returned["last_clicked"]["lng"]

    st.session_state["map_coord"] = (lat, lon)

    st.success(f"Selected coordinate: **{lat:.4f}, {lon:.4f}**")
else:
    st.caption("Click anywhere on the map to choose a coordinate.")

# 6. Debug info
with st.expander("Debug info"):
    st.markdown("**Fetched area statistics:**")
    st.dataframe(df_stats, use_container_width=True)

    st.markdown("**Stored coordinate:**")
    st.write(st.session_state.get("map_coord"))
